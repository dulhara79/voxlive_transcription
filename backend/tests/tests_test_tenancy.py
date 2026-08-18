"""Multi-tenancy: isolation, admission control, and quota enforcement.

These pin down the supervisor's §5 rule — "backend refuses ORG_002" — and the
§12/§13 admission behaviour. They use the real repository, the real quota
manager and the real WebSocket route; only the ASR provider is faked.
"""

import asyncio
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("DIARIZATION_MODE", "off")
os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("APP_ENV", "development")

from app.auth.context import TenantContext
from app.auth.principal import (
    AuthenticationError,
    AuthorizationError,
    DevPrincipalResolver,
    build_resolver,
)
from app.tenant.models import (
    PLANS,
    Organization,
    OrganizationStatus,
    User,
    UserRole,
)
from app.tenant.quotas import (
    AdmissionController,
    CapacityManager,
    QuotaManager,
    RejectReason,
)
from app.tenant.repository import InMemoryTenantRepository
from app.tenant.seed import seed_development_tenants

SR = 16000


async def _repo():
    repo = InMemoryTenantRepository()
    await seed_development_tenants(repo)
    return repo


# --------------------------------------------------------------- isolation


def test_cross_tenant_user_lookup_returns_none():
    """ACME's user must not be reachable through Globex's organization id."""

    async def run():
        repo = await _repo()
        same = await repo.get_user("usr_dev_alice", "org_dev_acme")
        cross = await repo.get_user("usr_dev_alice", "org_dev_globex")
        return same, cross

    same, cross = asyncio.run(run())
    assert same is not None and same.email == "alice@acme.example"
    assert cross is None, "cross-tenant lookup leaked a user"


def test_context_refuses_mismatched_pair():
    """Even if a caller hand-assembles the pair, the context refuses it."""
    org = Organization(id="org_a", name="A")
    user = User(id="usr_b", organization_id="org_b", email="b@b.example")
    with pytest.raises(ValueError):
        TenantContext.from_records(org, user)


def test_dev_resolver_refuses_cross_tenant_identity():
    async def run():
        repo = await _repo()
        r = DevPrincipalResolver(repo, "development")
        ok = await r.resolve(
            None, organization_id="org_dev_acme", user_id="usr_dev_alice"
        )
        try:
            await r.resolve(
                None, organization_id="org_dev_globex", user_id="usr_dev_alice"
            )
            return ok, None
        except AuthenticationError as exc:
            return ok, exc

    ok, err = asyncio.run(run())
    assert ok.organization_id == "org_dev_acme"
    assert isinstance(err, AuthenticationError)


def test_suspended_organization_is_refused():
    async def run():
        repo = await _repo()
        r = DevPrincipalResolver(repo, "development")
        try:
            await r.resolve(
                None, organization_id="org_dev_suspended", user_id="usr_dev_erin"
            )
        except AuthorizationError as exc:
            return exc
        return None

    assert isinstance(asyncio.run(run()), AuthorizationError)


def test_log_fields_exclude_email():
    """Personal data must not reach CloudWatch."""

    async def run():
        repo = await _repo()
        r = DevPrincipalResolver(repo, "development")
        return await r.resolve(
            None, organization_id="org_dev_acme", user_id="usr_dev_alice"
        )

    fields = asyncio.run(run()).log_fields()
    assert fields["organization_id"] == "org_dev_acme"
    assert "alice@acme.example" not in str(fields)


# ------------------------------------------------------------ fail-closed


@pytest.mark.parametrize("env", ["staging", "production"])
def test_dev_resolver_cannot_be_built_outside_development(env):
    repo = InMemoryTenantRepository()
    with pytest.raises(RuntimeError):
        DevPrincipalResolver(repo, env)
    with pytest.raises(RuntimeError):
        build_resolver(repo, env)


# ----------------------------------------------------------------- quotas


def test_plan_limit_and_override():
    org = Organization(id="o", name="n", plan_code="starter")
    assert org.concurrent_session_limit == PLANS["starter"].max_concurrent_sessions == 5
    org.max_concurrent_sessions = 40
    assert org.concurrent_session_limit == 40


def test_quota_blocks_at_plan_limit():
    """Globex is on Starter (5). The 6th concurrent session is refused."""

    async def run():
        repo = await _repo()
        live = {"n": 0}
        q = QuotaManager(repo, count_for_organization=lambda _: live["n"])
        ctx = await DevPrincipalResolver(repo, "development").resolve(
            None, organization_id="org_dev_globex", user_id="usr_dev_dave"
        )
        out = []
        for n in (0, 4, 5, 9):
            live["n"] = n
            out.append(await q.check(ctx))
        return out

    at0, at4, at5, at9 = asyncio.run(run())
    assert at0.allowed and at4.allowed
    assert not at5.allowed and at5.reason is RejectReason.QUOTA_EXCEEDED
    assert at5.limit == 5 and at5.retry_after_sec == 30
    assert not at9.allowed


def test_one_tenant_cannot_starve_another():
    """§13: ACME saturating its quota must not affect Globex."""

    async def run():
        repo = await _repo()
        counts = {"org_dev_acme": 25, "org_dev_globex": 0}
        q = QuotaManager(repo, count_for_organization=lambda o: counts[o])
        r = DevPrincipalResolver(repo, "development")
        acme = await r.resolve(
            None, organization_id="org_dev_acme", user_id="usr_dev_alice"
        )
        globex = await r.resolve(
            None, organization_id="org_dev_globex", user_id="usr_dev_dave"
        )
        return await q.check(acme), await q.check(globex)

    acme, globex = asyncio.run(run())
    assert not acme.allowed, "ACME should be at its Business limit of 25"
    assert globex.allowed, "Globex was starved by another tenant"


def test_platform_capacity_rejects_before_quota():
    """§12: at capacity, reject gracefully — and platform is checked first."""

    async def run():
        repo = await _repo()
        cap = CapacityManager(max_sessions=10, count_fn=lambda: 10)
        q = QuotaManager(repo, count_for_organization=lambda _: 0)
        ctrl = AdmissionController(cap, q)
        ctx = await DevPrincipalResolver(repo, "development").resolve(
            None, organization_id="org_dev_acme", user_id="usr_dev_alice"
        )
        return await ctrl.admit(ctx)

    d = asyncio.run(run())
    assert not d.allowed
    assert d.reason is RejectReason.PLATFORM_FULL
    assert d.retry_after_sec == 15


def test_viewer_authenticates_but_cannot_start_a_session():
    """A VIEWER is a valid identity — they just may not run a session.

    Authentication and per-action authorization are separate: the resolver
    returns a context, and the route refuses the action.
    """

    async def run():
        repo = await _repo()
        return await DevPrincipalResolver(repo, "development").resolve(
            None, organization_id="org_dev_acme", user_id="usr_dev_carol"
        )

    ctx = asyncio.run(run())
    assert ctx.role is UserRole.VIEWER
    assert ctx.organization_id == "org_dev_acme"
    assert not ctx.can_start_session


def test_ws_refuses_viewer_role(client):
    """And the route enforces it: close 1008, no session created."""
    from starlette.websockets import WebSocketDisconnect

    c, app = client
    with pytest.raises(WebSocketDisconnect) as exc:
        with c.websocket_connect(
            "/ws/transcribe?organization_id=org_dev_acme&user_id=usr_dev_carol"
        ) as ws:
            ws.receive_json()
    assert exc.value.code == 1008
    assert app.state.sessions.count_for_organization("org_dev_acme") == 0


# ----------------------------------------------------- end-to-end WebSocket


def _tone(seconds, freq=180.0, amp=9000):
    n = int(SR * seconds)
    t = np.arange(n) / SR
    w = (
        amp * np.sin(2 * math.pi * freq * t)
        + amp * 0.5 * np.sin(2 * math.pi * freq * 2 * t)
        + amp * 0.3 * np.sin(2 * math.pi * freq * 3 * t)
    )
    return w.astype(np.int16).tobytes()


def _silence(seconds):
    return np.zeros(int(SR * seconds), dtype=np.int16).tobytes()


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    from app.asr.gemini_provider import ASRResult
    from app.main import app

    class FakeProvider:
        def __init__(self):
            self.calls = 0

        async def transcribe_segment(self, pcm, sample_rate, context=None):
            self.calls += 1
            return ASRResult(text=f"utterance {self.calls}", language="en")

    with TestClient(app) as c:
        app.state.asr_scheduler.provider = FakeProvider()
        yield c, app


def test_ws_rejects_unknown_organization(client):
    from starlette.websockets import WebSocketDisconnect

    c, _ = client
    with pytest.raises(WebSocketDisconnect) as exc:
        with c.websocket_connect(
            "/ws/transcribe?organization_id=org_nope&user_id=usr_dev_alice"
        ) as ws:
            ws.receive_json()
    assert exc.value.code == 1008


def test_ws_rejects_cross_tenant_user(client):
    from starlette.websockets import WebSocketDisconnect

    c, _ = client
    with pytest.raises(WebSocketDisconnect) as exc:
        with c.websocket_connect(
            "/ws/transcribe?organization_id=org_dev_globex&user_id=usr_dev_alice"
        ) as ws:
            ws.receive_json()
    assert exc.value.code == 1008


def test_ws_rejects_when_platform_full(client):
    c, app = client
    original = app.state.admission.capacity.max_sessions
    app.state.admission.capacity.max_sessions = 1
    app.state.admission.capacity._count = lambda: 1
    try:
        with c.websocket_connect(
            "/ws/transcribe?organization_id=org_dev_acme&user_id=usr_dev_alice"
        ) as ws:
            msg = ws.receive_json()
    finally:
        app.state.admission.capacity.max_sessions = original
        app.state.admission.capacity._count = app.state.sessions.count
    assert msg["type"] == "rejected"
    assert msg["reason"] == "platform_full"
    assert msg["retry_after_sec"] == 15


def test_ws_happy_path_carries_tenant_context(client):
    c, app = client
    with c.websocket_connect(
        "/ws/transcribe?organization_id=org_dev_acme&user_id=usr_dev_alice&speakers=2"
    ) as ws:
        assert ws.receive_json() == {"type": "status", "state": "ready"}

        assert app.state.sessions.count_for_organization("org_dev_acme") == 1
        assert app.state.sessions.count_for_organization("org_dev_globex") == 0
        snap = app.state.sessions.snapshot()[0]
        assert snap["organization_id"] == "org_dev_acme"
        assert snap["user_id"] == "usr_dev_alice"
        assert snap["plan"] == "business"

        audio = _tone(1.6) + _silence(0.9)
        step = SR * 2 // 10
        for i in range(0, len(audio), step):
            ws.send_bytes(audio[i : i + step])
        ws.send_text("stop")

        seen = []
        for _ in range(40):
            m = ws.receive_json()
            seen.append(m)
            if m.get("type") == "status" and m.get("state") == "stopped":
                break

    assert any(m["type"] == "transcript" for m in seen)
    assert app.state.sessions.count_for_organization("org_dev_acme") == 0


def test_metrics_reports_per_organization(client):
    c, _ = client
    with c.websocket_connect(
        "/ws/transcribe?organization_id=org_dev_acme&user_id=usr_dev_bob"
    ) as ws:
        ws.receive_json()
        body = c.get("/metrics").json()
        assert body["sessions_by_organization"]["org_dev_acme"] == 1
        assert body["admission"]["platform"]["active_sessions"] == 1
