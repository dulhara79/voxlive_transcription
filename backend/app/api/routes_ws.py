"""
routes_ws.py — the /ws/transcribe endpoint.

It answers exactly one question:

    "How do I communicate with the browser?"

The supervisor's §16 defines the order of operations at this boundary, and
this file implements it exactly:

    Browser
       │  WSS
       ▼
    WebSocket Route      <- this file
       │
       ▼
    Authentication       auth/principal.py   -> TenantContext
       │
       ▼
    Tenant Context       auth/context.py     -> organization_id, user_id, role
       │
       ▼
    Quota Check          tenant/quotas.py    -> platform capacity, then quota
       │
       ▼
    Session Manager      session/manager.py
       │
       ├──► ASR Scheduler
       └──► Diarization

Everything about WHAT to do with the audio belongs to SessionState. Nothing in
this file should grow into processing logic.

WHY THE SOCKET IS ACCEPTED BEFORE AUTHENTICATION
------------------------------------------------
A browser's WebSocket API cannot set request headers, and it surfaces a
pre-handshake rejection to JavaScript as an indistinguishable "error" event —
the page cannot tell "wrong password" from "server down". So the handshake is
accepted, the credential is checked immediately, and a failure is reported as
a close frame with a specific code and a readable reason. Nothing the client
sends is processed before `resolve()` succeeds.

CLOSE CODES
-----------
    1008  policy violation  — authentication or authorization failed
    1013  try again later   — platform full or tenant quota exhausted
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..auth.context import TenantContext, reset_current, set_current
from ..auth.principal import (
    AuthenticationError,
    AuthorizationError,
    PrincipalResolver,
)
from ..config import settings
from ..observability.logging import bind
from ..session.manager import SessionManager
from ..tenant.quotas import AdmissionController
from .schemas import status_msg

log = logging.getLogger("voxlive.ws")

router = APIRouter()

WS_POLICY_VIOLATION = 1008
WS_TRY_AGAIN_LATER = 1013


def _expected_speakers(ws: WebSocket) -> int:
    """`?speakers=N` from the client, if it is a sane integer.

    What N MEANS is decided by `?speaker_mode=` below, not here: a ceiling in
    auto mode, an exact count in fixed mode. A bad value is ignored rather than
    rejected, because a malformed query parameter is not a reason to refuse
    someone's recording.
    """
    raw = ws.query_params.get("speakers")
    if not raw:
        return settings.expected_speakers
    try:
        return max(0, int(raw))
    except ValueError:
        log.warning("ignoring non-integer ?speakers=%r", raw)
        return settings.expected_speakers


def _speaker_mode(ws: WebSocket) -> str:
    """`?speaker_mode=auto|fixed`, defaulting to SPEAKER_MODE from .env.

    auto   estimate the speaker count; `?speakers=N` is a CEILING.
    fixed  the speaker count IS `?speakers=N`. For a recording where the user
           knows the count — a two-person interview — this is what stops the
           clustering stage from deciding, several minutes in, that the two
           voices were similar enough to be one person and relabelling the
           whole transcript.

    `fixed` without a positive `?speakers=` has nothing to fix K to. The engine
    falls back to auto and says so in its own log; the request is not refused,
    because dropping a session over a query-string mistake is worse than
    running it with the safer of the two behaviours.
    """
    raw = (ws.query_params.get("speaker_mode") or "").strip().lower()
    if not raw:
        return settings.speaker_mode
    if raw not in ("auto", "fixed"):
        log.warning("ignoring unknown ?speaker_mode=%r", raw)
        return settings.speaker_mode
    return raw


async def _authenticate(ws: WebSocket) -> Optional[TenantContext]:
    """Resolve the caller, or close the socket and return None.

    The token is read from `?token=` because browsers cannot set an
    Authorization header on a WebSocket. That places the credential in a URL,
    so it MUST be short-lived and the ALB access logs must not record query
    strings — both are deployment requirements for the Cognito commit, not
    optional hardening.
    """
    resolver: PrincipalResolver = ws.app.state.principal_resolver
    try:
        return await resolver.resolve(
            ws.query_params.get("token"),
            organization_id=ws.query_params.get("organization_id"),
            user_id=ws.query_params.get("user_id"),
        )
    except AuthenticationError as exc:
        log.warning("authentication failed: %s", exc)
        await ws.close(code=WS_POLICY_VIOLATION, reason=str(exc)[:120])
    except AuthorizationError as exc:
        log.warning("authorization failed: %s", exc)
        await ws.close(code=WS_POLICY_VIOLATION, reason=str(exc)[:120])
    return None


@router.websocket("/ws/transcribe")
async def transcribe(ws: WebSocket) -> None:
    await ws.accept()

    tenant = await _authenticate(ws)
    if tenant is None:
        return

    if not tenant.can_start_session:
        await ws.close(
            code=WS_POLICY_VIOLATION,
            reason=f"role {tenant.role.value} cannot start sessions",
        )
        return

    # Platform capacity first, then this organization's quota.
    admission: AdmissionController = ws.app.state.admission
    decision = await admission.admit(tenant)
    if not decision.allowed:
        # Send the detail as JSON before closing: the close reason is capped
        # at 123 bytes and clients render it poorly, but a JSON frame lets the
        # UI show the plan limit and a retry countdown.
        await ws.send_json({"type": "rejected", **decision.as_dict()})
        await ws.close(code=WS_TRY_AGAIN_LATER, reason=decision.reason.value)
        log.info(
            "session refused: %s (%d/%d)",
            decision.reason.value,
            decision.current,
            decision.limit,
            extra=tenant.log_fields(),
        )
        return

    manager: SessionManager = ws.app.state.sessions
    session = await manager.create(
        ws=ws,
        tenant=tenant,
        expected_speakers=_expected_speakers(ws),
        speaker_mode=_speaker_mode(ws),
        asr_scheduler=ws.app.state.asr_scheduler,
        postproc=ws.app.state.postproc,
    )

    # Ambient context for anything downstream that is too deep to be handed
    # the object, and log fields for every line emitted under this task.
    token = set_current(session.tenant)
    try:
        with bind(**session.tenant.log_fields()):
            session.start()
            await session.send(status_msg("ready"))

            try:
                while True:
                    msg = await ws.receive()

                    if msg.get("type") == "websocket.disconnect":
                        raise WebSocketDisconnect(msg.get("code", 1000))

                    if msg.get("bytes") is not None:
                        await session.feed_audio(msg["bytes"])
                        if session.limit_reached:
                            # The plan's per-session ceiling was hit and the
                            # transcript has already been drained and sent.
                            # Reading further frames would burn CPU decoding
                            # audio that is guaranteed to be discarded.
                            await ws.close(
                                code=WS_TRY_AGAIN_LATER,
                                reason="session length limit reached",
                            )
                            break

                    elif msg.get("text") == "stop":
                        await session.finish()

                    elif msg.get("text") == "new_recording":
                        # Explicit NEW RECORDING control (supervisor review
                        # §10/§11). Independent speaker identities from here
                        # on, transcript and socket preserved.
                        #
                        # This is a control the USER presses. There is
                        # deliberately no silence-triggered equivalent: a long
                        # pause in a conversation is not a new recording, and
                        # resetting identities there would invent new speakers
                        # mid-interview.
                        await session.new_recording()

            except WebSocketDisconnect:
                log.info(
                    "client disconnected after %d segment(s)",
                    session.seg_id,
                )
            finally:
                await manager.remove(session.session_id)
                await session.aclose()
    finally:
        reset_current(token)
