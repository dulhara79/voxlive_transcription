"""
manager.py — SessionManager: which sessions exist in THIS process.

It answers exactly one question:

    "Which sessions exist right now, in this backend process?"

SCOPE — READ THIS BEFORE ADDING REDIS
-------------------------------------
This registry is deliberately IN-MEMORY and PROCESS-LOCAL. That is not a
temporary shortcut, it is the correct first architecture:

    ALB ──┬──► ECS task 1 ──► SessionManager (its own sessions)
          ├──► ECS task 2 ──► SessionManager (its own sessions)
          └──► ECS task 3 ──► SessionManager (its own sessions)

A live WebSocket is physically bound to the process that accepted it, so the
process that owns the socket is the only one that can act on the session. A
shared Redis registry would not change that; it would only let OTHER tasks
observe sessions they cannot touch. Redis becomes genuinely useful later, for
fleet-wide counts, cross-task rate limiting and resume-after-reconnect — and
by then we will know exactly which fields actually need to be shared.

WHAT THIS UNLOCKS NEXT
----------------------
`count()` is the input to admission control, and `snapshot()` is the input to
the CloudWatch `active_sessions` metric. Neither is wired up yet — this commit
only establishes the registry.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from fastapi import WebSocket

from ..asr.scheduler import ASRScheduler
from ..auth.context import TenantContext
from .state import SessionState, new_session_id

log = logging.getLogger("voxlive.sessions")


class SessionManager:
    """Registry of the live sessions owned by this process.

    One instance per application, created in the lifespan and reachable as
    `app.state.sessions`. All mutation is guarded by an asyncio.Lock: the
    dict operations themselves are atomic under the GIL, but create/remove
    pair with logging and counting that should not interleave.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        # organization_id -> {session_id}. Maintained alongside _sessions so
        # a per-tenant count is O(1) on the admission path rather than a scan
        # of every live session on the task.
        self._by_org: dict[str, set[str]] = {}
        self._lock = asyncio.Lock()

    # --------------------------------------------------------------- create

    async def create(
        self,
        ws: WebSocket,
        tenant: TenantContext,
        expected_speakers: int,
        asr_scheduler: ASRScheduler,
        postproc: Any,
        session_id: Optional[str] = None,
    ) -> SessionState:
        """Build and register a session. Does not start it — the caller does
        that, so a failure to register can never leave a running worker
        orphaned."""
        sid = session_id or new_session_id()
        session = SessionState(
            ws=ws,
            session_id=sid,
            tenant=tenant.with_session(sid),
            expected_speakers=expected_speakers,
            asr_scheduler=asr_scheduler,
            postproc=postproc,
        )
        async with self._lock:
            self._sessions[sid] = session
            self._by_org.setdefault(tenant.organization_id, set()).add(sid)
            total = len(self._sessions)
            for_org = len(self._by_org[tenant.organization_id])
        log.info(
            "session_started (expected_speakers=%s, active=%d, org_active=%d)",
            expected_speakers or "auto",
            total,
            for_org,
            extra={"session_id": sid, **tenant.log_fields()},
        )
        return session

    # ---------------------------------------------------------------- query

    def get(self, session_id: str) -> Optional[SessionState]:
        return self._sessions.get(session_id)

    def count(self) -> int:
        """Live session count. Reads without the lock on purpose: this is
        called on the metrics/health path and a value that is one session
        stale is fine, whereas contending with connect/disconnect is not."""
        return len(self._sessions)

    def count_for_organization(self, organization_id: str) -> int:
        """Live sessions for one tenant, on THIS task.

        Lock-free like `count()`: it feeds the admission check, where being
        one session stale is acceptable and contending with every connect and
        disconnect is not. The race window admits at most a small overshoot,
        which is the right trade against serialising admissions.
        """
        return len(self._by_org.get(organization_id, ()))

    def snapshot(self) -> list[dict]:
        return [s.snapshot() for s in list(self._sessions.values())]

    def organization_breakdown(self) -> dict[str, int]:
        """Live session count per organization — the per-tenant CloudWatch
        dimension, and the first thing to look at when one customer is
        consuming the task."""
        return {org: len(ids) for org, ids in self._by_org.items() if ids}

    # --------------------------------------------------------------- remove

    async def remove(self, session_id: str) -> Optional[SessionState]:
        """Deregister a session. Does NOT close it — closing is the owner's
        job, and separating the two keeps this method safe to call from a
        `finally` block."""
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            if session is not None:
                bucket = self._by_org.get(session.organization_id)
                if bucket is not None:
                    bucket.discard(session_id)
                    if not bucket:
                        # Drop empty buckets, or _by_org grows one permanent
                        # entry per organization that ever connected.
                        del self._by_org[session.organization_id]
            total = len(self._sessions)
        if session is not None:
            log.info(
                "session_closed (duration_sec=%.1f, segments=%d, active=%d)",
                session.snapshot()["age_sec"],
                session.seg_id,
                total,
                extra={"session_id": session_id, **session.tenant.log_fields()},
            )
        return session

    # ------------------------------------------------------------- shutdown

    async def shutdown_all(self) -> None:
        """Close every session. Called from the lifespan on SIGTERM.

        This is the seed of ECS connection draining: today it closes sessions
        promptly; the graceful-shutdown commit will first stop admitting new
        ones, then notify clients, then wait out a drain deadline.
        """
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
            self._by_org.clear()
        if not sessions:
            return
        log.info("shutting down %d active session(s)", len(sessions))
        results = await asyncio.gather(
            *(s.aclose() for s in sessions), return_exceptions=True
        )
        for session, result in zip(sessions, results):
            if isinstance(result, Exception):
                log.error(
                    "error closing session: %s",
                    result,
                    extra={"session_id": session.session_id},
                )
