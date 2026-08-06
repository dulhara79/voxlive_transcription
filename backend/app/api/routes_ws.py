"""
routes_ws.py — the /ws/transcribe endpoint.

It answers exactly one question:

    "How do I communicate with the browser?"

Accept the socket, read the client's parameters, hand bytes to the session,
and clean up when the connection ends. Every decision about WHAT to do with
the audio belongs to SessionState; every decision about which sessions exist
belongs to SessionManager. Nothing in this file should ever grow into
processing logic — if it starts to, that logic belongs in `session/`.

    browser
       │  wss
       ▼
    /ws/transcribe  ── this file
       │
       ▼
    SessionManager.create()
       │
       ▼
    SessionState.feed_audio() / .finish() / .aclose()

NOT YET IMPLEMENTED HERE, ON PURPOSE
------------------------------------
Authentication (Cognito/JWT), tenant resolution and admission control all
belong at this boundary, and all three are later phases. The `accept()` below
is still unconditional, exactly as in the baseline.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..config import settings
from ..session.manager import SessionManager
from .schemas import status_msg

log = logging.getLogger("voxlive.ws")

router = APIRouter()


def _expected_speakers(ws: WebSocket) -> int:
    """`?speakers=N` from the client, if it is a sane integer.

    Treated as a CEILING by the speaker engine, not a quota — a bad value is
    ignored rather than rejected, because a malformed query parameter is not
    a reason to refuse someone's recording.
    """
    raw = ws.query_params.get("speakers")
    if not raw:
        return settings.expected_speakers
    try:
        return max(0, int(raw))
    except ValueError:
        log.warning("ignoring non-integer ?speakers=%r", raw)
        return settings.expected_speakers


@router.websocket("/ws/transcribe")
async def transcribe(ws: WebSocket) -> None:
    await ws.accept()

    manager: SessionManager = ws.app.state.sessions
    session = await manager.create(
        ws=ws,
        expected_speakers=_expected_speakers(ws),
        provider=ws.app.state.provider,
        postproc=ws.app.state.postproc,
    )
    session.start()
    await session.send(status_msg("ready"))

    try:
        while True:
            msg = await ws.receive()

            if msg.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect(msg.get("code", 1000))

            if msg.get("bytes") is not None:
                await session.feed_audio(msg["bytes"])

            elif msg.get("text") == "stop":
                await session.finish()

    except WebSocketDisconnect:
        log.info(
            "client disconnected after %d segment(s)",
            session.seg_id,
            extra={"session_id": session.session_id},
        )
    finally:
        await manager.remove(session.session_id)
        await session.aclose()
