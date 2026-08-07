"""
routes_health.py — /health and /ready.

WHY TWO ENDPOINTS
-----------------
They answer different questions, and an orchestrator needs both:

  /health  "Is this process alive, and how is it configured?"
           Always 200 while the process is running. Used by humans and by
           ECS/ALB liveness checks. Must never depend on a model being
           loaded, or a slow warm-up looks like a crash and the task gets
           killed in a restart loop.

  /ready   "Should this task receive real users yet?"
           200 only once the ASR provider is constructed and the diarization
           model is warm; 503 until then. Loading WeSpeaker takes seconds, and
           an ECS task that accepts a WebSocket during that window gives its
           first user a stall. The ALB target group should point here.

The `/health` payload is the baseline's, field for field, plus one addition:
`active_sessions`. That is the number the capacity work is about to be built
around, and it costs one `len()`.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status

from ..config import settings
from ..diarization.factory import resolve_backend

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict:
    sessions = getattr(request.app.state, "sessions", None)
    return {
        "status": "ok",
        "provider": settings.provider,
        "model": settings.gemini_model,
        "auth": "vertex" if settings.gemini_use_vertex else "api_key",
        "languages": list(settings.allowed_languages),
        "diarization": (
            resolve_backend(settings.diarization_backend)
            if settings.diarization_enabled
            else "off"
        ),
        "expected_speakers": settings.expected_speakers,
        "max_speakers": settings.max_speakers,
        "diarize_interval_sec": settings.diarize_interval_sec,
        "diarize_wait_ms": settings.diarize_wait_ms,
        "context_segments": settings.context_segments,
        "active_sessions": sessions.count() if sessions is not None else 0,
    }


@router.get("/metrics")
async def metrics(request: Request) -> dict:
    """Scheduler and session counters.

    These are the autoscaling signals CPU alone would miss: CPU can sit at 45%
    while `asr.queue_depth` is 200 and P95 latency is six seconds. The
    CloudWatch commit publishes these; for now they are readable by hand and
    scrapeable by the load-test harness.
    """
    st = request.app.state
    sessions = getattr(st, "sessions", None)
    asr = getattr(st, "asr_scheduler", None)
    diar = getattr(st, "diar_scheduler", None)
    admission = getattr(st, "admission", None)
    return {
        "active_sessions": sessions.count() if sessions is not None else 0,
        # Per-tenant session counts: the CloudWatch dimension that answers
        # "which customer is consuming this task?" during an incident.
        "sessions_by_organization": (
            sessions.organization_breakdown() if sessions is not None else {}
        ),
        "admission": admission.snapshot() if admission is not None else {},
        "asr": asr.snapshot() if asr is not None else {},
        "diarization": diar.snapshot() if diar is not None else {},
    }


@router.get("/ready")
async def ready(request: Request, response: Response) -> dict:
    """Readiness gate for ECS/ALB. 503 until warm-up has completed."""
    app_state = request.app.state
    provider_ok = getattr(app_state, "provider", None) is not None
    warm_ok = bool(getattr(app_state, "warm", False))
    ok = provider_ok and warm_ok

    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "ready": ok,
        "provider": provider_ok,
        "models_warm": warm_ok,
    }
