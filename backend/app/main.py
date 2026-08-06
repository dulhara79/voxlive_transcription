"""
VoxLive backend — application assembly only.

WHAT THIS FILE IS FOR
---------------------
Creating the FastAPI app, running start-up/shutdown, installing middleware,
and registering routes. Nothing else.

Everything that used to live here now has an owner:

    api/routes_ws.py        "How do I communicate with the browser?"
    api/routes_health.py    "Is this process alive / should it take traffic?"
    session/state.py        "What is happening in this one session?"
    session/manager.py      "Which sessions exist?"
    asr/scheduler.py        "How much ASR work may execute?"
    diarization/scheduler.py"How much diarization work may execute?"
    observability/logging.py"How do I record operational events?"

START-UP ORDER MATTERS
----------------------
    process starts
      -> logging configured        (so every later line is structured)
      -> configuration profile selected from APP_ENV
      -> schedulers started        (capacity exists before any session can)
      -> ASR provider constructed
      -> diarization model loaded and warmed
      -> app.state.warm = True
      -> /ready returns 200
      -> ALB begins sending traffic

`/ready` stays 503 for the whole warm-up, so an ECS task cannot receive a user
while WeSpeaker is still loading.

Run (development):
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Run (production — never --reload; it watches the filesystem and restarts the
process, dropping every live WebSocket with it):
    APP_ENV=production uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes_health import router as health_router
from .api.routes_ws import router as ws_router
from .asr.postprocess import PostProcessor
from .asr.scheduler import ASRScheduler
from .config import APP_ENV, settings
from .diarization import scheduler as diar_scheduler
from .diarization.factory import warmup_backend
from .observability.logging import configure_logging
from .session.manager import SessionManager

log = logging.getLogger("voxlive")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def build_provider():
    """Construct the configured ASR provider once, at start-up.

    Building a client per request re-resolves credentials and re-opens the
    connection pool, which shows up directly in tail latency.
    """
    from .asr.gemini_provider import GeminiProvider

    log.info("ASR provider: Gemini (%s)", settings.gemini_model)
    return GeminiProvider(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        allowed_languages=settings.allowed_languages,
        use_vertex=settings.gemini_use_vertex,
        project=settings.google_cloud_project,
        location=settings.google_cloud_location,
        max_words_per_sec=settings.max_words_per_sec,
        thinking_budget=settings.gemini_thinking_budget,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(
        level=os.getenv("LOG_LEVEL", "INFO"),
        json_format=os.getenv("LOG_JSON", "true").lower() in ("1", "true", "yes", "on"),
    )
    log.info("starting VoxLive (profile=%s)", APP_ENV)

    app.state.warm = False
    app.state.sessions = SessionManager()
    app.state.provider = build_provider()
    app.state.postproc = PostProcessor()

    app.state.asr_scheduler = ASRScheduler(
        provider=app.state.provider,
        max_concurrency=_int("ASR_MAX_CONCURRENCY", 6),
        queue_maxsize=_int("ASR_QUEUE_MAXSIZE", 64),
        timeout_sec=_float("ASR_TIMEOUT_SEC", 30.0),
    )
    await app.state.asr_scheduler.start()

    app.state.diar_scheduler = diar_scheduler.configure(
        max_concurrency=_int("DIARIZATION_MAX_CONCURRENCY", 2),
        queue_maxsize=_int("DIARIZATION_QUEUE_MAXSIZE", 32),
    )

    await warmup_backend(settings)

    app.state.warm = True
    log.info("VoxLive ready.")
    try:
        yield
    finally:
        # SIGTERM path. ECS replaces tasks routinely, so this runs often.
        app.state.warm = False
        await app.state.sessions.shutdown_all()
        await app.state.asr_scheduler.stop()
        diar_scheduler.shutdown()
        log.info("VoxLive stopped.")


def create_app() -> FastAPI:
    app = FastAPI(title="VoxLive", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router)
    app.include_router(ws_router)
    return app


app = create_app()
