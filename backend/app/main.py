"""
VoxLive backend — application assembly only.

WHAT THIS FILE IS FOR
---------------------
Creating the FastAPI app, running start-up/shutdown, installing middleware,
and registering routes. Nothing else. Production `main.py` files that also
contain VAD, ASR retries, queues, the WebSocket protocol and session lifecycle
are hard to test and harder to operate at 3 a.m.

Everything that used to live here now has an owner:

    api/routes_ws.py      "How do I communicate with the browser?"
    api/routes_health.py  "Is this process alive / should it take traffic?"
    session/state.py      "What is happening in this one session?"
    session/manager.py    "Which sessions exist?"
    asr/                  "How is speech turned into text?"
    diarization/          "Who is speaking?"

START-UP ORDER MATTERS
----------------------
    process starts
      -> configuration loaded
      -> ASR provider constructed
      -> diarization model loaded and warmed
      -> app.state.warm = True
      -> /ready returns 200
      -> ALB begins sending traffic

`/ready` stays 503 for the whole warm-up, so an ECS task cannot receive a user
while WeSpeaker is still loading.

Run (development):
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Run (production — no --reload, no auto-restart on file writes):
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes_health import router as health_router
from .api.routes_ws import router as ws_router
from .asr.postprocess import PostProcessor
from .config import settings
from .diarization.factory import warmup_backend
from .session.manager import SessionManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("voxlive")


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
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.warm = False
    app.state.sessions = SessionManager()
    app.state.provider = build_provider()
    app.state.postproc = PostProcessor()

    await warmup_backend(settings)

    app.state.warm = True
    log.info("VoxLive ready.")
    try:
        yield
    finally:
        # SIGTERM path. ECS replaces tasks routinely, so this runs often.
        await app.state.sessions.shutdown_all()
        app.state.warm = False
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
