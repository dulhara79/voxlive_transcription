"""
VoxLive backend — Gemini ASR + background speaker diarization (v10).

THE SHAPE OF THE CHANGE
-----------------------
v7 ran, per segment, strictly in order:

    VAD close -> diarize (1-4 s, blocking) -> ASR (1-3 s) -> emit

Diarization sat on the critical path even though it answers a question that
has nothing to do with the words. With a 6-second soft segment cap on top,
the transcript ran 8-13 seconds behind the room.

v10 runs the two independently and joins them on the audio clock:

    VAD close ──► ASR ──────────────┐
                                    ├──► text + speaker ──► client
    raw audio ──► DiarizationService┘
                  (background pass every ~1.5 s)

A diarization pass costs ~150-400 ms because every window of the pass is
embedded in ONE batch; a Gemini call costs 1-3 s. So the speaker timeline is
normally ready before the text is, and joining them is free. When it isn't,
the segment waits a bounded ~900 ms and otherwise ships with a best-effort
label that the next pass corrects via `refresh`.

Nothing here decides a speaker. That is entirely SpeakerEngine's job, run over
the whole session, every pass, from scratch — so there is no greedy decision
to regret and no online centroid to poison.

Run:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import asyncio
import logging
import os
import traceback
from collections import deque
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .audio import Segment, VADSegmenter
from .config import settings
from .diarizer_factory import build_diarizer, resolve_backend, warmup_backend
from .postprocess import PostProcessor
from .schemas import error_msg, refresh_msg, speakers_msg, status_msg
from .transcript import TranscriptStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("voxlive")

ASR_MAX_RETRIES = 3
ASR_CONCURRENCY = int(os.getenv("ASR_CONCURRENCY", "6"))

# Padding around a speaker-turn cut so word onsets/offsets aren't clipped.
CHUNK_PAD_PRE_S = 0.10
CHUNK_PAD_POST_S = 0.15

state: dict = {}


def _rms(pcm: bytes) -> float:
    if not pcm:
        return 0.0
    arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(arr * arr)))


def _is_rate_limit(err) -> bool:
    s = str(err)
    return "429" in s or "RESOURCE_EXHAUSTED" in s


def build_provider():
    from .providers.gemini_provider import GeminiProvider

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
    state["provider"] = build_provider()
    state["postproc"] = PostProcessor()
    await warmup_backend(settings)
    log.info("VoxLive ready.")
    yield
    state.clear()


app = FastAPI(title="VoxLive", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
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
    }


# ---------------------------------------------------------------------------


class Session:
    """Everything one WebSocket connection owns."""

    def __init__(self, ws: WebSocket, expected_speakers: int):
        self.ws = ws
        self.seg = VADSegmenter(
            sample_rate=settings.sample_rate,
            vad_aggressiveness=settings.vad_aggressiveness,
            silence_ms=settings.silence_ms,
            soft_max_segment_ms=settings.soft_max_segment_ms,
            max_segment_ms=settings.max_segment_ms,
            min_segment_ms=settings.min_segment_ms,
        )
        self.store = TranscriptStore()
        self.diar = build_diarizer(settings, expected_speakers)
        self.diar.on_change = self.on_diarization_change

        self.recent: "deque[str] | None" = (
            deque(maxlen=settings.context_segments)
            if settings.context_segments > 0
            else None
        )
        self.asr_slots = asyncio.Semaphore(ASR_CONCURRENCY)
        self.emit_lock = asyncio.Lock()  # serialises store mutation + sends
        self.inflight: set[asyncio.Task] = set()
        self.dead = False
        self.last_speaker_count = 0

    # ------------------------------------------------------------- outbound

    async def send(self, payload: dict) -> None:
        if self.dead:
            return
        try:
            await self.ws.send_json(payload)
        except (WebSocketDisconnect, RuntimeError):
            self.dead = True

    async def publish(self) -> None:
        """Emit whatever changed. Caller must hold emit_lock."""
        paras, mode = self.store.diff()
        if mode == "none":
            return
        if mode == "append":
            await self.send(paras[-1])
        else:
            await self.send(refresh_msg(paras))

        n = self.diar.engine.speaker_count()
        if n and n != self.last_speaker_count:
            self.last_speaker_count = n
            await self.send(speakers_msg(n))

    async def on_diarization_change(self) -> None:
        """A diarization pass moved the timeline: re-derive every label."""
        async with self.emit_lock:
            if self.store.relabel(self.diar.label_for):
                await self.publish()

    # --------------------------------------------------------------- segment

    async def handle_segment(self, segment: Segment, seg_id: int) -> None:
        energy = _rms(segment.pcm)
        if energy < settings.min_segment_rms:
            log.info(
                "segment %d gated: rms=%.0f < %d",
                seg_id,
                energy,
                settings.min_segment_rms,
            )
            return

        log.info(
            "segment %d: %.2f-%.2fs (%.2fs, %s) rms=%.0f",
            seg_id,
            segment.start,
            segment.end,
            segment.duration,
            segment.reason,
            energy,
        )

        # Give the background diarizer a bounded moment to cover this segment,
        # so the FIRST render already carries the right name. A refresh would
        # fix it anyway, so this wait is short and never blocks the pipeline.
        if settings.diarization_enabled and segment.duration >= 1.0:
            await self.diar.wait_for_coverage(
                segment.end, timeout=settings.diarize_wait_ms / 1000.0
            )

        pieces = self._split(segment)

        async with self.emit_lock:
            chunks = [(self.store.new_chunk(seg_id, a, b), pcm) for pcm, a, b in pieces]

        await self.send(status_msg("transcribing"))
        context = " ".join(self.recent) if self.recent else None

        async def one(chunk, pcm):
            async with self.asr_slots:
                result, limited, err = await self._asr(pcm, context)
            await self._emit(chunk, result, limited, err, seg_id)

        await asyncio.gather(*(one(c, p) for c, p in chunks))
        await self.send(status_msg("ready"))

    def _split(self, segment: Segment) -> list[tuple[bytes, float, float]]:
        """Cut a segment at speaker changes found by the diarizer.

        v7 had a bespoke change-point detector here: sliding 1.5 s windows at a
        0.4 s hop, 73% overlapped, cut at local maxima above 0.60. Overlapping
        windows structurally suppress distance except at boundaries, which
        forced the threshold up into the range where one speaker's own vowel
        changes also live — so it cut mid-sentence, and the resulting two-voice
        chunks poisoned the clustering that had to label them.

        The cuts now come from the same clustering that decides identity, so a
        split and its labels cannot disagree.
        """
        sr = settings.sample_rate
        cuts = self.diar.split_points(segment.start, segment.end)
        if not cuts:
            return [(segment.pcm, segment.start, segment.end)]

        bounds = [segment.start, *cuts, segment.end]
        min_bytes = int(sr * settings.min_segment_ms / 1000) * 2
        out: list[tuple[bytes, float, float]] = []
        for a, b in zip(bounds, bounds[1:]):
            a2 = max(segment.start, a - CHUNK_PAD_PRE_S)
            b2 = min(segment.end, b + CHUNK_PAD_POST_S)
            i0 = int((a2 - segment.start) * sr) * 2
            i1 = int((b2 - segment.start) * sr) * 2
            piece = segment.pcm[i0:i1]
            if len(piece) < min_bytes or _rms(piece) < settings.min_segment_rms:
                continue
            out.append((piece, a2, b2))

        if not out:
            return [(segment.pcm, segment.start, segment.end)]
        log.info(
            "segment split into %d turn(s) at %s",
            len(out),
            ", ".join(f"{c:.2f}s" for c in cuts),
        )
        return out

    async def _asr(self, pcm: bytes, context):
        last_err, limited = None, False
        for attempt in range(ASR_MAX_RETRIES):
            try:
                r = await state["provider"].transcribe_segment(
                    pcm, settings.sample_rate, context=context
                )
                return r, False, None
            except Exception as e:  # noqa: BLE001
                if isinstance(e, (WebSocketDisconnect, RuntimeError)):
                    raise
                last_err = e
                if _is_rate_limit(e):
                    limited = True
                    log.error("HTTP 429 from Vertex — backing off")
                    await asyncio.sleep(2**attempt)
                    continue
                log.warning(
                    "ASR attempt %d/%d failed: %s", attempt + 1, ASR_MAX_RETRIES, e
                )
                await asyncio.sleep(0.5 * (attempt + 1))
        return None, limited, last_err

    async def _emit(self, chunk, result, limited, err, seg_id: int) -> None:
        async with self.emit_lock:
            if result is None:
                self.store.drop(chunk)
                await self.send(
                    error_msg(
                        (
                            "Vertex AI rejected the request (HTTP 429). Please retry."
                            if limited
                            else f"transcription failed after {ASR_MAX_RETRIES} retries"
                        ),
                        seg_id,
                    )
                )
                return

            if not result.text or result.language not in settings.allowed_languages:
                if result.text:
                    log.info("dropped segment %d: lang=%s", seg_id, result.language)
                self.store.drop(chunk)
                return

            text = result.text
            if settings.enable_postprocess:
                text = await state["postproc"].process(text, result.language)

            # A long output identical to the previous one is a stuck model,
            # not a person repeating a sentence word for word.
            if self.recent and len(text) > 25 and self.recent[-1] == text:
                log.info("repeat guard: segment %d dropped", seg_id)
                self.store.drop(chunk)
                return
            if self.recent is not None:
                self.recent.append(text)

            chunk.text = text
            chunk.language = result.language
            chunk.speaker = self.diar.label_for(chunk.start, chunk.end)
            if chunk.speaker is None:
                prev = [
                    c
                    for c in self.store.chunks
                    if c.speaker is not None and c.end <= chunk.start
                ]
                chunk.speaker = prev[-1].speaker if prev else 0

            await self.publish()

    async def aclose(self) -> None:
        await self.diar.aclose()


# ---------------------------------------------------------------------------


@app.websocket("/ws/transcribe")
async def transcribe(ws: WebSocket):
    await ws.accept()

    expected = settings.expected_speakers
    raw = ws.query_params.get("speakers")
    if raw:
        try:
            expected = max(0, int(raw))
            log.info("session speaker count: %s (from client)", expected or "auto")
        except ValueError:
            pass

    s = Session(ws, expected)
    s.diar.start()

    queue: asyncio.Queue = asyncio.Queue()
    await s.send(status_msg("ready"))

    def reap(task: asyncio.Task) -> None:
        s.inflight.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            log.error("segment task failed: %s", exc, exc_info=exc)

    async def worker():
        while True:
            item = await queue.get()
            try:
                if item is None:
                    if s.inflight:
                        await asyncio.gather(*list(s.inflight), return_exceptions=True)
                    return
                if s.dead:
                    continue
                seg_id, segment = item
                task = asyncio.create_task(s.handle_segment(segment, seg_id))
                s.inflight.add(task)
                task.add_done_callback(reap)
                # Bound the backlog rather than queueing without limit.
                while len(s.inflight) >= ASR_CONCURRENCY * 2:
                    await asyncio.wait(
                        list(s.inflight), return_when=asyncio.FIRST_COMPLETED
                    )
            except Exception as e:  # noqa: BLE001
                log.error("worker error: %s", e)
                traceback.print_exc()
                await s.send(error_msg(f"internal error while processing audio: {e}"))
            finally:
                queue.task_done()

    worker_task = asyncio.create_task(worker())
    seg_id = 0
    last_beat = 0.0

    try:
        while True:
            msg = await ws.receive()

            if msg.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect(msg.get("code", 1000))

            if msg.get("bytes") is not None:
                # Raw audio goes to the diarizer CONTINUOUSLY — it does its own
                # VAD and needs an uninterrupted clock, not VAD segments with
                # pre-roll padding glued on.
                s.diar.feed(msg["bytes"])
                for segment in s.seg.add_audio(msg["bytes"]):
                    seg_id += 1
                    queue.put_nowait((seg_id, segment))
                if s.seg.now - last_beat >= 10.0:
                    last_beat = s.seg.now
                    log.info(
                        "audio: %.0fs received, %d segment(s), diar=%s",
                        s.seg.now,
                        seg_id,
                        s.diar.stats(),
                    )

            elif msg.get("text") == "stop":
                segment = s.seg.flush()
                if segment:
                    seg_id += 1
                    queue.put_nowait((seg_id, segment))
                await queue.join()
                if s.inflight:
                    log.info("draining %d in-flight segment(s)", len(s.inflight))
                    await asyncio.gather(*list(s.inflight), return_exceptions=True)

                # Final pass: every window of the session is available, so this
                # is plain offline diarization — the best labelling the system
                # can produce. Always worth one last refresh.
                try:
                    await s.diar.finalize()
                    async with s.emit_lock:
                        if s.store.relabel(s.diar.label_for):
                            log.info("final pass corrected the transcript")
                        await s.send(refresh_msg(s.store.paragraphs()))
                except Exception as e:  # noqa: BLE001
                    log.warning("final diarization pass failed: %s", e)

                log.info("session done: %s", s.diar.stats())
                await s.send(status_msg("stopped"))

    except WebSocketDisconnect:
        log.info("client disconnected after %d segments", seg_id)
    finally:
        s.dead = True
        queue.put_nowait(None)
        try:
            await asyncio.wait_for(worker_task, timeout=10)
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001
            worker_task.cancel()
        await s.aclose()
