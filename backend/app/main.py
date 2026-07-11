"""
VoxLive backend — Gemini build + pyannote diarization + paragraph grouping (v3).

Pipeline per WebSocket connection (DECOUPLED so audio is never dropped while
waiting on the Gemini API):

    mic PCM (binary frames)
      -> [ingest loop]  VADSegmenter cuts the stream at pauses and pushes
                        finalized segments onto an async queue. NEVER blocks
                        on transcription, so the socket is always drained.
      -> [worker loop]  pulls segments in order, energy-gates them (silence
                        never reaches Gemini), then runs Gemini transcription
                        and pyannote diarization IN PARALLEL (latency), merges
                        consecutive same-speaker segments into one paragraph,
                        and sends transcript JSON to the client.

Run:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

DESIGN NOTES
  - Diarizer is PER-CONNECTION (speaker memory is session-local); the heavy
    pyannote model + voiceprints are shared via caches in diarization.py.
  - LATENCY: diarization (CPU, 100-400ms) runs concurrently with the Gemini
    call, so it is effectively free.
  - Timestamps come from the AUDIO CLOCK (bytes / 2 / sample_rate), so they
    reflect when words were spoken, not when Gemini finished.
  - The worker DRAINS the queue if the socket dies (never deadlocks join()).
  - ROLLING CONTEXT: last N transcripts are passed to Gemini per call —
    isolated 2-4s chunks are exactly where an LLM transcriber hallucinates.
  - PARAGRAPH GROUPING: consecutive segments by the SAME speaker share one
    paragraph_id; every transcript message carries the FULL accumulated
    paragraph text, so the frontend must UPSERT (replace) by paragraph_id —
    NOT append every message. See schemas.py and TranscriptView.jsx.
"""

import asyncio
import logging
import traceback
from collections import deque
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .audio import VADSegmenter
from .schemas import status_msg, transcript_msg
from .diarization import Diarizer
from .postprocess import PostProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("voxlive")

ASR_MAX_RETRIES = 3  # retry a failed Gemini call before surfacing an error


def build_provider():
    """Single engine in this build: Gemini. The SpeechProvider abstraction is
    kept so another engine could be slotted in without touching the pipeline."""
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


def build_diarizer():
    """Called PER CONNECTION (cheap: model + voiceprints are cached at module
    level in diarization.py). Each connection gets its own speaker memory so
    speakers never leak between sessions."""
    mode = settings.diarization_mode.lower()
    if mode == "identify":
        from .diarization import IdentifyingDiarizer

        return IdentifyingDiarizer(
            voiceprints_dir=settings.voiceprints_dir,
            id_threshold=settings.diarization_threshold,
            cluster_threshold=settings.diarization_threshold,
            max_speakers=settings.max_speakers,
            hf_token=settings.huggingface_token,
            min_new_speaker_sec=settings.min_new_speaker_sec,
            new_speaker_margin=settings.diarization_new_speaker_margin,
        )
    if mode in ("pyannote", "cluster"):  # "cluster" kept as an alias
        from .diarization import PyannoteDiarizer

        return PyannoteDiarizer(
            threshold=settings.diarization_threshold,
            max_speakers=settings.max_speakers,
            hf_token=settings.huggingface_token,
            min_new_speaker_sec=settings.min_new_speaker_sec,
            new_speaker_margin=settings.diarization_new_speaker_margin,
        )
    return Diarizer()


def _rms(pcm: bytes) -> float:
    """Root-mean-square energy of an int16 PCM segment (0..32767 scale)."""
    if not pcm:
        return 0.0
    arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(arr * arr)))


class ParagraphState:
    """Per-connection paragraph merger: consecutive segments by the same
    speaker accumulate into one paragraph (one paragraph_id)."""

    def __init__(self):
        self.pid = 0
        self.speaker: str | None = None
        self.texts: list[str] = []
        self.start: float = 0.0

    def add(self, speaker: str, text: str, start: float) -> tuple[int, str, float]:
        """Returns (paragraph_id, full_paragraph_text, paragraph_start)."""
        if speaker != self.speaker:
            self.pid += 1
            self.speaker = speaker
            self.texts = [text]
            self.start = start
        else:
            self.texts.append(text)
        return self.pid, " ".join(self.texts), self.start


state: dict = {}  # provider/postproc are stateless per segment -> shared per process


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["provider"] = build_provider()
    state["postproc"] = PostProcessor()
    if settings.diarization_mode.lower() in ("pyannote", "cluster", "identify"):
        # Warm the shared pyannote model (+ voiceprints) at startup so the
        # FIRST connection doesn't pay a multi-second model download/load.
        build_diarizer()
        log.info("Diarization mode: %s (model warmed)", settings.diarization_mode)
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
        "diarization": settings.diarization_mode,
        "max_speakers": settings.max_speakers,
        "diarization_threshold": settings.diarization_threshold,
        "diarization_new_speaker_margin": settings.diarization_new_speaker_margin,
        "context_segments": settings.context_segments,
    }


@app.websocket("/ws/transcribe")
async def transcribe(ws: WebSocket):
    await ws.accept()
    seg = VADSegmenter(
        sample_rate=settings.sample_rate,
        vad_aggressiveness=settings.vad_aggressiveness,
        silence_ms=settings.silence_ms,
        soft_max_segment_ms=settings.soft_max_segment_ms,
        max_segment_ms=settings.max_segment_ms,
        min_segment_ms=settings.min_segment_ms,
    )
    diarizer = build_diarizer()  # per-connection speaker state
    para = ParagraphState()  # per-connection paragraph merger

    # Rolling context: last N transcripts from THIS conversation only.
    recent: "deque[str] | None" = (
        deque(maxlen=settings.context_segments)
        if settings.context_segments > 0
        else None
    )

    # Unbounded queue: we would rather grow lag than ever drop a spoken word.
    queue: "asyncio.Queue" = asyncio.Queue()
    bytes_seen = 0  # audio clock: exact stream time, immune to queue lag
    await ws.send_json(status_msg("ready"))

    async def worker():
        """Transcribe queued segments in order (one at a time = ordered output).

        If the socket dies mid-stream we set `dead` and keep DRAINING the queue
        (calling task_done on every item) instead of returning — otherwise a
        pending queue.join() in the stop path would hang forever."""
        dead = False
        while True:
            item = await queue.get()
            try:
                if item is None:  # shutdown sentinel
                    return
                if dead:
                    continue
                seg_id, pcm, end_time = item
                try:
                    await _handle(ws, diarizer, para, recent, pcm, seg_id, end_time)
                except (WebSocketDisconnect, RuntimeError):
                    # RuntimeError = starlette "send after close"
                    dead = True
                    log.info("socket closed mid-transcription; draining queue")
                    traceback.print_exc()
            except Exception as e:  # noqa: BLE001
                log.error("worker error: %s", e)
            finally:
                queue.task_done()

    worker_task = asyncio.create_task(worker())
    seg_id = 0

    try:
        while True:
            msg = await ws.receive()

            if msg.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect(msg.get("code", 1000))

            if msg.get("bytes") is not None:
                # Pure ingestion: segment + enqueue. Never awaits Gemini, so the
                # socket is always drained and no audio is lost under API latency.
                bytes_seen += len(msg["bytes"])
                audio_now = bytes_seen / 2 / settings.sample_rate
                for pcm in seg.add_audio(msg["bytes"]):
                    seg_id += 1
                    queue.put_nowait((seg_id, pcm, audio_now))

            elif msg.get("text") is not None:
                if msg["text"] == "stop":
                    pcm = seg.flush()
                    if pcm:
                        seg_id += 1
                        queue.put_nowait(
                            (seg_id, pcm, bytes_seen / 2 / settings.sample_rate)
                        )
                    # Wait for every queued segment to be sent before "stopped"
                    # so the tail of the meeting is never lost. The frontend
                    # keeps the socket open until it sees "stopped".
                    await queue.join()
                    try:
                        await ws.send_json(status_msg("stopped"))
                    except (WebSocketDisconnect, RuntimeError):
                        log.info("client left before 'stopped' could be sent")

    except WebSocketDisconnect:
        log.info("client disconnected after %d segments", seg_id)
    finally:
        queue.put_nowait(None)
        try:
            await asyncio.wait_for(worker_task, timeout=10)
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001
            worker_task.cancel()


def _is_rate_limit(err) -> bool:
    s = str(err)
    return "429" in s or "RESOURCE_EXHAUSTED" in s


async def _handle(
    ws: WebSocket,
    diarizer: Diarizer,
    para: ParagraphState,
    recent,
    pcm: bytes,
    seg_id: int,
    end_time: float,
):
    # --- ANTI-HALLUCINATION GATE 0: energy. Near-silent segments (breath,
    # hum, keyboard) are the #1 trigger for Gemini inventing fluent speech.
    # They never reach the API at all. ---
    energy = _rms(pcm)
    if energy < settings.min_segment_rms:
        log.info(
            "segment %d gated: rms=%.0f < %d (silence/noise)",
            seg_id,
            energy,
            settings.min_segment_rms,
        )
        return

    await ws.send_json(status_msg("transcribing"))

    context = " ".join(recent) if recent else None

    # LATENCY: pyannote embedding takes 100-400ms on CPU; run it CONCURRENTLY
    # with the Gemini call instead of after it, so it costs ~0 extra time.
    diarize_task = asyncio.create_task(diarizer.assign(pcm, settings.sample_rate))

    # Retry transient ASR failures with backoff (Vertex 429s are often
    # transient capacity, not just quota).
    result, last_err, rate_limited = None, None, False
    try:
        for attempt in range(ASR_MAX_RETRIES):
            try:
                result = await state["provider"].transcribe_segment(
                    pcm, settings.sample_rate, context=context
                )
                break
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                if isinstance(e, (WebSocketDisconnect, RuntimeError)):
                    raise
                last_err = e
                if _is_rate_limit(e):
                    rate_limited = True
                    log.error(
                        "Gemini/Vertex returned HTTP 429 on segment %d — "
                        "retrying with backoff",
                        seg_id,
                    )
                    await asyncio.sleep(2**attempt)
                    continue
                log.warning(
                    "ASR attempt %d/%d failed on seg %d: %s",
                    attempt + 1,
                    ASR_MAX_RETRIES,
                    seg_id,
                    e,
                )
                await asyncio.sleep(0.5 * (attempt + 1))
    except BaseException:
        diarize_task.cancel()  # never leak the parallel task
        raise

    if result is None:
        diarize_task.cancel()
        msg = (
            "Vertex AI temporarily rejected the request (HTTP 429). Please retry."
            if rate_limited
            else f"transcription failed after {ASR_MAX_RETRIES} retries"
        )
        log.error(
            "ASR gave up on segment %d (%s)",
            seg_id,
            "quota" if rate_limited else last_err,
        )
        await ws.send_json(
            {
                "type": "error",
                "segment_id": seg_id,
                "message": msg,
            }
        )
        await ws.send_json(status_msg("ready"))
        return

    if not result.text:
        diarize_task.cancel()
        await ws.send_json(status_msg("ready"))
        return

    # Only Sinhala / English / Tamil are shown — drop anything else.
    if result.language not in settings.allowed_languages:
        diarize_task.cancel()
        log.info(
            "dropped non-target language segment %d: lang=%s", seg_id, result.language
        )
        await ws.send_json(status_msg("ready"))
        return

    text = result.text
    if settings.enable_postprocess:
        text = await state["postproc"].process(text, result.language)

    # --- ANTI-HALLUCINATION GATE 5: exact-repeat. A long output IDENTICAL to
    # the previous segment is a stuck/looping model, not a person repeating a
    # full sentence word-for-word. Short repeats ("හරි", "ok ok") pass. ---
    if recent and len(text) > 25 and recent[-1] == text:
        diarize_task.cancel()
        log.info("repeat guard: segment %d identical to previous, dropping", seg_id)
        await ws.send_json(status_msg("ready"))
        return

    speaker = await diarize_task  # already finished (or nearly) by now

    # Feed this transcript into the rolling context for the NEXT segment.
    if recent is not None:
        recent.append(text)

    # Audio-clock timestamps: end_time was captured at segmentation time from
    # the byte count, so start/end reflect when the words were actually SPOKEN.
    duration = len(pcm) / 2 / settings.sample_rate
    seg_start = round(max(0.0, end_time - duration), 2)

    # PARAGRAPH MERGE: same speaker as the previous segment -> same
    # paragraph_id, text = full accumulated paragraph. Speaker change ->
    # new paragraph. The frontend MUST replace by paragraph_id, not append
    # (see schemas.py / TranscriptView.jsx) — appending every message is
    # exactly the "repeated growing lines" bug.
    pid, paragraph_text, para_start = para.add(speaker, text, seg_start)

    await ws.send_json(
        transcript_msg(
            segment_id=seg_id,
            paragraph_id=pid,
            speaker=speaker,
            language=result.language,
            text=paragraph_text,
            start=para_start,
            end=round(end_time, 2),
            final=True,
        )
    )
    await ws.send_json(status_msg("ready"))
