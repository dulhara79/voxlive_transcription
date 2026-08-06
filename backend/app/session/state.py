"""
state.py — SessionState: everything ONE live transcription session owns.

WHAT THIS IS
------------
This is the `Session` class that used to live in `main.py`, lifted out
unchanged in behaviour and given the identity/lifecycle fields a production
system needs. It answers exactly one question:

    "What is happening in this one live session?"

It deliberately knows NOTHING about AWS, RDS, Redis, ECS or Cognito. It does
not know how many other sessions exist (that is SessionManager's job) and it
does not know how the browser is addressed (that is routes_ws.py's job).

WHAT DELIBERATELY DID *NOT* CHANGE
----------------------------------
This commit is a responsibility split, not a behaviour change. The acceptance
test is that the refactored branch behaves identically to the frozen
`v1-demo-baseline` demo. So two known production problems are preserved here
EXACTLY as they were, and are fixed in the NEXT commit:

  1. `self.asr_slots` is still a PER-SESSION semaphore. 500 sessions x 6 =
     up to 3,000 concurrent ASR calls. This becomes a single process-wide
     ASRScheduler in `app/asr/scheduler.py`.

  2. `self.queue` is still an unbounded `asyncio.Queue()`. The number of
     in-flight TASKS is bounded (`ASR_CONCURRENCY * 2`), but the waiting
     backlog is not. This gains a `maxsize` and an explicit overflow policy
     in the same commit as the scheduler.

Both are marked `PRODUCTION-GAP` below. Do not fix them here — fixing them in
this commit would make it impossible to tell whether a behaviour change came
from the restructure or from the concurrency work.

THE PIPELINE THIS OWNS
----------------------
    raw audio ──┬──► VADSegmenter ──► segment ──► queue ──► ASR ──► text
                │                                                    │
                └──► DiarizationService ──► speaker timeline ────────┘
                       (background)               joined on the audio clock
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections import deque
from enum import Enum
from typing import Any, Optional

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from ..api.schemas import error_msg, refresh_msg, speakers_msg, status_msg
from ..audio.vad import Segment, VADSegmenter
from ..config import settings
from ..diarization.factory import build_diarizer
from .transcript import TranscriptStore

log = logging.getLogger("voxlive.session")

ASR_MAX_RETRIES = 3

# PRODUCTION-GAP (1 of 2): per-session ASR ceiling. Moves into a single
# process-wide ASRScheduler in the next commit; kept here, with the same env
# var and the same default, so this commit changes no behaviour.
ASR_CONCURRENCY = int(os.getenv("ASR_CONCURRENCY", "6"))

# Padding around a speaker-turn cut so word onsets/offsets aren't clipped.
CHUNK_PAD_PRE_S = 0.10
CHUNK_PAD_POST_S = 0.15


def new_session_id() -> str:
    """A sortable, log-greppable session id: sess_<ms>_<8 hex>.

    The millisecond prefix means a plain `sort` on a log file is chronological,
    which matters a lot when reading 500 interleaved sessions.
    """
    return f"sess_{int(time.time() * 1000):013d}_{uuid.uuid4().hex[:8]}"


def rms(pcm: bytes) -> float:
    if not pcm:
        return 0.0
    arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(arr * arr)))


def is_rate_limit(err: Any) -> bool:
    s = str(err)
    return "429" in s or "RESOURCE_EXHAUSTED" in s


class SessionStatus(str, Enum):
    """Lifecycle of one session. Str-valued so it logs and serialises cleanly."""

    CONNECTING = "connecting"
    ACTIVE = "active"
    DRAINING = "draining"  # client sent `stop`; finishing in-flight work
    CLOSED = "closed"


class SessionState:
    """One live WebSocket transcription session."""

    def __init__(
        self,
        ws: WebSocket,
        session_id: str,
        expected_speakers: int,
        provider: Any,
        postproc: Any,
    ):
        # ---- identity -----------------------------------------------------
        self.session_id = session_id
        self.expected_speakers = expected_speakers
        self.created_at = time.time()
        self.last_activity = self.created_at
        self.status = SessionStatus.CONNECTING

        # ---- collaborators (injected, never reached for globally) ---------
        self.ws = ws
        self.provider = provider
        self.postproc = postproc

        # ---- pipeline -----------------------------------------------------
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

        self.recent: Optional[deque] = (
            deque(maxlen=settings.context_segments)
            if settings.context_segments > 0
            else None
        )

        # ---- concurrency --------------------------------------------------
        # PRODUCTION-GAP (1 of 2): see module docstring.
        self.asr_slots = asyncio.Semaphore(ASR_CONCURRENCY)
        # PRODUCTION-GAP (2 of 2): no maxsize. Bounded in the scheduler commit.
        self.queue: asyncio.Queue = asyncio.Queue()
        self.emit_lock = asyncio.Lock()  # serialises store mutation + sends
        self.inflight: set[asyncio.Task] = set()
        self._worker_task: Optional[asyncio.Task] = None

        # ---- counters -----------------------------------------------------
        self.seg_id = 0
        self.sequence = 0  # outbound message counter
        self.dead = False
        self.last_speaker_count = 0
        self._last_beat = 0.0

    # ----------------------------------------------------------- life cycle

    def start(self) -> None:
        """Begin background diarization and the segment worker."""
        self.diar.start()
        self._worker_task = asyncio.create_task(self._worker())
        self.status = SessionStatus.ACTIVE

    async def aclose(self) -> None:
        """Stop everything this session owns. Safe to call more than once."""
        self.dead = True
        self.status = SessionStatus.CLOSED
        self.queue.put_nowait(None)
        if self._worker_task is not None:
            try:
                await asyncio.wait_for(self._worker_task, timeout=10)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._worker_task.cancel()
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "worker shutdown error: %s",
                    exc,
                    exc_info=exc,
                    extra={"session_id": self.session_id},
                )
                self._worker_task.cancel()
            self._worker_task = None
        await self.diar.aclose()

    def snapshot(self) -> dict:
        """Cheap, allocation-light view for logging, /health and (later)
        CloudWatch metrics. Never touches the model or the transcript."""
        now = time.time()
        return {
            "session_id": self.session_id,
            "status": self.status.value,
            "age_sec": round(now - self.created_at, 1),
            "idle_sec": round(now - self.last_activity, 1),
            "audio_sec": round(self.seg.now, 1),
            "segments": self.seg_id,
            "inflight": len(self.inflight),
            "queue_depth": self.queue.qsize(),
            "speakers": self.last_speaker_count,
            "sequence": self.sequence,
        }

    # ------------------------------------------------------------- outbound

    async def send(self, payload: dict) -> None:
        if self.dead:
            return
        # Counted but NOT yet stamped onto the payload: adding a `sequence`
        # field is a wire-protocol change and this commit must not change the
        # wire. The counter is here so the protocol commit is a one-line diff.
        self.sequence += 1
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

    # -------------------------------------------------------------- inbound

    async def feed_audio(self, pcm: bytes) -> None:
        """Accept one chunk of raw int16 PCM from the client."""
        self.last_activity = time.time()

        # Raw audio goes to the diarizer CONTINUOUSLY — it does its own VAD and
        # needs an uninterrupted clock, not VAD segments with pre-roll padding
        # glued on.
        self.diar.feed(pcm)
        for segment in self.seg.add_audio(pcm):
            self.seg_id += 1
            self.queue.put_nowait((self.seg_id, segment))

        if self.seg.now - self._last_beat >= 10.0:
            self._last_beat = self.seg.now
            log.info(
                "audio: %.0fs received, %d segment(s), diar=%s",
                self.seg.now,
                self.seg_id,
                self.diar.stats(),
                extra={"session_id": self.session_id},
            )

    async def finish(self) -> None:
        """Client sent `stop`: flush, drain, run the final offline pass."""
        self.status = SessionStatus.DRAINING
        self.last_activity = time.time()

        segment = self.seg.flush()
        if segment:
            self.seg_id += 1
            self.queue.put_nowait((self.seg_id, segment))

        await self.queue.join()
        if self.inflight:
            log.info(
                "draining %d in-flight segment(s)",
                len(self.inflight),
                extra={"session_id": self.session_id},
            )
            await asyncio.gather(*list(self.inflight), return_exceptions=True)

        # Final pass: every window of the session is available, so this is plain
        # offline diarization — the best labelling the system can produce.
        try:
            await self.diar.finalize()
            async with self.emit_lock:
                if self.store.relabel(self.diar.label_for):
                    log.info(
                        "final pass corrected the transcript",
                        extra={"session_id": self.session_id},
                    )
                await self.send(refresh_msg(self.store.paragraphs()))
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "final diarization pass failed: %s",
                exc,
                extra={"session_id": self.session_id},
            )

        log.info(
            "session done: %s",
            self.diar.stats(),
            extra={"session_id": self.session_id},
        )
        await self.send(status_msg("stopped"))
        # Baseline parity: after `stop` the socket stays open and the client may
        # start a new recording on the same connection.
        self.status = SessionStatus.ACTIVE

    # --------------------------------------------------------------- worker

    def _reap(self, task: asyncio.Task) -> None:
        self.inflight.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            log.error(
                "segment task failed: %s",
                exc,
                exc_info=exc,
                extra={"session_id": self.session_id},
            )

    async def _worker(self) -> None:
        while True:
            item = await self.queue.get()
            try:
                if item is None:
                    if self.inflight:
                        await asyncio.gather(
                            *list(self.inflight), return_exceptions=True
                        )
                    return
                if self.dead:
                    continue
                seg_id, segment = item
                task = asyncio.create_task(self.handle_segment(segment, seg_id))
                self.inflight.add(task)
                task.add_done_callback(self._reap)
                # Bound the backlog rather than queueing without limit.
                while len(self.inflight) >= ASR_CONCURRENCY * 2:
                    await asyncio.wait(
                        list(self.inflight), return_when=asyncio.FIRST_COMPLETED
                    )
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "worker error: %s",
                    exc,
                    exc_info=exc,
                    extra={"session_id": self.session_id},
                )
                await self.send(
                    error_msg(f"internal error while processing audio: {exc}")
                )
            finally:
                self.queue.task_done()

    # -------------------------------------------------------------- segment

    async def handle_segment(self, segment: Segment, seg_id: int) -> None:
        energy = rms(segment.pcm)
        if energy < settings.min_segment_rms:
            log.info(
                "segment %d gated: rms=%.0f < %d",
                seg_id,
                energy,
                settings.min_segment_rms,
                extra={"session_id": self.session_id, "segment_id": seg_id},
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
            extra={"session_id": self.session_id, "segment_id": seg_id},
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

        The cuts come from the same clustering that decides identity, so a
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
            if len(piece) < min_bytes or rms(piece) < settings.min_segment_rms:
                continue
            out.append((piece, a2, b2))

        if not out:
            return [(segment.pcm, segment.start, segment.end)]
        log.info(
            "segment split into %d turn(s) at %s",
            len(out),
            ", ".join(f"{c:.2f}s" for c in cuts),
            extra={"session_id": self.session_id},
        )
        return out

    async def _asr(self, pcm: bytes, context):
        last_err, limited = None, False
        for attempt in range(ASR_MAX_RETRIES):
            try:
                r = await self.provider.transcribe_segment(
                    pcm, settings.sample_rate, context=context
                )
                return r, False, None
            except Exception as e:  # noqa: BLE001
                if isinstance(e, (WebSocketDisconnect, RuntimeError)):
                    raise
                last_err = e
                if is_rate_limit(e):
                    limited = True
                    log.error(
                        "HTTP 429 from Vertex — backing off",
                        extra={"session_id": self.session_id},
                    )
                    await asyncio.sleep(2**attempt)
                    continue
                log.warning(
                    "ASR attempt %d/%d failed: %s",
                    attempt + 1,
                    ASR_MAX_RETRIES,
                    e,
                    extra={"session_id": self.session_id},
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
                    log.info(
                        "dropped segment %d: lang=%s",
                        seg_id,
                        result.language,
                        extra={"session_id": self.session_id, "segment_id": seg_id},
                    )
                self.store.drop(chunk)
                return

            text = result.text
            if settings.enable_postprocess:
                text = await self.postproc.process(text, result.language)

            # A long output identical to the previous one is a stuck model, not
            # a person repeating a sentence word for word.
            if self.recent and len(text) > 25 and self.recent[-1] == text:
                log.info(
                    "repeat guard: segment %d dropped",
                    seg_id,
                    extra={"session_id": self.session_id, "segment_id": seg_id},
                )
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
