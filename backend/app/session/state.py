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

CAPACITY IS NOT DECIDED HERE ANY MORE
-------------------------------------
The baseline gave every session its own `Semaphore(6)`, so total ASR pressure
was a per-session opinion multiplied by however many browsers connected. Both
of those gaps are now closed:

  * ASR concurrency belongs to the process-wide `ASRScheduler`. This class
    only submits work to it. `MAX_INFLIGHT_PER_SESSION` remains, but it is a
    FAIRNESS limit — it stops one session monopolising the shared queue — not
    a capacity limit.

  * The segment queue is bounded (`SEGMENT_QUEUE_MAXSIZE`) with an explicit
    overflow policy: drop the oldest waiting segment. Under overload the
    transcript now falls behind and recovers, instead of growing a backlog
    that never drains.

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
from ..auth.context import TenantContext
from ..asr.scheduler import ASRQueueFull, ASRScheduler
from ..audio.music_gate import is_music
from ..audio.vad import Segment, VADSegmenter
from ..config import settings
from ..diarization.factory import build_diarizer
from ..tenant.models import PLANS
from .transcript import TranscriptStore

log = logging.getLogger("voxlive.session")

# How many segment tasks one session may have in flight. This is a FAIRNESS
# limit, not a capacity limit: total ASR pressure is now the scheduler's job,
# and this only stops one very talkative session from filling the shared queue
# with its own backlog.
MAX_INFLIGHT_PER_SESSION = int(os.getenv("MAX_INFLIGHT_PER_SESSION", "12"))

# Bounded per-session audio backlog. The baseline used an unbounded
# asyncio.Queue(): if processing fell behind, segments accumulated with no
# ceiling and one slow session could exhaust process memory. Overflow now
# drops the OLDEST waiting segment, because in live transcription stale audio
# is the least valuable thing in the queue.
SEGMENT_QUEUE_MAXSIZE = int(os.getenv("SEGMENT_QUEUE_MAXSIZE", "32"))

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
        tenant: TenantContext,
        expected_speakers: int,
        asr_scheduler: ASRScheduler,
        postproc: Any,
        speaker_mode: str = "",
    ):
        # ---- identity -----------------------------------------------------
        self.session_id = session_id
        # Established once at connect time and never reassigned. Every log
        # line, every future DB write and every S3 key derives from this.
        self.tenant = tenant
        self.organization_id = tenant.organization_id
        self.user_id = tenant.user_id
        self.expected_speakers = expected_speakers
        # "auto" = estimate K (expected_speakers is a ceiling).
        # "fixed" = K IS expected_speakers. Chosen per session by the client.
        self.speaker_mode = (
            speaker_mode or getattr(settings, "speaker_mode", "auto") or "auto"
        ).lower()
        self.created_at = time.time()
        self.last_activity = self.created_at
        self.status = SessionStatus.CONNECTING

        # ---- collaborators (injected, never reached for globally) ---------
        self.ws = ws
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
        self.diar = build_diarizer(settings, expected_speakers, self.speaker_mode)
        self.diar.on_change = self.on_diarization_change

        # ---- recording segmentation (supervisor review, §10/§11) ----------
        # A session may contain SEVERAL independent recordings. The review is
        # explicit that "pause in conversation" and "new recording" are
        # different events and that speakers must NOT be reset on silence — a
        # six-second thinking pause in an interview is not a new recording, and
        # resetting there would invent fresh identities mid-conversation.
        #
        # So the reset is an explicit client control, and this counter marks
        # which recording each chunk belongs to. Without it, two different
        # people in two different recordings would both render as "Speaker 1"
        # in one transcript with nothing to distinguish them.
        self.recording = 1
        self.recording_started_at = 0.0

        self.recent: Optional[deque] = (
            deque(maxlen=settings.context_segments)
            if settings.context_segments > 0
            else None
        )

        # ---- concurrency --------------------------------------------------
        self.asr = asr_scheduler
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=SEGMENT_QUEUE_MAXSIZE)
        self.dropped_segments = 0
        self.emit_lock = asyncio.Lock()  # serialises store mutation + sends
        self.inflight: set[asyncio.Task] = set()
        self._worker_task: Optional[asyncio.Task] = None

        # ---- counters -----------------------------------------------------
        self.seg_id = 0
        self.sequence = 0  # outbound message counter
        self.dead = False

        # ---- hard duration cap ---------------------------------------------
        # `Plan.max_session_minutes` existed in the model and was enforced
        # nowhere, so a browser tab left open on a stream billed Vertex AI for
        # as long as it stayed open. Admission control only ever counted
        # CONCURRENT sessions, which does not bound the duration of any of
        # them. This is the cheapest possible ceiling: it costs one float
        # comparison per audio chunk and it is the difference between a
        # forgotten tab costing minutes and costing a weekend.
        plan = PLANS.get(tenant.plan_code)
        self.max_audio_sec = (plan.max_session_minutes * 60) if plan else 0
        self.limit_reached = False
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
            "organization_id": self.organization_id,
            "user_id": self.user_id,
            "plan": self.tenant.plan_code,
            "status": self.status.value,
            "age_sec": round(now - self.created_at, 1),
            "idle_sec": round(now - self.last_activity, 1),
            "audio_sec": round(self.seg.now, 1),
            "segments": self.seg_id,
            "inflight": len(self.inflight),
            "queue_depth": self.queue.qsize(),
            "dropped_segments": self.dropped_segments,
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
            # Scoped to the CURRENT recording: the diarizer's timeline only
            # covers this one, and earlier recordings keep the labels they
            # finished with.
            if self.store.relabel(self.diar.label_for, recording=self.recording):
                await self.publish()

    # -------------------------------------------------------------- inbound

    async def feed_audio(self, pcm: bytes) -> None:
        """Accept one chunk of raw int16 PCM from the client."""
        self.last_activity = time.time()

        # Stop ACCEPTING audio at the plan ceiling, then drain normally, so
        # the user keeps every word already transcribed. Dropping the socket
        # outright would discard whatever is still in flight and leave them
        # with a truncated transcript and no explanation.
        if (
            self.max_audio_sec
            and not self.limit_reached
            and self.seg.now >= self.max_audio_sec
        ):
            self.limit_reached = True
            log.info(
                "session reached the %.0f-minute plan limit; draining",
                self.max_audio_sec / 60,
                extra={"session_id": self.session_id, "event": "session_limit"},
            )
            await self.send(
                error_msg(
                    f"This session reached its {self.max_audio_sec // 60}-minute "
                    "limit. Download the transcript, then start a new one."
                )
            )
            await self.finish()
            return
        if self.limit_reached:
            return

        # Raw audio goes to the diarizer CONTINUOUSLY — it does its own VAD and
        # needs an uninterrupted clock, not VAD segments with pre-roll padding
        # glued on.
        self.diar.feed(pcm)
        for segment in self.seg.add_audio(pcm):
            self.seg_id += 1
            self._enqueue(self.seg_id, segment)

        if self.seg.now - self._last_beat >= 10.0:
            self._last_beat = self.seg.now
            log.info(
                "audio: %.0fs received, %d segment(s), diar=%s",
                self.seg.now,
                self.seg_id,
                self.diar.stats(),
                extra={"session_id": self.session_id},
            )

    def _enqueue(self, seg_id: int, segment: Segment) -> None:
        """Queue a segment, dropping the OLDEST if the backlog is full.

        Dropping the oldest rather than the newest is deliberate: this is live
        transcription, and a segment that has been waiting so long that the
        queue filled behind it is the one the user has already stopped caring
        about. Refusing the newest would mean the transcript freezes at the
        moment of overload, which is the worst possible symptom.
        """
        try:
            self.queue.put_nowait((seg_id, segment))
        except asyncio.QueueFull:
            try:
                stale_id, _ = self.queue.get_nowait()
                self.queue.task_done()
                self.dropped_segments += 1
                log.warning(
                    "segment backlog full (%d) — dropped segment %d",
                    SEGMENT_QUEUE_MAXSIZE,
                    stale_id,
                    extra={
                        "session_id": self.session_id,
                        "segment_id": stale_id,
                        "event": "segment_dropped",
                    },
                )
            except asyncio.QueueEmpty:
                pass
            self.queue.put_nowait((seg_id, segment))

    async def finish(self) -> None:
        """Client sent `stop`: flush, drain, run the final offline pass."""
        self.status = SessionStatus.DRAINING
        self.last_activity = time.time()

        segment = self.seg.flush()
        if segment:
            self.seg_id += 1
            self._enqueue(self.seg_id, segment)

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
                if self.store.relabel(self.diar.label_for, recording=self.recording):
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

        blended, blended_sec = self.store.unsplit_chunks(self.diar.timeline())
        if blended:
            log.warning(
                "%d paragraph(s) (%.1fs) contain a speaker change that could "
                "not be split: the timeline did not cover them when they were "
                "cut. Shorten segments (SILENCE_MS / SOFT_MAX_SEGMENT_MS / "
                "MAX_SEGMENT_MS) if this stays high.",
                blended,
                blended_sec,
                extra={
                    "session_id": self.session_id,
                    "event": "diarization_unsplit",
                    "unsplit_chunks": blended,
                },
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

    async def new_recording(self) -> None:
        """Client pressed NEW RECORDING (supervisor review, §10).

        Starts an independent speaker universe on the SAME WebSocket session,
        without touching the transcript already on screen and without stopping
        the session clock.

        WHY THIS IS AN EXPLICIT CONTROL AND NOT A SILENCE TIMER
        ------------------------------------------------------
        The review is direct about this: do NOT reset on silence. Consider

            Interviewer: "Can you explain that?"
            [6 second thinking pause]
            Interviewee: "Yes..."

        A silence-triggered reset there would fabricate a whole new set of
        identities in the middle of one conversation. The application has to
        distinguish "pause" from "new recording", and only the user knows
        which one just happened — so only the user gets to say.

        WHAT IS AND IS NOT RESET
          reset:      speaker identities, centroids, windows, timeline
          preserved:  the transcript, the audio clock, the session, the socket

        The clock matters. `DiarizationService.reset()` used to zero its
        offsets, which would place the next windows at t=0 while
        `VADSegmenter.now` keeps stamping chunks at t=430 — timeline and text
        describing different moments. It now takes the current audio time so
        the two stay on one clock.
        """
        async with self.emit_lock:
            # Drain what the old identities still owe the transcript first, so
            # the previous recording keeps its best labelling rather than being
            # frozen mid-correction.
            try:
                await self.diar.finalize()
                self.store.relabel(self.diar.label_for, recording=self.recording)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "final pass before new recording failed: %s",
                    exc,
                    extra={"session_id": self.session_id},
                )

            at = self.seg.now
            self.diar.reset(at)
            self.recording += 1
            self.recording_started_at = at
            # Rolling ASR context must not leak across the boundary either: the
            # previous recording's last sentence is not context for this one.
            if self.recent is not None:
                self.recent.clear()

            log.info(
                "new recording %d started at %.1fs — speaker identities reset",
                self.recording,
                at,
                extra={
                    "session_id": self.session_id,
                    "event": "new_recording",
                    "recording": self.recording,
                },
            )
            self.last_speaker_count = 0
            await self.send(refresh_msg(self.store.paragraphs()))
        await self.send(status_msg("ready"))

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
                while len(self.inflight) >= MAX_INFLIGHT_PER_SESSION:
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

        # ---- speech / music gate (supervisor review, Fix #4) -------------
        # The RMS check above is an ENERGY gate and music is loud, so it never
        # rejected a song. WebRTC VAD is a SPEECH detector and sung vocals pass
        # it. Until now the only thing between a Sinhala song and a transcript
        # of it was a line in the Gemini prompt asking a generative model to
        # also be an audio classifier.
        #
        # Runs BEFORE the ASR call but AFTER the diarizer has already been fed
        # (that happens continuously in feed_audio), so a rejected segment
        # costs a Gemini request and nothing else — the speaker timeline keeps
        # its uninterrupted clock either way.
        if settings.music_gate_mode in ("log", "drop"):
            discard, verdict = is_music(
                segment.pcm, settings.sample_rate, settings.music_gate_threshold
            )
            if settings.music_gate_mode == "log":
                # Observation only. Read these lines on real Sinhala speech and
                # real songs, then set MUSIC_GATE_THRESHOLD from the actual
                # distribution before switching the mode to `drop`.
                log.info(
                    "music-gate: segment %d %s (mode=log, nothing dropped)",
                    seg_id,
                    verdict.as_log(),
                    extra={"session_id": self.session_id, "segment_id": seg_id},
                )
            elif discard:
                log.info(
                    "segment %d discarded as music: %s",
                    seg_id,
                    verdict.as_log(),
                    extra={
                        "session_id": self.session_id,
                        "segment_id": seg_id,
                        "event": "music_rejected",
                    },
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
            chunks = [
                (self.store.new_chunk(seg_id, a, b, self.recording), pcm)
                for pcm, a, b in pieces
            ]

        await self.send(status_msg("transcribing"))
        context = " ".join(self.recent) if self.recent else None

        async def one(chunk, pcm):
            result, limited, err = await self._asr(pcm, context, seg_id)
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

    async def _asr(self, pcm: bytes, context, seg_id: int):
        """Hand one chunk to the global scheduler.

        Retries, jittered backoff, the per-call timeout and the process-wide
        429 cooldown all live in ASRScheduler now. This method only has to
        translate the outcome into the (result, limited, error) triple the
        emit path expects.
        """
        try:
            result = await self.asr.submit(
                pcm,
                settings.sample_rate,
                context=context,
                session_id=self.session_id,
                segment_id=seg_id,
            )
            return result, False, None
        except ASRQueueFull as exc:
            log.warning(
                "ASR capacity reached — segment %d dropped",
                seg_id,
                extra={
                    "session_id": self.session_id,
                    "segment_id": seg_id,
                    "event": "asr_rejected",
                },
            )
            return None, True, exc
        except Exception as exc:  # noqa: BLE001
            return None, is_rate_limit(exc), exc

    async def _emit(self, chunk, result, limited, err, seg_id: int) -> None:
        async with self.emit_lock:
            if result is None:
                self.store.drop(chunk)
                await self.send(
                    error_msg(
                        (
                            "Transcription is at capacity right now. Please retry."
                            if limited
                            else "transcription failed"
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
            # Phase 4. Spans index `result.text`; they stay valid only while
            # the text does. `PostProcessor` may rewrite it (ENABLE_POSTPROCESS
            # is off by default), so recompute rather than carrying offsets
            # that silently point at the pre-edit string.
            spans = list(getattr(result, "language_spans", []) or [])
            if settings.enable_postprocess and text != result.text:
                from ..asr.language_spans import language_spans as _spans

                spans = _spans(text, result.language)
            chunk.language_spans = spans
            chunk.speaker = self.diar.label_for(chunk.start, chunk.end)
            if chunk.speaker is None:
                prev = [
                    c
                    for c in self.store.chunks
                    if c.speaker is not None and c.end <= chunk.start
                ]
                chunk.speaker = prev[-1].speaker if prev else 0

            await self.publish()
