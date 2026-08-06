"""
diarization_service.py — diarization as a BACKGROUND service (v10).

THE LATENCY BUG THIS FIXES
--------------------------
In v7, `_diarize()` was awaited before a segment was even queued for ASR:

    chunks, seg_start = await _diarize(...)     # 1-4 s, blocking
    task = asyncio.create_task(_transcribe...)  # only now does ASR start

So every segment paid for diarization *before* transcription began, serially,
for the whole session. Combined with a 6-second soft segment cap and 1-3s of
Gemini latency, the transcript ran 8-13 seconds behind the speaker. That is
not "slow real-time"; it is a batch system with a WebSocket in front of it.

Diarization does not need to be on that path. Speech recognition and speaker
identity are independent questions about the same audio, and the only thing
that has to join them is a timestamp. So:

    audio ──┬──► VAD ──► segment ──► ASR ──► text with (start, end)
            │                                        │
            └──► this service ──► speaker timeline ───┘
                  (background, every DIARIZE_INTERVAL_S)

The two run CONCURRENTLY. Because a diarization pass costs ~150-400 ms of
batched GPU/CPU work while a Gemini call costs 1-3 s, the speaker timeline is
usually ready *before* the text is — the joining is free. When it isn't ready,
the caller waits a bounded few hundred milliseconds and otherwise proceeds
with a best-effort label that a later pass corrects.

WHAT A PASS DOES
  1. Take the audio buffered since the last pass, plus a WIN_SEC overlap tail
     so no window is lost at a buffer seam.
  2. Mask it with VAD. Silence is never embedded.
  3. Cut speech into 1.5s windows at 0.75s hop and embed them in ONE batch.
  4. Hand them to SpeakerEngine, which re-clusters the entire session.
  5. If the timeline changed, fire the callback so the server can push a
     corrected transcript.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable, Optional

import numpy as np

from .embedder import Embedder
from .speaker_engine import (
    WIN_SEC,
    SpeakerEngine,
    Window,
    slice_windows,
    speech_regions,
)

log = logging.getLogger("voxlive.diar")


class DiarizationService:
    """One instance per WebSocket session."""

    def __init__(
        self,
        hf_token: str,
        sample_rate: int = 16000,
        expected_speakers: int = 0,
        max_speakers: int = 6,
        interval_sec: float = 1.5,
        vad_aggressiveness: int = 2,
        device: Optional[str] = None,
        enabled: bool = True,
        **engine_kw,
    ):
        self.sample_rate = sample_rate
        self.interval = float(interval_sec)
        self.vad_aggressiveness = int(vad_aggressiveness)
        self.enabled = bool(enabled)

        self.engine = SpeakerEngine(
            expected_speakers=expected_speakers,
            max_speakers=max_speakers,
            **engine_kw,
        )
        self.embedder = Embedder(hf_token, device=device, sample_rate=sample_rate)

        # Audio not yet turned into windows, with its absolute session offset.
        self._pending = bytearray()
        self._pending_offset = 0.0  # session seconds at _pending[0]
        self._fed_until = 0.0  # session seconds of audio handed in
        self._covered_until = 0.0  # session seconds the timeline reaches

        self._task: Optional[asyncio.Task] = None
        self._wake = asyncio.Event()
        self._stopping = False
        self._lock = asyncio.Lock()
        self._coverage = asyncio.Event()
        self.on_change: Optional[Callable[[], Awaitable[None]]] = None

        self._passes = 0
        self._embed_ms = 0.0

    # ------------------------------------------------------------------ input

    def feed(self, pcm_bytes: bytes) -> None:
        """Hand in raw int16 PCM. Never blocks; never does model work."""
        if not self.enabled or not pcm_bytes:
            return
        self._pending.extend(pcm_bytes)
        self._fed_until += len(pcm_bytes) / 2 / self.sample_rate
        if len(self._pending) / 2 / self.sample_rate >= self.interval:
            self._wake.set()

    # ------------------------------------------------------------- life cycle

    def start(self) -> None:
        if self.enabled and self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def aclose(self) -> None:
        self._stopping = True
        self._wake.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=15)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
            self._task = None

    async def _loop(self) -> None:
        while not self._stopping:
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self.interval)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()
            try:
                await self._pass()
            except Exception as exc:  # noqa: BLE001
                # Diarization failing must never take the transcript with it.
                # The user still gets their words; they get them unlabelled.
                log.error("diarization pass failed: %s", exc, exc_info=True)

    # -------------------------------------------------------------- one pass

    async def _pass(self, final: bool = False) -> None:
        async with self._lock:
            if not self._pending:
                return

            pcm = np.frombuffer(bytes(self._pending), dtype=np.int16)
            pcm = pcm.astype(np.float32) / 32768.0
            offset = self._pending_offset
            dur = len(pcm) / self.sample_rate
            if dur < 0.9 and not final:
                return

            regions = await asyncio.to_thread(
                speech_regions,
                pcm,
                self.sample_rate,
                offset,
                self.vad_aggressiveness,
            )

            # Don't consume speech that is still in progress at the buffer
            # edge: cutting a live sentence produces a stub window and, worse,
            # a false turn boundary. Hold it back for the next pass.
            keep_from = offset + dur
            if regions and not final:
                last_a, last_b = regions[-1]
                if last_b >= offset + dur - 0.05:
                    regions = regions[:-1]
                    keep_from = last_a
                else:
                    keep_from = last_b

            waves, spans = slice_windows(pcm, self.sample_rate, regions, offset)

            # Retain a WIN_SEC tail so a window straddling the seam is not lost.
            keep_from = max(offset, keep_from - WIN_SEC)
            cut = int((keep_from - offset) * self.sample_rate) * 2
            if cut > 0:
                del self._pending[:cut]
                self._pending_offset = keep_from

            if not waves:
                return

            t0 = time.perf_counter()
            embs = await asyncio.to_thread(self.embedder.embed_batch, waves)
            self._embed_ms += (time.perf_counter() - t0) * 1000
            self._passes += 1

            new = [Window(s, e, v) for (s, e), v in zip(spans, embs) if v is not None]
            if not new:
                return

            self.engine.add_windows(new)
            changed = await asyncio.to_thread(self.engine.recluster)

            tl = self.engine.timeline()
            if tl:
                self._covered_until = max(self._covered_until, tl[-1][1])
            self._coverage.set()
            self._coverage.clear()

            log.debug(
                "pass %d: +%d window(s) in %.0f ms, covered to %.1fs, %d speaker(s)",
                self._passes,
                len(new),
                (time.perf_counter() - t0) * 1000,
                self._covered_until,
                self.engine.speaker_count(),
            )

        if changed and self.on_change:
            await self.on_change()

    async def finalize(self) -> None:
        """Flush the tail and run one last pass with every window available.

        At this point it is ordinary offline diarization — the most accurate
        labelling the system can produce — so it is always worth doing before
        the client stops listening.
        """
        if not self.enabled:
            return
        await self._pass(final=True)
        changed = await asyncio.to_thread(self.engine.recluster)
        if changed and self.on_change:
            await self.on_change()

    # ------------------------------------------------------------------ query

    async def wait_for_coverage(self, until: float, timeout: float = 0.9) -> bool:
        """Block briefly until the timeline reaches `until` seconds.

        Bounded on purpose. Waiting a few hundred ms so the FIRST render of a
        line already carries the right name is worth it; waiting a second and
        a half is not, because a `refresh` will fix it anyway.
        """
        if not self.enabled:
            return False
        deadline = time.monotonic() + timeout
        while self._covered_until < until:
            left = deadline - time.monotonic()
            if left <= 0:
                return False
            self._wake.set()
            try:
                await asyncio.wait_for(self._coverage.wait(), timeout=min(left, 0.25))
            except asyncio.TimeoutError:
                continue
        return True

    def label_for(self, start: float, end: float) -> Optional[int]:
        return self.engine.label_for(start, end) if self.enabled else 0

    def timeline(self) -> list[tuple[float, float, int]]:
        return self.engine.timeline()

    def speaker_count(self) -> int:
        return self.engine.speaker_count()

    def split_points(self, start: float, end: float) -> list[float]:
        """Speaker-change times strictly inside [start, end).

        This is what replaces v7's hand-rolled change-point detector: the cut
        points come from the same clustering that decides identity, so a split
        and its labels can never disagree.
        """
        if not self.enabled:
            return []
        cuts: list[float] = []
        prev = None
        for a, b, s in self.engine.timeline():
            if b <= start or a >= end:
                continue
            if prev is not None and s != prev and start + 0.35 < a < end - 0.35:
                cuts.append(a)
            prev = s
        return cuts

    def stats(self) -> dict:
        return {
            "passes": self._passes,
            "windows": self.engine.n_windows(),
            "speakers": self.engine.speaker_count(),
            "avg_embed_ms": round(self._embed_ms / max(1, self._passes), 1),
            "covered_until": round(self._covered_until, 2),
        }

    def reset(self) -> None:
        self.engine.reset()
        self.embedder.reset()
        self._pending.clear()
        self._pending_offset = 0.0
        self._fed_until = 0.0
        self._covered_until = 0.0
