"""
diarizer_sortformer.py — NVIDIA Streaming Sortformer backend (optional).

WHAT THIS BUYS YOU OVER THE EMBEDDING BACKEND
---------------------------------------------
`diarization_service.py` cuts speech into windows, embeds each one, and
clusters. That is the classical cascade, and it has one structural limit no
amount of tuning removes: a window is assigned to exactly ONE speaker, so when
two people talk at the same time, one of them disappears from the transcript.

Sortformer is end-to-end. It emits a per-frame activity probability for each of
4 speaker slots independently, so overlapping speech is representable rather
than something the model has to pick a winner for. It also sidesteps the
permutation problem entirely: speakers come out in arrival-time order, held in
an Arrival-Order Speaker Cache, so slot 0 is whoever spoke first and stays that
way — no clustering step, no threshold, no Hungarian matching within a window.

Reported DER on CALLHOME 2-speaker is 6.57% (1.04s latency config) against a
classical cascade's roughly 10-15% on the same data.

THREE THINGS TO KNOW BEFORE YOU SWITCH
--------------------------------------
1. IT REALLY WANTS A GPU. NVIDIA report RTF 0.002-0.18 depending on config, on
   an RTX 6000 Ada. On CPU expect roughly 20-50x that. Your ECS Fargate target
   has no GPU by default, so this is an infrastructure decision as much as a
   code one.

2. MAXIMUM 4 SPEAKERS. Performance degrades at 5+. The embedding backend has
   no such ceiling. If SLT sessions can run to six people, keep the embedding
   backend.

3. IT WAS TRAINED PRIMARILY ON ENGLISH — the model card says so explicitly,
   and warns that performance may degrade on non-English speech. Diarization
   is mostly language-independent (it keys on voice, not words) and NVIDIA
   report strong Mandarin results, but Sinhala and Tamil are untested here.
   **A/B this against the embedding backend on your own recordings before
   trusting it.** Do not assume the DER numbers above transfer.

WHY A ROLLING WINDOW RATHER THAN TRUE INCREMENTAL FEEDING
---------------------------------------------------------
NeMo's public surface is `diarize(audio=...)`, a whole-input call; the
"streaming" is chunked processing with a speaker cache *inside* that call, not
an incremental feed API. Re-running it on the whole session every pass would
grow without bound, so this backend re-runs it on the last WINDOW_SEC and
stitches each result onto the session timeline by matching speaker slots on the
overlap region.

That also flips which config you want. NVIDIA's table trades input-buffer
latency against RTF — but input-buffer latency only costs you something when
audio is arriving one chunk at a time. Here the window is already in hand, so
the "very high latency" row is strictly better: same accuracy, RTF 0.002
instead of 0.093. That is the difference between 0.18s and 8.4s per pass on a
90-second window.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

log = logging.getLogger("voxlive.sortformer")

MODEL_ID = "nvidia/diar_streaming_sortformer_4spk-v2.1"

# Streaming config, in 80 ms frames. This is NVIDIA's "very high latency" row:
# lowest RTF, and the input-buffer latency it trades away is free for us
# because we re-run on a window we already hold.
STREAM_CFG = {
    "chunk_len": 340,
    "chunk_right_context": 40,
    "fifo_len": 40,
    "spkcache_update_period": 300,
    "spkcache_len": 188,
}

_MODEL_CACHE: dict = {}


def _load(device: Optional[str] = None):
    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dev in _MODEL_CACHE:
        return _MODEL_CACHE[dev]

    if dev == "cpu":
        log.warning(
            "Sortformer on CPU: expect roughly 20-50x the published RTF. "
            "Use the embedding backend unless a GPU is available."
        )

    log.info("loading %s on %s ...", MODEL_ID, dev)
    model = SortformerEncLabelModel.from_pretrained(MODEL_ID, map_location=dev)
    model.eval()
    for k, v in STREAM_CFG.items():
        setattr(model.sortformer_modules, k, v)
    # Fails loudly on a bad combination rather than producing quiet garbage.
    model.sortformer_modules._check_streaming_parameters()
    _MODEL_CACHE[dev] = model
    log.info("Sortformer ready on %s (max 4 speakers)", dev)
    return model


def warmup(device: Optional[str] = None) -> None:
    model = _load(device)
    model.diarize(
        audio=[np.zeros(16000 * 5, dtype=np.float32)], batch_size=1, sample_rate=16000
    )
    log.info("Sortformer warm")


class SortformerDiarizer:
    """Drop-in alternative to DiarizationService.

    Implements exactly the surface main.py depends on: feed / start / aclose /
    finalize / wait_for_coverage / label_for / split_points / timeline / stats
    / reset. Nothing else in the system needs to know which backend is running.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        interval_sec: float = 2.0,
        window_sec: float = 90.0,
        min_activity: float = 0.5,
        min_turn_sec: float = 0.32,
        device: Optional[str] = None,
        enabled: bool = True,
        max_speakers: int = 4,
        **_ignored,
    ):
        self.sample_rate = sample_rate
        self.interval = float(interval_sec)
        self.window_sec = float(window_sec)
        self.min_activity = float(min_activity)
        self.min_turn_sec = float(min_turn_sec)
        self.device = device
        self.enabled = bool(enabled)
        self.max_speakers = min(4, int(max_speakers))

        self._audio = bytearray()  # rolling window only
        self._audio_offset = 0.0  # session seconds at _audio[0]
        self._fed_until = 0.0
        self._covered_until = 0.0

        # Session timeline, already stitched into a stable naming.
        self._timeline: list[tuple[float, float, int]] = []
        self._max_sid = -1

        self._model = None
        self._task: Optional[asyncio.Task] = None
        self._wake = asyncio.Event()
        self._coverage = asyncio.Event()
        self._lock = asyncio.Lock()
        self._stopping = False
        self.on_change: Optional[Callable[[], Awaitable[None]]] = None

        self._passes = 0
        self._infer_ms = 0.0

    # ------------------------------------------------------------------ input

    def feed(self, pcm_bytes: bytes) -> None:
        if not self.enabled or not pcm_bytes:
            return
        self._audio.extend(pcm_bytes)
        self._fed_until += len(pcm_bytes) / 2 / self.sample_rate

        # Trim to the rolling window. Identity across the seam is preserved by
        # the stitching step, not by keeping the audio.
        max_bytes = int(self.window_sec * self.sample_rate) * 2
        if len(self._audio) > max_bytes:
            drop = len(self._audio) - max_bytes
            del self._audio[:drop]
            self._audio_offset += drop / 2 / self.sample_rate

        if self._fed_until - self._covered_until >= self.interval:
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
                await asyncio.wait_for(self._task, timeout=20)
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
                log.error("Sortformer pass failed: %s", exc, exc_info=True)

    # --------------------------------------------------------------- one pass

    async def _pass(self) -> None:
        async with self._lock:
            if len(self._audio) < self.sample_rate * 2:  # < 1 s
                return
            pcm = np.frombuffer(bytes(self._audio), dtype=np.int16)
            pcm = pcm.astype(np.float32) / 32768.0
            offset = self._audio_offset

            t0 = time.perf_counter()
            segs = await asyncio.to_thread(self._infer, pcm)
            self._infer_ms += (time.perf_counter() - t0) * 1000
            self._passes += 1
            if segs is None:
                return

            window = [
                (a + offset, b + offset, s)
                for a, b, s in segs
                if b - a >= self.min_turn_sec and s < self.max_speakers
            ]
            window.sort()
            changed = self._stitch(window, offset)

            if self._timeline:
                self._covered_until = max(self._covered_until, self._timeline[-1][1])
            self._coverage.set()
            self._coverage.clear()

            log.debug(
                "pass %d: %d run(s) in %.0f ms, covered to %.1fs",
                self._passes,
                len(window),
                (time.perf_counter() - t0) * 1000,
                self._covered_until,
            )

        if changed and self.on_change:
            await self.on_change()

    def _infer(self, pcm: np.ndarray):
        """Run the model. Returns [(start_s, end_s, speaker_idx)] or None."""
        if self._model is None:
            self._model = _load(self.device)
        try:
            out = self._model.diarize(
                audio=[pcm], batch_size=1, sample_rate=self.sample_rate
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Sortformer inference failed: %s", exc)
            return None

        segs = []
        for item in (out[0] if out else []):
            # NeMo returns either "start end speaker" strings or tuples,
            # depending on version. Accept both rather than pinning a version.
            if isinstance(item, str):
                parts = item.replace(",", " ").split()
                if len(parts) < 3:
                    continue
                a, b, spk = float(parts[0]), float(parts[1]), parts[2]
            else:
                a, b, spk = float(item[0]), float(item[1]), item[2]
            idx = int(str(spk).replace("speaker_", "").strip() or 0)
            segs.append((a, b, idx))
        return segs

    # ------------------------------------------------------------- stitching

    def _stitch(self, window: list, offset: float) -> bool:
        """Merge a window's local speaker slots into the session timeline.

        Sortformer numbers speakers by arrival order *within the input it was
        given*. Across two overlapping windows those numberings agree only by
        luck, so slot 0 in this pass may be slot 1 in the session. Match them
        on the region the two share, by total overlapping speech, and carry the
        session's names forward.
        """
        prev = self._timeline
        if not prev:
            self._timeline = window
            self._max_sid = max((s for _, _, s in window), default=-1)
            return bool(window)

        # The part of the session timeline this window also covers.
        overlap = [(a, b, s) for a, b, s in prev if b > offset]
        local_ids = sorted({s for _, _, s in window})
        sess_ids = sorted({s for _, _, s in overlap})

        mapping: dict[int, int] = {}
        if overlap and local_ids and sess_ids:
            cost = np.zeros((len(local_ids), len(sess_ids)))
            for i, li in enumerate(local_ids):
                for j, sj in enumerate(sess_ids):
                    shared = 0.0
                    for a1, b1, s1 in window:
                        if s1 != li:
                            continue
                        for a2, b2, s2 in overlap:
                            if s2 != sj:
                                continue
                            ov = min(b1, b2) - max(a1, a2)
                            if ov > 0:
                                shared += ov
                    cost[i, j] = -shared
            rows, cols = linear_sum_assignment(cost)
            mapping = {
                local_ids[r]: sess_ids[c] for r, c in zip(rows, cols) if cost[r, c] < 0
            }

        nxt = self._max_sid + 1
        for li in local_ids:
            if li in mapping:
                continue
            # A local slot with no overlap evidence is USUALLY a genuinely new
            # participant. But it can also be someone who simply said nothing
            # during the overlap region — Sortformer has no memory across two
            # separate `diarize()` calls, so there is no acoustic way to tell
            # from here. Minting freely would let one person collect several
            # ids over a long meeting, so the count is capped.
            if nxt >= self.max_speakers and self._timeline:
                mapping[li] = self._nearest_prior_speaker(window, li)
                log.debug(
                    "slot %d unmatched at the speaker cap; folded into %d",
                    li,
                    mapping[li],
                )
                continue
            mapping[li] = nxt
            nxt += 1
        self._max_sid = max(self._max_sid, nxt - 1)

        merged = [(a, b, s) for a, b, s in prev if b <= offset]
        merged += [(a, b, mapping[s]) for a, b, s in window]
        merged.sort()

        # Collapse runs the re-run split at a chunk edge.
        out: list[tuple[float, float, int]] = []
        for a, b, s in merged:
            if out and out[-1][2] == s and a - out[-1][1] < 0.12:
                out[-1] = (out[-1][0], max(out[-1][1], b), s)
            else:
                out.append((a, b, s))

        changed = out != prev
        self._timeline = out
        return changed

    def _nearest_prior_speaker(self, window: list, local_id: int) -> int:
        """Last resort at the speaker cap: whoever was talking most recently
        before this slot's first appearance. Conversation is sticky, so
        turn-taking adjacency is the only signal left once acoustic evidence
        has run out."""
        first = min((a for a, _, s in window if s == local_id), default=0.0)
        best, best_gap = 0, float("inf")
        for a, b, s in self._timeline:
            if b <= first and (first - b) < best_gap:
                best, best_gap = s, first - b
        return best

    async def finalize(self) -> None:
        if not self.enabled:
            return
        await self._pass()

    # ------------------------------------------------------------------ query

    async def wait_for_coverage(self, until: float, timeout: float = 0.9) -> bool:
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
        if not self.enabled:
            return 0
        totals: dict[int, float] = {}
        for a, b, s in self._timeline:
            ov = min(end, b) - max(start, a)
            if ov > 0:
                totals[s] = totals.get(s, 0.0) + ov
        return max(totals, key=totals.get) if totals else None

    def split_points(self, start: float, end: float) -> list[float]:
        cuts: list[float] = []
        prev = None
        for a, b, s in self._timeline:
            if b <= start or a >= end:
                continue
            if prev is not None and s != prev and start + 0.35 < a < end - 0.35:
                cuts.append(a)
            prev = s
        return cuts

    def timeline(self) -> list[tuple[float, float, int]]:
        return list(self._timeline)

    def speaker_count(self) -> int:
        return len({s for _, _, s in self._timeline})

    # `main.py` reads diar.engine.speaker_count(); expose the same shape so the
    # two backends stay interchangeable without a conditional at the call site.
    @property
    def engine(self):
        return self

    def stats(self) -> dict:
        return {
            "backend": "sortformer",
            "passes": self._passes,
            "speakers": self.speaker_count(),
            "avg_infer_ms": round(self._infer_ms / max(1, self._passes), 1),
            "covered_until": round(self._covered_until, 2),
        }

    def reset(self, at: Optional[float] = None) -> None:
        """Forget every speaker identity, preserving the session clock.

        Signature matches DiarizationService.reset so `SessionState` can call
        the new-recording control without knowing which backend is loaded.
        """
        t = 0.0 if at is None else float(at)
        self._audio.clear()
        self._audio_offset = t
        self._fed_until = t
        self._covered_until = t
        self._timeline.clear()
        self._max_sid = -1
