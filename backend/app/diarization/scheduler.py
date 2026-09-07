"""
scheduler.py — the process-wide diarization inference limiter.

It answers exactly one question:

    "How much diarization work is allowed to execute right now?"

WHY A SHARED MODEL IS NOT ENOUGH
--------------------------------
`embedder.py` already caches the WeSpeaker weights process-wide, which is the
right call: 500 sessions share one copy instead of loading 500. But sharing
the weights bounds MEMORY, not CONCURRENCY. Every session still runs its own
background loop, and every loop dispatches its own work:

    500 sessions -> 500 diarization loops -> 500 concurrent to_thread() calls
                                          -> one shared model

`asyncio.to_thread` uses the default executor, which sizes itself from the CPU
count and has no application-level ceiling. Under load the result is more
inference requests in flight than the machine has cores to serve, and every
one of them gets slower together.

This scheduler puts a single ceiling in front of all of it:

    session A ─┐
    session B ─┼─► semaphore(max_concurrency) ─► bounded thread pool ─► model
    session C ─┘

WHY THE THREAD POOL IS SEPARATE FROM asyncio's DEFAULT
------------------------------------------------------
The default executor is shared with anything else that calls `to_thread`. A
dedicated pool means diarization can never starve unrelated work, and its
size is a number you set deliberately rather than one Python picked.

A NOTE ON WHAT THIS DOES *NOT* FIX
----------------------------------
Limiting concurrency does not make a slow pass fast. `SpeakerEngine`'s
per-pass cost grows with SESSION LENGTH, because `_label_and_build()` runs on
every pass over every window in the session and its median filter is a
Python-level loop over session frames. Long sessions therefore get more
expensive over time regardless of how many run at once, and that is a separate
fix in `speaker_engine.py` — measure it in the single-instance benchmark
before sizing a fleet around it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Optional

log = logging.getLogger("voxlive.diar.scheduler")


class DiarizationQueueFull(Exception):
    """Too many diarization passes already waiting.

    Diarization is best-effort by design — the transcript is still correct
    without it, just unlabelled — so a rejection here degrades quality rather
    than breaking the session. The caller should log and skip the pass.
    """


@dataclass
class DiarizationStats:
    submitted: int = 0
    completed: int = 0
    failed: int = 0
    rejected: int = 0

    def as_dict(self) -> dict:
        return {
            "submitted": self.submitted,
            "completed": self.completed,
            "failed": self.failed,
            "rejected": self.rejected,
        }


class DiarizationScheduler:
    """Global concurrency limit for diarization CPU/GPU work.

    Unlike the ASR scheduler this has no worker pool of its own. ASR work is
    I/O to a remote provider, so a queue plus workers is the right shape; this
    is local compute, so the correct control is "how many may run at once"
    plus "how many may wait". A semaphore and a waiter count express that in
    far less machinery.
    """

    def __init__(
        self,
        max_concurrency: int = 2,
        queue_maxsize: int = 32,
        thread_name_prefix: str = "diar",
    ):
        self.max_concurrency = max(1, int(max_concurrency))
        self.queue_maxsize = max(1, int(queue_maxsize))
        self._sem = asyncio.Semaphore(self.max_concurrency)
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_concurrency,
            thread_name_prefix=thread_name_prefix,
        )
        self._waiting = 0
        self._running = 0
        self.stats = DiarizationStats()
        self._latencies: list[float] = []

    # ------------------------------------------------------------------ run

    async def run(self, fn: Callable[..., Any], /, *args: Any) -> Any:
        """Run a blocking function under the global limit.

        Drop-in for `asyncio.to_thread(fn, *args)` — same call shape, same
        return value, plus a ceiling and metrics.
        """
        if self._waiting >= self.queue_maxsize:
            self.stats.rejected += 1
            raise DiarizationQueueFull(
                f"{self._waiting} diarization pass(es) already waiting "
                f"(max {self.queue_maxsize})"
            )

        self.stats.submitted += 1

        # Acquire and release are spelled out rather than using `async with`,
        # so the waiter count is decremented on the CANCELLED path too. A
        # session that disconnects mid-wait must not leave the limiter
        # believing it is one pass fuller than it is, forever.
        self._waiting += 1
        try:
            await self._sem.acquire()
        finally:
            self._waiting -= 1

        self._running += 1
        started = time.monotonic()
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(self._executor, fn, *args)
        except Exception:
            self.stats.failed += 1
            raise
        else:
            self._record_latency((time.monotonic() - started) * 1000)
            self.stats.completed += 1
            return result
        finally:
            self._running -= 1
            self._sem.release()

    # ------------------------------------------------------------- shutdown

    def shutdown(self, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=not wait)
        log.info("diarization scheduler stopped: %s", self.stats.as_dict())

    # ------------------------------------------------------------ telemetry

    def _record_latency(self, ms: float) -> None:
        self._latencies.append(ms)
        if len(self._latencies) > 512:
            del self._latencies[: len(self._latencies) - 512]

    def _percentile(self, p: float) -> float:
        if not self._latencies:
            return 0.0
        ordered = sorted(self._latencies)
        idx = min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1))))
        return round(ordered[idx], 1)

    def snapshot(self) -> dict:
        return {
            **self.stats.as_dict(),
            "max_concurrency": self.max_concurrency,
            "running": self._running,
            "queue_depth": self._waiting,
            "queue_maxsize": self.queue_maxsize,
            "p50_latency_ms": self._percentile(50),
            "p95_latency_ms": self._percentile(95),
        }


# ---------------------------------------------------------------------------
# Process-wide default.
#
# DiarizationService is constructed per session, deep inside the factory, and
# threading a scheduler through every call site would touch a lot of code for
# no benefit. So the service takes an OPTIONAL scheduler and falls back to
# this one. `configure()` is called once from the lifespan.
# ---------------------------------------------------------------------------

_default: Optional[DiarizationScheduler] = None


def configure(max_concurrency: int, queue_maxsize: int) -> DiarizationScheduler:
    """Install the process-wide scheduler. Call once, at start-up."""
    global _default
    if _default is not None:
        _default.shutdown(wait=False)
    _default = DiarizationScheduler(
        max_concurrency=max_concurrency, queue_maxsize=queue_maxsize
    )
    log.info(
        "diarization scheduler configured: concurrency=%d queue_maxsize=%d",
        max_concurrency,
        queue_maxsize,
    )
    return _default


def get_scheduler() -> DiarizationScheduler:
    """The process-wide scheduler, created with defaults if `configure()` was
    never called — so tests and `bench_diarization.py` work unchanged."""
    global _default
    if _default is None:
        _default = DiarizationScheduler()
    return _default


def shutdown() -> None:
    global _default
    if _default is not None:
        _default.shutdown(wait=False)
        _default = None
