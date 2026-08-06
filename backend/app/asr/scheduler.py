"""
scheduler.py — the process-wide ASR capacity controller.

It answers exactly one question:

    "How much ASR work is allowed to execute right now?"

THE PROBLEM THIS FIXES
----------------------
The baseline gave EVERY session its own semaphore:

    Session 1 -> Semaphore(6)
    Session 2 -> Semaphore(6)
    ...
    Session 500 -> Semaphore(6)          =  up to 3,000 concurrent ASR calls

Nothing in the system knew the total. Capacity was a per-session opinion
multiplied by however many browsers happened to connect, which is not a
capacity policy — it is an accident. Worse, the 429 backoff was also
per-session: 500 sessions each discovered the provider was rate-limited
independently, each slept `2**attempt`, and each woke up and retried into the
same wall. That is a retry storm.

    ONE global capacity, shared by every session in this process:

                       submit()
                          |
                   bounded queue  --- full? -> ASRQueueFull (backpressure)
                          |
                  N ASR workers (N = max_concurrency)
                          |
                    provider.transcribe_segment()
                          |
                  timeout / retry / jitter / global cooldown

FOUR THINGS THIS ADDS THAT THE BASELINE DID NOT HAVE
----------------------------------------------------
1. A GLOBAL CEILING. `max_concurrency` is the total for the process, derived
   from provider quota and measured latency — not from how many people are
   connected.

2. A BOUNDED QUEUE. When work arrives faster than it can be processed, the
   queue fills and `submit()` raises `ASRQueueFull` immediately instead of
   growing without limit. Failing one segment fast is better than degrading
   every session slowly.

3. A PER-CALL TIMEOUT. The baseline awaited the Gemini call with no ceiling.
   A hung call held a slot forever, raised nothing, and was therefore never
   retried — capacity leaked silently with no error in the logs.

4. A GLOBAL 429 COOLDOWN. One worker seeing a 429 pauses ALL workers briefly.
   That is the whole point of centralising: the backoff now reflects the
   provider's state, not one session's bad luck.

SCOPE — WHY THIS IS NOT IN REDIS
--------------------------------
This limits ONE process. Each ECS task runs its own scheduler and the ALB
spreads connections across tasks, so fleet capacity is
`tasks x max_concurrency`. A distributed token bucket only becomes necessary
when the sum across tasks can exceed the provider quota — a real problem, but
one to solve with measurements in hand, not before the first deployment.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional

log = logging.getLogger("voxlive.asr.scheduler")


class ASRQueueFull(Exception):
    """The scheduler is saturated and refused the work.

    This is a healthy, expected signal under overload — not a bug. The caller
    should surface it to the user as "busy, retry" rather than swallow it.
    """


class ASRTimeout(Exception):
    """The provider did not answer within `timeout_sec`."""


def is_rate_limit(err: Any) -> bool:
    s = str(err)
    return "429" in s or "RESOURCE_EXHAUSTED" in s


@dataclass
class _Job:
    pcm: bytes
    sample_rate: int
    context: Optional[str]
    session_id: str
    segment_id: int
    future: asyncio.Future
    queued_at: float = field(default_factory=time.monotonic)


@dataclass
class ASRStats:
    """Counters for logs, /health and (later) CloudWatch custom metrics.

    `queue_depth` and `p95_latency_ms` are the two the supervisor called out
    as autoscaling signals that CPU alone would miss: CPU can read 45% while
    the queue is 200 deep and users are waiting six seconds.
    """

    submitted: int = 0
    completed: int = 0
    failed: int = 0
    rejected: int = 0
    timeouts: int = 0
    rate_limited: int = 0
    retries: int = 0

    def as_dict(self) -> dict:
        return {
            "submitted": self.submitted,
            "completed": self.completed,
            "failed": self.failed,
            "rejected": self.rejected,
            "timeouts": self.timeouts,
            "rate_limited": self.rate_limited,
            "retries": self.retries,
        }


class ASRScheduler:
    """Process-wide bounded queue plus a fixed pool of ASR workers."""

    def __init__(
        self,
        provider: Any,
        max_concurrency: int = 6,
        queue_maxsize: int = 64,
        timeout_sec: float = 30.0,
        max_retries: int = 3,
        cooldown_sec: float = 2.0,
        max_cooldown_sec: float = 30.0,
    ):
        self.provider = provider
        self.max_concurrency = max(1, int(max_concurrency))
        self.queue_maxsize = max(1, int(queue_maxsize))
        self.timeout_sec = float(timeout_sec)
        self.max_retries = max(1, int(max_retries))
        self.cooldown_sec = float(cooldown_sec)
        self.max_cooldown_sec = float(max_cooldown_sec)

        self._queue: asyncio.Queue[Optional[_Job]] = asyncio.Queue(
            maxsize=self.queue_maxsize
        )
        self._workers: list[asyncio.Task] = []
        self._running = False

        # Global 429 state. `_cooldown_until` is a monotonic deadline every
        # worker respects, so a rate limit pauses the whole process at once.
        self._cooldown_until = 0.0
        self._consecutive_429 = 0

        self.stats = ASRStats()
        self._latencies: list[float] = []  # ms, most recent first, capped

    # ----------------------------------------------------------- life cycle

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._workers = [
            asyncio.create_task(self._worker(i)) for i in range(self.max_concurrency)
        ]
        log.info(
            "ASR scheduler started: concurrency=%d queue_maxsize=%d timeout=%.0fs",
            self.max_concurrency,
            self.queue_maxsize,
            self.timeout_sec,
        )

    async def stop(self) -> None:
        """Drain: stop workers, fail anything still queued rather than hanging
        the callers awaiting those futures."""
        if not self._running:
            return
        self._running = False
        for _ in self._workers:
            self._queue.put_nowait(None)
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

        while not self._queue.empty():
            job = self._queue.get_nowait()
            if job is not None and not job.future.done():
                job.future.set_exception(ASRQueueFull("scheduler shutting down"))
            self._queue.task_done()
        log.info("ASR scheduler stopped: %s", self.stats.as_dict())

    # --------------------------------------------------------------- submit

    async def submit(
        self,
        pcm: bytes,
        sample_rate: int,
        context: Optional[str] = None,
        session_id: str = "-",
        segment_id: int = 0,
    ):
        """Queue one segment and await its transcription.

        Raises ASRQueueFull when the process is already saturated. That
        rejection IS the backpressure: it happens in microseconds, it names
        the reason, and it keeps the queue from becoming the memory leak.
        """
        if not self._running:
            raise ASRQueueFull("scheduler is not running")

        job = _Job(
            pcm=pcm,
            sample_rate=sample_rate,
            context=context,
            session_id=session_id,
            segment_id=segment_id,
            future=asyncio.get_running_loop().create_future(),
        )
        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull:
            self.stats.rejected += 1
            log.warning(
                "ASR queue full (%d) — rejecting segment",
                self.queue_maxsize,
                extra={
                    "service": "asr",
                    "session_id": session_id,
                    "segment_id": segment_id,
                    "event": "asr_rejected",
                },
            )
            raise ASRQueueFull(
                f"ASR queue is full ({self.queue_maxsize} waiting)"
            ) from None

        self.stats.submitted += 1
        return await job.future

    # --------------------------------------------------------------- worker

    async def _worker(self, index: int) -> None:
        while True:
            job = await self._queue.get()
            try:
                if job is None:
                    return
                if job.future.cancelled():
                    # Caller gave up (disconnect) while this sat in the queue.
                    continue
                try:
                    result = await self._transcribe(job)
                except Exception as exc:  # noqa: BLE001
                    if not job.future.done():
                        job.future.set_exception(exc)
                else:
                    if not job.future.done():
                        job.future.set_result(result)
            finally:
                self._queue.task_done()

    async def _transcribe(self, job: _Job):
        wait_ms = (time.monotonic() - job.queued_at) * 1000
        last_err: Optional[Exception] = None

        for attempt in range(self.max_retries):
            await self._await_cooldown()

            started = time.monotonic()
            try:
                result = await asyncio.wait_for(
                    self.provider.transcribe_segment(
                        job.pcm, job.sample_rate, context=job.context
                    ),
                    timeout=self.timeout_sec,
                )
            except asyncio.TimeoutError:
                self.stats.timeouts += 1
                last_err = ASRTimeout(f"no response in {self.timeout_sec:.0f}s")
                log.warning(
                    "ASR timeout after %.0fs (attempt %d/%d)",
                    self.timeout_sec,
                    attempt + 1,
                    self.max_retries,
                    extra={
                        "service": "asr",
                        "session_id": job.session_id,
                        "segment_id": job.segment_id,
                        "event": "asr_timeout",
                    },
                )
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if is_rate_limit(exc):
                    self.stats.rate_limited += 1
                    self._enter_cooldown()
                    log.warning(
                        "ASR rate limited (attempt %d/%d)",
                        attempt + 1,
                        self.max_retries,
                        extra={
                            "service": "asr",
                            "session_id": job.session_id,
                            "segment_id": job.segment_id,
                            "event": "asr_rate_limited",
                        },
                    )
                else:
                    log.warning(
                        "ASR attempt %d/%d failed: %s",
                        attempt + 1,
                        self.max_retries,
                        exc,
                        extra={
                            "service": "asr",
                            "session_id": job.session_id,
                            "segment_id": job.segment_id,
                            "event": "asr_error",
                        },
                    )
            else:
                self._clear_cooldown()
                latency_ms = (time.monotonic() - started) * 1000
                self._record_latency(latency_ms)
                self.stats.completed += 1
                log.info(
                    "asr_completed",
                    extra={
                        "service": "asr",
                        "session_id": job.session_id,
                        "segment_id": job.segment_id,
                        "event": "asr_completed",
                        "latency_ms": round(latency_ms),
                        "queue_wait_ms": round(wait_ms),
                        "attempt": attempt + 1,
                    },
                )
                return result

            if attempt + 1 < self.max_retries:
                self.stats.retries += 1
                # Full jitter. Without it, every retry from every session lands
                # in the same millisecond and rebuilds the thundering herd the
                # backoff was meant to break up.
                delay = min(2.0**attempt, 8.0) * random.random()
                await asyncio.sleep(delay)

        self.stats.failed += 1
        raise last_err if last_err else RuntimeError("ASR failed")

    # -------------------------------------------------------- 429 cooldown

    def _enter_cooldown(self) -> None:
        self._consecutive_429 += 1
        delay = min(
            self.cooldown_sec * (2 ** (self._consecutive_429 - 1)),
            self.max_cooldown_sec,
        )
        self._cooldown_until = max(self._cooldown_until, time.monotonic() + delay)

    def _clear_cooldown(self) -> None:
        self._consecutive_429 = 0

    async def _await_cooldown(self) -> None:
        remaining = self._cooldown_until - time.monotonic()
        if remaining > 0:
            # Small jitter so the workers do not all resume on the same tick.
            await asyncio.sleep(remaining + random.random() * 0.25)

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
            "queue_depth": self._queue.qsize(),
            "queue_maxsize": self.queue_maxsize,
            "cooling_down": time.monotonic() < self._cooldown_until,
            "p50_latency_ms": self._percentile(50),
            "p95_latency_ms": self._percentile(95),
            "p99_latency_ms": self._percentile(99),
        }
