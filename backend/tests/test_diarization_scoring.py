"""Diarization coverage: the timeline must reach a segment BEFORE it is cut.

WHY THIS FILE EXISTS
--------------------
`SessionState._split` asks the diarizer for speaker-change times and cuts the
segment there. That happens once, before ASR, and the resulting chunk
boundaries are never revised — `relabel()` can move a chunk to another speaker
but cannot divide one. So if the timeline does not reach the segment at the
moment it is cut, a segment containing two speakers becomes one mono-speaker
paragraph permanently, and the final offline pass cannot repair it.

That is the non-overlap diarization failure. These tests pin the mechanism.

No model is loaded: `Embedder` imports pyannote lazily inside `_load`, so
`embed_batch` can be replaced with a deterministic fake.
"""

import asyncio
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("DIARIZATION_MODE", "off")
os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("APP_ENV", "development")

from app.diarization.service import DiarizationService
from app.session.transcript import TranscriptStore

SR = 16000
DIM = 32


def _voice(seed: int) -> np.ndarray:
    """A fixed, well-separated unit vector standing in for a speaker."""
    v = np.zeros(DIM, dtype=np.float64)
    v[seed] = 1.0
    return v


def two_speaker_pcm(seconds=12.0, turn=3.0) -> bytes:
    """Speech-like audio that webrtcvad detects for its whole duration.

    Speaker identity here comes from the fake embeddings (assigned by window
    start time), not from the waveform — so the signal only has to clear the
    VAD. A plain low tone does NOT: webrtcvad rejected the 130 Hz burst used in
    an earlier version of this fixture, which silently removed the first turn
    from every region and made the test measure the fixture, not the code.
    """
    n = int(seconds * SR)
    t = np.arange(n) / SR
    v = sum(np.sin(2 * np.pi * 180 * k * t) / k for k in range(1, 8))
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 5 * t)
    return (v * env * 0.5 * 32767 * 0.6).astype(np.int16).tobytes()


def fake_service(**kw) -> DiarizationService:
    """A service whose embeddings depend only on WHEN a window occurs.

    Windows in even 3 s slots get voice A, odd slots voice B. That makes the
    expected timeline knowable exactly, so a test can assert on coverage rather
    than on whatever a real model happens to think.
    """
    svc = DiarizationService(
        hf_token="fake",
        sample_rate=SR,
        expected_speakers=2,
        interval_sec=1.5,
        enabled=True,
        **kw,
    )
    svc._spans: list = []

    def embed_batch(windows):
        out = []
        for _ in windows:
            out.append(None)
        return out

    svc.embedder.embed_batch = embed_batch
    return svc


def timed_service() -> DiarizationService:
    """As above, but embeddings are chosen from each window's START TIME."""
    svc = DiarizationService(
        hf_token="fake",
        sample_rate=SR,
        expected_speakers=2,
        interval_sec=1.5,
        enabled=True,
    )

    original = svc.engine.add_windows

    def add_windows(windows):
        for w in windows:
            slot = int(w.start // 3.0) % 2
            w.embedding = _voice(0 if slot == 0 else 1)
        original(windows)

    svc.engine.add_windows = add_windows
    svc.embedder.embed_batch = lambda windows: [_voice(0) for _ in windows]
    return svc


# --------------------------------------------------- the coverage regression


def test_coverage_reaches_a_finished_segment():
    """wait_for_coverage must extend the timeline over a CLOSED segment.

    Before the fix this returned False almost every time: `_pass` held back the
    speech region touching the buffer edge — which is precisely where a
    just-ended segment sits — and the caller's 900 ms timeout was shorter than
    the 1.5 s pass interval, so the next pass had not run yet.
    """

    async def go():
        svc = timed_service()
        pcm = two_speaker_pcm(seconds=9.0)
        svc.feed(pcm)
        # No background loop started: this is exactly the case the old code
        # could not serve, because it could only wait for someone else's pass.
        # 9.0 == the end of the finished segment, which is what
        # handle_segment actually asks for.
        covered = await svc.wait_for_coverage(9.0, timeout=5.0)
        return covered, svc._covered_until, svc.engine.timeline()

    covered, until, timeline = asyncio.run(go())
    assert covered is True
    assert until >= 8.0
    assert timeline


def test_split_points_found_across_a_turn_change():
    """With coverage, the cut the session needs actually exists."""

    async def go():
        svc = timed_service()
        svc.feed(two_speaker_pcm(seconds=12.0))
        await svc.wait_for_coverage(12.0, timeout=5.0)
        return svc.split_points(0.0, 11.0), svc.engine.speaker_count()

    cuts, speakers = asyncio.run(go())
    assert speakers == 2
    # Turns change at 3 s, 6 s, 9 s. Exact times depend on VAD framing, so
    # assert that cuts exist and land near a boundary rather than on equality.
    assert cuts, "no speaker-change cuts found — segments would go unsplit"
    assert any(abs(c - b) < 1.0 for c in cuts for b in (3.0, 6.0, 9.0))


def test_in_progress_speech_is_still_held_back():
    """The flush must not consume a sentence that has not finished.

    `cover_until` is allowed to release only regions that END at or before the
    caller's finished segment. Anything later is still held for the next pass —
    cutting live speech is what produces stub windows and false boundaries.
    """

    async def go():
        svc = timed_service()
        svc.feed(two_speaker_pcm(seconds=9.0))
        # Ask only for the first 3 s. The speech running to the buffer edge
        # must not be consumed.
        await svc.wait_for_coverage(3.0, timeout=5.0)
        return svc._pending_offset

    offset = asyncio.run(go())
    assert offset < 9.0


def test_disabled_service_reports_no_coverage():
    async def go():
        svc = DiarizationService(hf_token="fake", sample_rate=SR, enabled=False)
        return await svc.wait_for_coverage(5.0, timeout=0.1)

    assert asyncio.run(go()) is False


# ------------------------------------------------------- the final-pass fix


def test_finalize_forces_a_full_recluster():
    """`finalize()` must re-derive clusters, not reuse stale centroids.

    `recluster()` only re-derives after RECLUSTER_AFTER_SEC (4 s) of new
    trusted audio. The final flush usually adds less than that, so without
    force=True the "best labelling the system can produce" quietly skipped the
    whole-session re-derivation it exists to perform.
    """
    seen = {}

    async def go():
        svc = timed_service()
        svc.feed(two_speaker_pcm(seconds=9.0))
        await svc.wait_for_coverage(9.0, timeout=5.0)

        original = svc.engine.recluster

        def spy(force=False):
            seen["force"] = force
            return original(force=force)

        svc.engine.recluster = spy
        await svc.finalize()

    asyncio.run(go())
    assert seen.get("force") is True


# ------------------------------------------- the diagnostic for what remains


def test_unsplit_chunk_is_detected():
    """A chunk spanning a turn change is counted, because it cannot be fixed."""
    store = TranscriptStore()
    chunk = store.new_chunk(1, 0.0, 9.0)
    chunk.text = "two people talking"
    chunk.speaker = 0

    timeline = [(0.0, 4.0, 0), (4.0, 9.0, 1)]
    count, seconds = store.unsplit_chunks(timeline)

    assert count == 1
    assert seconds == pytest.approx(9.0)


def test_correctly_split_chunks_are_not_counted():
    store = TranscriptStore()
    a = store.new_chunk(1, 0.0, 4.0)
    a.text = "first speaker"
    b = store.new_chunk(1, 4.0, 9.0)
    b.text = "second speaker"

    timeline = [(0.0, 4.0, 0), (4.0, 9.0, 1)]
    count, seconds = store.unsplit_chunks(timeline)

    assert count == 0
    assert seconds == 0.0


def test_unsplit_ignores_chunks_with_no_text():
    store = TranscriptStore()
    store.new_chunk(1, 0.0, 9.0)  # dropped by a guard, never transcribed
    assert store.unsplit_chunks([(0.0, 4.0, 0), (4.0, 9.0, 1)]) == (0, 0.0)


def test_relabel_cannot_divide_a_chunk():
    """Pins the limitation the diagnostic reports, so it stays documented.

    This is the reason coverage has to be right BEFORE the cut: afterwards,
    the best the system can do is pick which of the two speakers gets the
    whole paragraph.
    """
    store = TranscriptStore()
    c = store.new_chunk(1, 0.0, 9.0)
    c.text = "A talking then B talking"

    def lookup(start, end):
        return 0 if start < 4.0 else 1

    store.relabel(lookup)
    paras = store.paragraphs()

    assert len(paras) == 1
    assert paras[0]["speaker"] == "Speaker 1"
