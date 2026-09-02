"""Regression tests for real-time diarization failure modes seen in production logs."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from app.diarization.speaker_engine import Window, l2norm
from app.diarization.stable_speaker_engine import StableSpeakerEngine


def test_overlap_tail_windows_do_not_duplicate_shifted_history():
    """A re-windowed overlap tail may shift slightly; old starts must still be ignored."""
    eng = StableSpeakerEngine(calibrate=False, detect_turns=False)
    e = l2norm(np.array([1.0, 0.0, 0.0]))
    eng.add_windows([
        Window(0.0, 2.0, e.copy()),
        Window(0.75, 2.75, e.copy()),
    ])

    # The next pass starts its retained VAD tail on a shifted grid. 0.50 s is
    # historical overlap; 1.25 s advances the timeline and is genuinely new.
    eng.add_windows([
        Window(0.50, 2.50, e.copy()),
        Window(1.25, 3.25, e.copy()),
    ])

    assert eng.n_windows() == 3, "shifted overlap history was counted as new evidence"


def test_reset_allows_a_new_recording_to_start_at_zero():
    eng = StableSpeakerEngine(calibrate=False, detect_turns=False)
    e = l2norm(np.array([1.0, 0.0, 0.0]))
    eng.add_windows([Window(10.0, 12.0, e.copy())])
    eng.reset()
    eng.add_windows([Window(0.0, 2.0, e.copy())])
    assert eng.n_windows() == 1


def test_strong_short_interjection_can_create_new_identity():
    """A coherent ~4 s minority turn must not require the old 6 s discovery floor."""
    eng = StableSpeakerEngine(
        calibrate=False,
        detect_turns=False,
        new_identity_min_sec=6.0,
        new_identity_min_windows=4,
        new_identity_short_sec=3.2,
        new_identity_short_windows=2,
        new_identity_strong_dist=0.68,
    )

    known = np.stack([l2norm(np.array([1.0, 0.0, 0.0]))])
    x = np.stack([
        l2norm(np.array([0.0, 1.0, 0.02])),
        l2norm(np.array([0.0, 1.0, -0.02])),
    ])
    dur = np.array([2.0, 2.0])

    fresh = eng._discover(x, dur, known)
    assert fresh is not None, "strong 4 s minority speaker was ignored"
    assert float(1.0 - np.dot(fresh, known[0])) >= 0.68


def test_short_discovery_tunables_are_not_silently_ignored():
    eng = StableSpeakerEngine(
        new_identity_min_sec=7.0,
        new_identity_min_windows=5,
        new_identity_short_sec=2.8,
        new_identity_short_windows=2,
        new_identity_strong_dist=0.72,
    )

    assert eng.new_identity_min_sec == 7.0
    assert eng.new_identity_min_windows == 5
    assert eng.new_identity_short_sec == 2.8
    assert eng.new_identity_short_windows == 2
    assert eng.new_identity_strong_dist == 0.72
