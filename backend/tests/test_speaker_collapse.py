"""Regression tests for the speaker-collapse defect (supervisor review, P0).

THE DEFECT
----------
Two speakers whose centroids sit CLOSER than SAME_SPEAKER_MAX (0.50) are
vetoed by `_choose_k`, which then falls back to its initialiser `best_k = 1`.
Because v11 re-derived clusters from the whole session every
RECLUSTER_AFTER_SEC, that verdict could arrive AFTER two speakers had already
been shown — so Speaker 2 appeared, then vanished, and the entire transcript
was relabelled Speaker 1.

The review's example numbers are reproduced literally here: a voice pair at
centroid distance ~0.43 (its "Video 2") versus a pair at ~0.75 (its "Video 3").
Same code, same session length, opposite outcome — which is the point. The
behaviour was never random.

WHAT EACH TEST PINS
  test_auto_mode_collapses_a_close_pair   the defect still exists in AUTO, and
                                          is *documented* rather than silently
                                          fixed, because auto genuinely cannot
                                          know the pair is two people
  test_fixed_mode_holds_two_speakers      FIX #1: speaker_mode="fixed" keeps
                                          K=2 on the identical audio
  test_fixed_mode_never_collapses_live    FIX #3: and keeps it across many
                                          live passes, which is the actual
                                          reported symptom
  test_established_identities_survive     FIX #2: once established, a live pass
                                          cannot delete an identity
  test_auto_still_discovers_a_new_voice   the stability fix did not freeze the
                                          engine: a real third voice still
                                          gets in
  test_monologue_caveat_is_real           the review's own caveat, asserted so
                                          nobody is surprised by it later
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from app.diarization.speaker_engine import (
    TRUST_SEC,
    WIN_SEC,
    SpeakerEngine,
    Window,
    l2norm,
)

RNG = np.random.default_rng(11)
DIM = 256
_COMMON = l2norm(np.random.default_rng(0).normal(size=DIM))


def voice(seed: int) -> np.ndarray:
    r = np.random.default_rng(seed)
    return l2norm(0.45 * _COMMON + l2norm(r.normal(size=DIM)))


def near_voice(base: np.ndarray, seed: int, mix: float) -> np.ndarray:
    """A second person who happens to sound similar to `base`.

    mix=1.35 puts the two centroids ~0.43 apart — under SAME_SPEAKER_MAX, so
    the separation veto rejects K=2. This is the review's "Video 2".
    """
    r = np.random.default_rng(seed)
    return l2norm(base + mix * l2norm(r.normal(size=DIM)))


def utterance(base: np.ndarray, jitter: float = 0.75) -> np.ndarray:
    return l2norm(base + jitter * l2norm(RNG.normal(size=DIM)))


def build(turns, start: float = 0.0):
    assert WIN_SEC >= TRUST_SEC, "test windows must be trusted or this is vacuous"
    wins, t = [], start
    for base, n in turns:
        for _ in range(n):
            wins.append(Window(t, t + WIN_SEC, utterance(base)))
            t += 0.75
        t += 0.5
    return wins, t


def n_speakers(eng: SpeakerEngine) -> int:
    return len({w.prev for w in eng.windows if w.prev >= 0})


def stream(eng: SpeakerEngine, turns, start=0.0, chunk=4):
    """Feed windows a few at a time with a recluster between, the way the live
    service does. One `add_windows` + one `recluster` per pass."""
    wins, end = build(turns, start)
    for i in range(0, len(wins), chunk):
        eng.add_windows(wins[i : i + chunk])
        eng.recluster()
    return end


# ---------------------------------------------------------------- the defect


def test_auto_mode_collapses_a_close_pair():
    """AUTO on a ~0.43-separated pair returns ONE speaker. This is the veto."""
    a = voice(21)
    b = near_voice(a, 22, mix=1.35)
    eng = SpeakerEngine(expected_speakers=0, max_speakers=6)
    wins, _ = build([(a, 8), (b, 8), (a, 6), (b, 6)])
    eng.add_windows(wins)
    eng.recluster()
    n = n_speakers(eng)
    assert n == 1, f"expected the documented collapse to 1, got {n}"
    print(f"  AUTO, centroids ~0.43 apart: collapses to {n} speaker (the defect)  OK")


def test_fixed_mode_holds_two_speakers():
    """FIX #1. Identical audio, speaker_mode='fixed' -> exactly 2."""
    a = voice(21)
    b = near_voice(a, 22, mix=1.35)
    eng = SpeakerEngine(expected_speakers=2, speaker_mode="fixed")
    wins, _ = build([(a, 8), (b, 8), (a, 6), (b, 6)])
    eng.add_windows(wins)
    eng.recluster()
    n = n_speakers(eng)
    assert n == 2, f"fixed K=2 produced {n} speaker(s)"
    print(f"  FIXED K=2, same audio: {n} speakers held  OK")


def test_fixed_mode_never_collapses_live():
    """FIX #3 and the actual reported symptom.

    Stream the close pair through ~20 live passes. v11 showed 2 and then
    dropped to 1 the moment a re-derivation vetoed the split. The count must
    never fall below 2 once it has been reached.
    """
    a = voice(31)
    b = near_voice(a, 32, mix=1.35)
    eng = SpeakerEngine(expected_speakers=2, speaker_mode="fixed")

    seen, counts = False, []
    wins, _ = build([(a, 10), (b, 10), (a, 10), (b, 10), (a, 8), (b, 8)])
    for i in range(0, len(wins), 3):
        eng.add_windows(wins[i : i + 3])
        eng.recluster()
        n = n_speakers(eng)
        counts.append(n)
        if n >= 2:
            seen = True
        elif seen:
            raise AssertionError(
                f"COLLAPSE: speaker count fell to {n} after reaching 2 "
                f"(sequence: {counts})"
            )
    assert eng.established(), "identities should be established after this much audio"
    assert counts[-1] == 2, f"ended at {counts[-1]} speaker(s): {counts}"
    print(f"  FIXED K=2 across {len(counts)} live passes: never collapsed  OK")


def test_established_identities_survive_live_passes():
    """FIX #2. AUTO, two clearly separate voices, then a long single-speaker
    stretch. Once established, no live pass may delete the quiet speaker."""
    a, b = voice(41), voice(42)
    eng = SpeakerEngine(expected_speakers=0, max_speakers=6)
    end = stream(eng, [(a, 8), (b, 8), (a, 8), (b, 8)])
    assert n_speakers(eng) == 2, "setup failed: two separate voices should give 2"
    assert eng.established(), "should be established after ~30s of trusted audio"

    # Now 40 more windows of ONLY speaker A — the situation where a fresh
    # whole-session K search is most tempted to decide there is just one person.
    end = stream(eng, [(a, 40)], start=end)
    n = n_speakers(eng)
    assert n == 2, f"established Speaker 2 was destroyed by live passes: {n}"
    print(f"  AUTO established: survived 40 windows of one speaker, still {n}  OK")


def test_auto_still_discovers_a_new_voice():
    """The stability fix must not freeze the engine. A genuine third person,
    consistently unlike both known voices, still gets an identity."""
    a, b, c = voice(51), voice(52), voice(53)
    eng = SpeakerEngine(expected_speakers=0, max_speakers=6)
    end = stream(eng, [(a, 8), (b, 8), (a, 8), (b, 8)])
    assert n_speakers(eng) == 2 and eng.established(), "setup failed"

    end = stream(eng, [(c, 14), (a, 6), (c, 10)], start=end)
    n = n_speakers(eng)
    assert n == 3, f"a real third voice was not discovered: {n} speaker(s)"
    print(f"  AUTO established: genuine third voice still discovered -> {n}  OK")


def test_monologue_caveat_is_real():
    """The review's own caveat, asserted rather than hoped away.

    'If the user selects 2 while only one person actually speaks, a fixed-K
    diarizer can split one person's voice into two clusters.' It does. The UI
    contract for FIXED is 'assume exactly N', so this is correct behaviour --
    and it is exactly why AUTO stays the default.
    """
    a = voice(61)
    eng = SpeakerEngine(expected_speakers=2, speaker_mode="fixed")
    wins, _ = build([(a, 24)])
    eng.add_windows(wins)
    eng.recluster()
    assert n_speakers(eng) == 2, "fixed K=2 should split even a monologue"

    auto = SpeakerEngine(expected_speakers=2)  # mode defaults to auto
    wins, _ = build([(a, 24)])
    auto.add_windows(wins)
    auto.recluster()
    assert n_speakers(auto) == 1, "AUTO with a ceiling of 2 must keep a monologue at 1"
    print("  FIXED splits a monologue; AUTO does not — both as designed  OK")


if __name__ == "__main__":
    print("Speaker-collapse regression tests (supervisor review P0)")
    for fn in [
        test_auto_mode_collapses_a_close_pair,
        test_fixed_mode_holds_two_speakers,
        test_fixed_mode_never_collapses_live,
        test_established_identities_survive_live_passes,
        test_auto_still_discovers_a_new_voice,
        test_monologue_caveat_is_real,
    ]:
        fn()
    print("all passed")
