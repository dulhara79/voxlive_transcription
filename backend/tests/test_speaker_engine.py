"""Synthetic tests for SpeakerEngine.

These use fabricated embeddings with realistic WeSpeaker geometry:
same-speaker cosine distance ~0.15-0.40, cross-speaker ~0.70-1.05. The point
is to pin down the behaviours that v7 got wrong, so they cannot regress.

TWO THINGS THIS FILE LEARNED THE HARD WAY
-----------------------------------------
1. WINDOW GEOMETRY IS NOT A FREE PARAMETER. These tests were originally
   written against WIN_SEC 1.5. The engine later moved to 2.00 with
   TRUST_SEC 1.80, and every fabricated window silently became UNTRUSTED --
   so `recluster()` returned False without clustering anything and the whole
   file failed. `build()` now imports the real constants and asserts the
   relationship, so a future geometry change breaks loudly instead of
   quietly turning these into vacuous tests.

2. THE ENGINE HAS NO `_labels`. Per-window labels live on `Window.prev`,
   written by `_label_and_build()`. Read them through `labels_of()` below
   rather than reaching for a private list that no longer exists.
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

RNG = np.random.default_rng(7)
DIM = 256


# A shared component every human voice has (recording channel, language,
# the model's own bias). Without it, random 256-d vectors sit at cosine
# distance ~1.0 and the test is far easier than reality.
_COMMON = l2norm(np.random.default_rng(0).normal(size=DIM))


def voice(seed: int) -> np.ndarray:
    """A speaker identity. Cross-speaker cosine distance lands ~0.70-0.80,
    matching WeSpeaker on real 2 s windows."""
    r = np.random.default_rng(seed)
    return l2norm(0.45 * _COMMON + l2norm(r.normal(size=DIM)))


def utterance(base: np.ndarray, jitter: float = 0.75) -> np.ndarray:
    """One window of one person. jitter=0.75 puts same-speaker cosine distance
    around 0.18-0.25 — the real within-speaker spread at this window length."""
    return l2norm(base + jitter * l2norm(RNG.normal(size=DIM)))


def build(turns, win_sec: float = WIN_SEC, hop: float = 0.75):
    """turns: [(speaker_base, n_windows)] laid out consecutively in time.

    Defaults to the engine's own WIN_SEC so the fabricated windows have the
    same geometry `slice_windows()` produces in production.
    """
    assert win_sec >= TRUST_SEC, (
        f"test windows ({win_sec}s) are below TRUST_SEC ({TRUST_SEC}s) — every "
        "window would be untrusted, recluster() would return False, and these "
        "tests would pass vacuously without ever clustering anything."
    )
    wins, truth, t = [], [], 0.0
    for base, n in turns:
        for _ in range(n):
            wins.append(Window(t, t + win_sec, utterance(base)))
            truth.append(id(base))
            t += hop
        t += 0.5  # inter-turn pause
    return wins, truth


def labels_of(eng: SpeakerEngine) -> list[int]:
    """Per-window display labels, in time order.

    `_label_and_build()` writes the display id onto each Window as `prev`.
    `eng.windows` is kept sorted by start time, and `build()` emits in time
    order, so this stays index-aligned with the `truth` list.
    """
    return [w.prev for w in eng.windows]


def purity(labels, truth):
    """Fraction of windows whose label agrees with the majority label of their
    true speaker (permutation-invariant accuracy)."""
    best = {}
    for tr in set(truth):
        idx = [i for i, x in enumerate(truth) if x == tr]
        vals, counts = np.unique([labels[i] for i in idx], return_counts=True)
        best[tr] = vals[np.argmax(counts)]
    return sum(1 for i, tr in enumerate(truth) if labels[i] == best[tr]) / len(truth)


def test_two_speakers_alternating():
    a, b = voice(1), voice(2)
    wins, truth = build([(a, 6), (b, 5), (a, 4), (b, 6), (a, 3)])
    eng = SpeakerEngine(expected_speakers=2)
    eng.add_windows(wins)
    assert eng.recluster()
    labels = labels_of(eng)
    assert len(set(labels)) == 2, f"expected 2 speakers, got {len(set(labels))}"
    p = purity(labels, truth)
    assert p > 0.95, f"purity {p:.3f}"
    print(f"  two speakers alternating: 2 clusters, purity {p:.3f}  OK")


def test_monologue_with_k_known_is_not_split():
    """THE v7 KILLER. One person, EXPECTED_SPEAKERS=2.

    v7's _cluster() did fcluster(maxclust=K) unconditionally, so a monologue
    was forcibly cut into two 'speakers' and the transcript alternated between
    Speaker 1 and Speaker 2 for a single person.
    """
    a = voice(3)
    wins, _ = build([(a, 24)])
    eng = SpeakerEngine(expected_speakers=2)
    eng.add_windows(wins)
    eng.recluster()
    n = len(set(labels_of(eng)))
    assert n == 1, f"monologue split into {n} speakers"
    print("  monologue with K=2 known: stayed 1 speaker  OK")


def test_three_speakers_auto_k():
    a, b, c = voice(4), voice(5), voice(6)
    wins, truth = build([(a, 6), (b, 6), (c, 6), (a, 5), (c, 5), (b, 4)])
    eng = SpeakerEngine(expected_speakers=0, max_speakers=6)
    eng.add_windows(wins)
    eng.recluster()
    labels = labels_of(eng)
    n = len(set(labels))
    p = purity(labels, truth)
    assert n == 3, f"expected 3, got {n}"
    assert p > 0.95, f"purity {p:.3f}"
    print(f"  three speakers, K unknown: found 3, purity {p:.3f}  OK")


def test_short_tail_cannot_create_a_speaker():
    """A sub-TRUST_SEC window must consume the clustering, never shape it."""
    a, b = voice(7), voice(8)
    wins, _ = build([(a, 8), (b, 8)])
    # A 0.6 s garbage tail — the kind of thing that spawned 'Speaker 9' in v7.
    # Deliberately below TRUST_SEC: it should be labelled but get no vote.
    tail = Window(wins[-1].end + 0.3, wins[-1].end + 0.9, voice(99))
    assert not tail.trusted, "the tail must be untrusted for this test to mean anything"
    wins.append(tail)
    eng = SpeakerEngine(expected_speakers=0, max_speakers=6)
    eng.add_windows(wins)
    eng.recluster()
    n = len(set(labels_of(eng)))
    assert n == 2, f"short tail created a phantom: {n} speakers"
    print("  0.6s garbage tail: no phantom speaker created  OK")


def test_label_stability_across_passes():
    """Speaker 1 must stay the same human when new audio arrives."""
    a, b = voice(10), voice(11)
    eng = SpeakerEngine(expected_speakers=2)
    w1, _ = build([(a, 6), (b, 6)])
    eng.add_windows(w1)
    eng.recluster()
    first = list(labels_of(eng))

    t = eng.windows[-1].end + 0.5
    more = [
        Window(t + i * 0.75, t + i * 0.75 + WIN_SEC, utterance(a)) for i in range(6)
    ]
    eng.add_windows(more)
    eng.recluster()

    after = labels_of(eng)
    kept = sum(1 for i, lab in enumerate(first) if after[i] == lab) / len(first)
    assert kept > 0.95, f"only {kept:.2f} of labels survived the new pass"
    print(f"  label stability after new audio: {kept:.0%} unchanged  OK")


def test_timeline_and_overlap_lookup():
    a, b = voice(12), voice(13)
    wins, _ = build([(a, 8), (b, 8)])
    eng = SpeakerEngine(expected_speakers=2)
    eng.add_windows(wins)
    eng.recluster()
    tl = eng.timeline()
    assert len(tl) >= 2, tl
    # A span inside the first turn must resolve to the first turn's speaker.
    early = eng.label_for(1.0, 3.0)
    late = eng.label_for(tl[-1][0] + 0.3, tl[-1][1] - 0.3)
    assert early is not None and late is not None
    assert early != late, "two distinct turns collapsed to one speaker"
    print(f"  timeline: {len(tl)} run(s); overlap lookup resolves both turns  OK")


if __name__ == "__main__":
    print("SpeakerEngine synthetic tests")
    for fn in [
        test_two_speakers_alternating,
        test_monologue_with_k_known_is_not_split,
        test_three_speakers_auto_k,
        test_short_tail_cannot_create_a_speaker,
        test_label_stability_across_passes,
        test_timeline_and_overlap_lookup,
    ]:
        fn()
    print("all passed")
