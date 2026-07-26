"""Synthetic tests for SpeakerEngine.

These use fabricated embeddings with realistic WeSpeaker geometry:
same-speaker cosine distance ~0.15-0.40, cross-speaker ~0.70-1.05. The point
is to pin down the behaviours that v7 got wrong, so they cannot regress.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from app.speaker_engine import SpeakerEngine, Window, l2norm

RNG = np.random.default_rng(7)
DIM = 256


# A shared component every human voice has (recording channel, language,
# the model's own bias). Without it, random 256-d vectors sit at cosine
# distance ~1.0 and the test is far easier than reality.
_COMMON = l2norm(np.random.default_rng(0).normal(size=DIM))


def voice(seed: int) -> np.ndarray:
    """A speaker identity. Cross-speaker cosine distance lands ~0.70-0.80,
    matching WeSpeaker on real 1.5s windows."""
    r = np.random.default_rng(seed)
    return l2norm(0.45 * _COMMON + l2norm(r.normal(size=DIM)))


def utterance(base: np.ndarray, jitter: float = 0.75) -> np.ndarray:
    """One window of one person. jitter=0.75 puts same-speaker cosine distance
    around 0.18-0.25 — the real within-speaker spread at 1.5s."""
    return l2norm(base + jitter * l2norm(RNG.normal(size=DIM)))


def build(turns, win_sec=1.5, hop=0.75):
    """turns: [(speaker_base, n_windows)] laid out consecutively in time."""
    wins, truth, t = [], [], 0.0
    for base, n in turns:
        for _ in range(n):
            wins.append(Window(t, t + win_sec, utterance(base)))
            truth.append(id(base))
            t += hop
        t += 0.5  # inter-turn pause
    return wins, truth


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
    labels = eng._labels
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
    wins, truth = build([(a, 24)])
    eng = SpeakerEngine(expected_speakers=2)
    eng.add_windows(wins)
    eng.recluster()
    n = len(set(eng._labels))
    assert n == 1, f"monologue split into {n} speakers"
    print("  monologue with K=2 known: stayed 1 speaker  OK")


def test_three_speakers_auto_k():
    a, b, c = voice(4), voice(5), voice(6)
    wins, truth = build([(a, 6), (b, 6), (c, 6), (a, 5), (c, 5), (b, 4)])
    eng = SpeakerEngine(expected_speakers=0, max_speakers=6)
    eng.add_windows(wins)
    eng.recluster()
    n = len(set(eng._labels))
    p = purity(eng._labels, truth)
    assert n == 3, f"expected 3, got {n}"
    assert p > 0.95, f"purity {p:.3f}"
    print(f"  three speakers, K unknown: found 3, purity {p:.3f}  OK")


def test_short_tail_cannot_create_a_speaker():
    """Untrusted (<1.4s) windows must consume the clustering, never shape it."""
    a, b = voice(7), voice(8)
    wins, _ = build([(a, 8), (b, 8)])
    # A 0.6s garbage tail — the kind of thing that spawned 'Speaker 9' in v7.
    wins.append(Window(wins[-1].end + 0.3, wins[-1].end + 0.9, voice(99)))
    eng = SpeakerEngine(expected_speakers=0, max_speakers=6)
    eng.add_windows(wins)
    eng.recluster()
    n = len(set(eng._labels))
    assert n == 2, f"short tail created a phantom: {n} speakers"
    print("  0.6s garbage tail: no phantom speaker created  OK")


def test_label_stability_across_passes():
    """Speaker 1 must stay the same human when new audio arrives."""
    a, b = voice(10), voice(11)
    eng = SpeakerEngine(expected_speakers=2)
    w1, _ = build([(a, 6), (b, 6)])
    eng.add_windows(w1)
    eng.recluster()
    first = dict(zip(range(len(eng._labels)), eng._labels))

    t = eng.windows[-1].end + 0.5
    more = [Window(t + i * 0.75, t + i * 0.75 + 1.5, utterance(a)) for i in range(6)]
    eng.add_windows(more)
    eng.recluster()
    kept = sum(1 for i in first if eng._labels[i] == first[i]) / len(first)
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
