"""Head-to-head: v7's greedy online labelling vs v10's session-global engine.

Scenario is the one VoxLive actually fails on: a fast two-person exchange with
many short turns, embeddings noisy the way real 1-2s windows are.

The v7 baseline below is a faithful reimplementation of
SpeakerClusterer._assign_provisional: nearest running centroid, create a new
speaker when distance > match + margin and the turn is long enough, update the
centroid with a duration-weighted running mean.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.spatial.distance import cdist

from app.speaker_engine import SpeakerEngine, Window, l2norm
from test_speaker_engine import voice, utterance


def accuracy(pred, truth):
    """Standard permutation-invariant speaker accuracy.

    Hungarian-match predicted clusters to true speakers, then count agreement.
    NOT 'purity': purity scores 100% when every window collapses into a single
    cluster, which is precisely one of the failure modes under test.
    """
    from scipy.optimize import linear_sum_assignment
    pi, ti = sorted(set(pred)), sorted(set(truth))
    M = np.zeros((len(pi), len(ti)))
    for a, b in zip(pred, truth):
        M[pi.index(a), ti.index(b)] += 1
    r, c = linear_sum_assignment(-M)
    return M[r, c].sum() / len(truth)

RNG = np.random.default_rng(4242)


def v7_greedy(embs, durs, expected=0, max_spk=6, match=0.55, margin=0.15,
              min_new=1.5, reliable=1.0):
    """v7 tier-1. Returns a label per turn."""
    cents, weights, labels, nxt = {}, {}, [], 0
    cap = expected if expected > 0 else max_spk
    for e, d in zip(embs, durs):
        if not cents:
            cents[nxt] = e.copy(); weights[nxt] = d if d >= reliable else 0.0
            labels.append(nxt); nxt += 1
            continue
        ids = sorted(cents)
        dist = cdist(e[None, :], np.stack([cents[i] for i in ids]), "cosine")[0]
        bi = int(np.argmin(dist)); bid, bd = ids[bi], float(dist[bi])
        if len(cents) < cap and d >= min_new and bd > match + margin:
            cents[nxt] = e.copy(); weights[nxt] = d
            labels.append(nxt); nxt += 1
            continue
        if d >= reliable:
            wo, wn = weights.get(bid, 0.0), d
            tot = wo + wn
            if tot > 0:
                cents[bid] = l2norm((cents[bid] * wo + e * wn) / tot)
                weights[bid] = tot
        labels.append(bid)
    return labels


def similar_voices(seed=21, closeness=0.40):
    """Two people of the same gender, same room, same microphone.

    This is the case VoxLive actually runs in and the case random 256-d
    vectors do NOT model: cross-speaker distance drops to ~0.55-0.65, close
    enough that a single short turn genuinely is ambiguous.
    """
    a = voice(seed)
    r = np.random.default_rng(seed + 1)
    b = l2norm(closeness * a + (1 - closeness) * 2.2 * l2norm(r.normal(size=len(a))))
    return a, b


def scenario(n_turns=40, seed=11):
    """Fast two-way exchange. Turn lengths 0.7s-3.0s, skewed short — a real
    conversation, not read speech."""
    rng = np.random.default_rng(seed)
    a, b = similar_voices()
    bases = [a, b]
    embs, durs, truth, spans, t = [], [], [], [], 0.0
    for i in range(n_turns):
        who = i % 2 if rng.random() > 0.25 else int(rng.integers(0, 2))
        dur = float(np.clip(rng.gamma(2.0, 0.6), 0.7, 3.0))
        # A short window yields a noisier embedding — the duration dependence
        # that made a single threshold impossible in v7.
        jitter = 0.60 + 1.45 * max(0.0, (1.8 - dur) / 1.8)
        e = utterance(bases[who], jitter)
        # The session opens with a throat-clear / mic pop. In v7 this became
        # Speaker 1's centroid and every later comparison was made against it.
        if i == 0:
            e = l2norm(0.45 * e + 0.55 * l2norm(rng.normal(size=len(e))))
        embs.append(e)
        durs.append(dur)
        truth.append(who)
        spans.append((t, t + dur))
        t += dur + rng.uniform(0.15, 0.6)
    return embs, durs, truth, spans


def run():
    embs, durs, truth, spans = scenario()

    v7 = v7_greedy(embs, durs, expected=2)
    v7_p = accuracy(v7, truth)
    v7_n = len(set(v7))

    v7_auto = v7_greedy(embs, durs, expected=0, max_spk=6)
    v7a_p = accuracy(v7_auto, truth)
    v7a_n = len(set(v7_auto))

    # v10: each turn contributes its window(s) at real duration.
    eng = SpeakerEngine(expected_speakers=2)
    eng.add_windows([Window(s, e, emb) for (s, e), emb in zip(spans, embs)])
    eng.recluster()
    new_p = accuracy(eng._labels, truth)
    new_n = len(set(eng._labels))

    eng2 = SpeakerEngine(expected_speakers=0, max_speakers=6)
    eng2.add_windows([Window(s, e, emb) for (s, e), emb in zip(spans, embs)])
    eng2.recluster()
    new2_p = accuracy(eng2._labels, truth)
    new2_n = len(set(eng2._labels))

    print(f"  40-turn fast exchange, 2 real speakers\n")
    print(f"  {'':28} {'speakers found':>15} {'accuracy':>10}")
    print(f"  {'-'*56}")
    print(f"  {'v7 greedy  (K=2 known)':28} {v7_n:>15} {v7_p:>9.1%}")
    print(f"  {'v7 greedy  (K auto)':28} {v7a_n:>15} {v7a_p:>9.1%}")
    print(f"  {'v10 engine (K=2 known)':28} {new_n:>15} {new_p:>9.1%}")
    print(f"  {'v10 engine (K auto)':28} {new2_n:>15} {new2_p:>9.1%}")
    print()
    # ---- the decisive part -------------------------------------------------
    # v7's answer to bad diarization was always "tune DIARIZATION_THRESHOLD".
    # Sweep it. There is no setting that works, because the quantity being
    # thresholded (embedding distance) depends on turn DURATION as strongly as
    # it depends on speaker identity.
    print("  v7 threshold sweep (K=2 known):")
    print(f"  {'threshold':>10} {'speakers':>10} {'accuracy':>10}")
    best = 0.0
    for th in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        lab = v7_greedy(embs, durs, expected=2, match=th)
        acc = accuracy(lab, truth)
        best = max(best, acc)
        print(f"  {th:>10.2f} {len(set(lab)):>10} {acc:>9.1%}")
    print(f"\n  best v7 across ALL thresholds: {best:.1%}")
    print(f"  v10 (no threshold tuned):      {new_p:.1%}\n")

    assert new_p >= v7_p, "regression vs v7 with K known"
    assert new2_n == 2, f"auto-K found {new2_n} speakers, expected 2"
    assert new_p >= best, "v10 must beat v7's best-case threshold"
    return v7_p, v7a_n, new_p, new2_n


if __name__ == "__main__":
    print("v7 vs v10 head-to-head\n")
    run()
    print("  OK")
