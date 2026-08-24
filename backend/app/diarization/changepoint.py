"""
changepoint.py — find speaker turn boundaries from the window embeddings that
have already been computed.

THE DEFECT THIS ADDRESSES
=========================

`slice_windows` cuts each VAD speech region into WIN_SEC windows at HOP_SEC,
blind to who is talking. When two people take turns without a gap long enough
for the VAD to split them — which is most of a real conversation — the windows
spanning the changeover contain BOTH voices. Their embeddings land between the
two speakers' centroids.

With long turns that is a nuisance: a handful of mixed windows against many
clean ones. With short turns it is fatal. At WIN_SEC 2.0 and three-second
turns, only about half the windows sit inside a single turn; the rest are
mixtures, and the embedding cloud stops being two clusters with a gap and
becomes a continuum from one speaker to the other. Average linkage then finds
no split worth making, the separation veto rejects every K >= 2, and the whole
conversation is reported as one person.

Measured on the synthetic geometry in `bench_turns.py`, with continuous speech
and no VAD gaps at the turn changes:

    turn length   K found (all windows)   K found (mixtures excluded)   truth
    6 s                    2.00                       2.00                2
    3 s                    1.00                       2.00                2
    2 s                    1.00                       1.12                2

That is the same collapse the calibration work fixed for a different reason,
arriving by a different route — and no threshold fixes it, because with a
continuum there is genuinely no gap to find. The mixtures have to be removed
from the evidence instead.

HOW BOUNDARIES ARE FOUND WITHOUT KNOWING THE SPEAKERS
-----------------------------------------------------
This is the classical distance-curve segmentation, and the useful observation
is that it costs NOTHING here. Windows overlap (HOP_SEC < WIN_SEC), so a window
and the one `stride` positions later are DISJOINT in time:

    window i        [t,        t + WIN]
    window i+stride            [t + WIN,  t + 2*WIN]

Their cosine distance is large when a speaker change falls between them and
small when both sit inside one turn. Both embeddings already exist — they were
computed for clustering — so the curve is a vector of dot products over data
already in memory. No second model pass, no extra audio, no GPU.

Peaks in that curve, above a threshold taken from the curve's OWN distribution,
are the turn boundaries.

WHAT THE THRESHOLD ASSUMES
--------------------------
That a conversation spends more time inside turns than changing between them,
so the MEDIAN of the curve is the same-speaker level. The threshold sits a
fraction of the way from the median to the 90th percentile. A recording where
speakers alternate every window would break that assumption — and would also be
beyond what 2-second windows can represent at all.

THE RESOLUTION LIMIT, STATED PLAINLY
------------------------------------
Because both sides of the comparison are WIN_SEC long, a turn shorter than
about WIN_SEC cannot appear as a clean block on either side, and the curve
never resolves it. Measured recall against turn length (precision stays at or
near 100% throughout):

    6 s turns    100%
    3 s turns     99%
    2 s turns     66%
    1.5 s turns   detected only when the neighbouring turns are long

Recovering short turns needs a FINER detection grid — 1.0 s windows at 0.25 s
hop measured 98% recall on 2-second turns — but that grid costs about 8x
realtime in embedding against the 2.7x the current windowing costs, so it
belongs in the final offline pass rather than the live loop. It is not
implemented here; `bench_turns.py --fine` reproduces the measurement.

Below MIN_EMBED_SEC (0.90 s) a turn cannot be embedded at all, by anyone. That
is a floor of the embedding backend, not of this module.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

log = logging.getLogger("voxlive.speaker.cp")

# Fraction of the way from the median of the distance curve to its 90th
# percentile. Lower finds more boundaries and more false ones; higher is
# conservative. 0.45 was chosen on the synthetic geometry as the point where
# precision stays at 100% while recall is highest.
PEAK_HEIGHT_FRAC = 0.45

# ABSOLUTE floor on a peak, as a multiple of the calibrated same-speaker band.
#
# The relative threshold above cannot stand alone, and the failure is not
# subtle. In a MONOLOGUE the curve has no real peaks — it is same-speaker
# distance throughout — but it still has a median, a 90th percentile and local
# maxima, so a purely relative rule finds "boundaries" in one person's natural
# variation. Measured on 60 windows of a single voice, the relative rule alone
# reported 11 turn changes. Every one of them would have become a paragraph
# break in the transcript.
#
# A boundary is only a boundary if the two sides differ by MORE than two
# windows of the same person typically differ, and calibration.py already
# measures exactly that. Measured eligibility at each multiple:
#
#     floor         monologue     6 s turns     3 s turns
#     1.00 * band      29 pts        49 pts        65 pts
#     1.15 * band       1 pt         27 pts        58 pts
#     1.30 * band       0 pts        25 pts        56 pts    <- chosen
#     1.50 * band       0 pts        18 pts        35 pts
#
# 1.30 is the first multiple that silences the monologue completely while
# leaving the conversation peaks largely intact.
PEAK_FLOOR_FRAC = 1.30

# Two boundaries closer together than this collapse to the stronger one. A turn
# shorter than this is below the resolution of the window geometry anyway, so
# reporting two boundaries around it only invents a phantom turn.
MIN_SEPARATION_SEC = 1.00

# Windows either side of the comparison must be genuinely adjacent in time. A
# gap larger than this means a VAD silence sits between them — the two windows
# describe different moments and their distance says nothing about a turn.
MAX_GAP_SEC = 0.75

# Below this many usable pairs the curve has no distribution to take a
# threshold from, and any "peak" is one sample.
MIN_PAIRS = 8


def detect_change_points(
    spans: Sequence[tuple[float, float]],
    embeddings: np.ndarray,
    stride: Optional[int] = None,
    height_frac: float = PEAK_HEIGHT_FRAC,
    min_separation: float = MIN_SEPARATION_SEC,
    min_height: float = 0.0,
) -> list[float]:
    """Speaker change times, in the same clock as `spans`.

    `spans` and `embeddings` must be time-sorted and the same length. `stride`
    is how many positions apart two windows must be to be disjoint; leave it as
    None and it is derived from the spans themselves, so it stays correct if
    the window geometry is ever retuned.

    `min_height` is the ABSOLUTE floor a peak must clear. Pass
    `PEAK_FLOOR_FRAC * band.same_max` from the calibrated band — without it a
    monologue produces spurious boundaries, because a relative threshold always
    finds something in any curve. It defaults to 0 only so the function stays
    usable in isolation; the engine always passes a real value.

    Returns an ascending list of times. An empty list means "no boundary found",
    which is the right answer for a monologue and also what you get from a
    recording too short to have a distribution.
    """
    n = len(spans)
    if n < MIN_PAIRS or len(embeddings) != n:
        return []

    starts = np.array([s for s, _ in spans], dtype=np.float64)
    ends = np.array([e for _, e in spans], dtype=np.float64)

    if stride is None:
        # Smallest k such that window i+k starts at or after window i ends.
        win = float(np.median(ends - starts))
        hop = float(np.median(np.diff(starts))) if n > 1 else win
        stride = max(1, int(np.ceil(win / hop))) if hop > 1e-6 else 1

    if n <= stride:
        return []

    left, right = embeddings[:-stride], embeddings[stride:]
    dist = 1.0 - np.sum(left * right, axis=1)

    # Only pairs that are actually adjacent in time. Everything else is
    # comparing across a silence, which is not a turn boundary.
    gap = starts[stride:] - ends[:-stride]
    usable = (gap >= -1e-6) & (gap <= MAX_GAP_SEC)
    if int(usable.sum()) < MIN_PAIRS:
        return []

    # The boundary, if there is one, lies in the gap between the two windows.
    mid = (ends[:-stride] + starts[stride:]) / 2.0

    d = dist[usable]
    t = mid[usable]
    median = float(np.median(d))
    high = float(np.percentile(d, 90.0))
    if high - median < 1e-6:
        return []
    threshold = max(median + height_frac * (high - median), float(min_height))

    peaks: list[tuple[float, float]] = []  # (time, height)
    for i in range(1, len(d) - 1):
        if d[i] < threshold or d[i] < d[i - 1] or d[i] < d[i + 1]:
            continue
        if peaks and t[i] - peaks[-1][0] < min_separation:
            # Keep whichever of the two is the stronger evidence.
            if d[i] > peaks[-1][1]:
                peaks[-1] = (float(t[i]), float(d[i]))
            continue
        peaks.append((float(t[i]), float(d[i])))

    log.debug(
        "change points: %d found over %d pair(s), threshold=%.3f "
        "(median=%.3f, p90=%.3f, floor=%.3f)",
        len(peaks),
        len(d),
        threshold,
        median,
        high,
        min_height,
    )
    return [p for p, _ in peaks]


def mark_straddling(
    spans: Sequence[tuple[float, float]],
    change_points: Sequence[float],
    guard: float = 0.0,
) -> np.ndarray:
    """Boolean mask: True where a window CONTAINS a change point.

    These are the windows holding two voices. They are still labelled — the
    transcript must stay continuous — but they must not define a speaker, so
    the engine demotes them out of the trusted set.

    `guard` widens each change point, for when the boundary estimate itself is
    uncertain. It defaults to 0 because the estimate is already the midpoint of
    a gap, and widening it discards clean windows either side.
    """
    mask = np.zeros(len(spans), dtype=bool)
    if not change_points:
        return mask
    cps = np.asarray(sorted(change_points), dtype=np.float64)
    for i, (a, b) in enumerate(spans):
        lo, hi = a - guard, b + guard
        idx = np.searchsorted(cps, lo, side="left")
        mask[i] = idx < len(cps) and cps[idx] < hi
    return mask
