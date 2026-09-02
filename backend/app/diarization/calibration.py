"""
Session-adaptive calibration for speaker embedding cosine distances.

The speaker engine uses a recording-specific same-speaker distance band instead
of a fixed global threshold.  This module intentionally has no dependency on
the clustering engine so it can be tested in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Conservative bounds for cosine-distance thresholds.  The fallback value used
# by SpeakerEngine (0.50) sits safely inside this range.
MIN_SAME_MAX = 0.22
MAX_SAME_MAX = 0.65
MIN_WINDOWS = 8

# Estimate the centroid-separation threshold a little above the local
# same-speaker neighbourhood spread.  This keeps a monologue's artificial
# bipartition below the veto while allowing genuinely distinct, acoustically
# similar speakers to separate.
DEFAULT_MARGIN = 0.05

# Running calibration is deliberately slow-moving so one noisy live pass cannot
# abruptly change K.
BLEND_ALPHA = 0.25


@dataclass(frozen=True)
class Band:
    """Distance thresholds derived from one recording."""

    same_max: float
    n_windows: int
    raw_estimate: float
    clamped: bool = False

    @property
    def measured(self) -> bool:
        return self.n_windows >= MIN_WINDOWS

    @property
    def ambiguous_margin(self) -> float:
        # Preserve the v12 fallback exactly: same_max=0.50 -> 0.05.
        return float(np.clip(self.same_max * 0.10, 0.03, 0.10))

    @property
    def core_max(self) -> float:
        return self.same_max

    @property
    def outlier_max(self) -> float:
        # Preserve the v12 fallback exactly: 0.50 -> 0.75.
        return float(np.clip(self.same_max * 1.50, self.same_max, 0.95))

    @property
    def identity_match_max(self) -> float:
        # Preserve the v12 fallback exactly: 0.50 -> 0.55.
        return float(np.clip(self.same_max + 0.05, self.same_max, 0.95))

    @property
    def new_identity_min(self) -> float:
        # New identities need stronger evidence than ordinary assignment.
        return self.identity_match_max

    def as_log(self) -> str:
        state = "measured" if self.measured else "fallback"
        clamp = ",clamped" if self.clamped else ""
        return (
            f"same<={self.same_max:.3f} "
            f"(raw={self.raw_estimate:.3f}, n={self.n_windows}, {state}{clamp})"
        )


def knn_spread(dist_square: np.ndarray, neighbours: int = 5) -> np.ndarray:
    """Mean cosine distance to each row's nearest neighbours.

    `dist_square` must be an NxN square distance matrix with a zero diagonal.
    """
    d = np.asarray(dist_square, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("dist_square must be a square distance matrix")

    n = d.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    k = min(max(1, int(neighbours)), max(1, n - 1))
    if n == 1:
        return np.zeros(1, dtype=np.float64)

    work = d.copy()
    np.fill_diagonal(work, np.inf)
    nearest = np.partition(work, kth=k - 1, axis=1)[:, :k]
    return np.mean(nearest, axis=1)


def estimate_band(
    dist_square: np.ndarray,
    *,
    fallback: float = 0.50,
    neighbours: int = 5,
    margin: float = DEFAULT_MARGIN,
    min_same_max: float = MIN_SAME_MAX,
    max_same_max: float = MAX_SAME_MAX,
) -> Band:
    """Estimate the same-speaker separation band from local neighbourhoods.

    The local k-NN spread is much less sensitive to speaker count than the full
    pairwise-distance distribution because each window's nearest neighbours are
    normally other windows from the same voice.
    """
    d = np.asarray(dist_square, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("dist_square must be a square distance matrix")

    n = d.shape[0]
    if n < MIN_WINDOWS:
        return Band(
            same_max=float(fallback),
            n_windows=n,
            raw_estimate=float(fallback),
            clamped=False,
        )

    spread = knn_spread(d, neighbours)
    finite = spread[np.isfinite(spread)]
    if finite.size < MIN_WINDOWS:
        return Band(
            same_max=float(fallback),
            n_windows=n,
            raw_estimate=float(fallback),
            clamped=False,
        )

    # Use the 70th percentile rather than the maximum so occasional boundary or
    # noise windows do not inflate the threshold.  The additive margin converts
    # window-level local spread into a conservative centroid-separation veto.
    raw = float(np.percentile(finite, 70.0) + float(margin))
    same = float(np.clip(raw, min_same_max, max_same_max))

    return Band(
        same_max=same,
        n_windows=n,
        raw_estimate=raw,
        clamped=abs(same - raw) > 1e-12,
    )


def blend(previous: Optional[Band], fresh: Band, alpha: float = BLEND_ALPHA) -> Band:
    """Smooth a fresh calibration estimate into the running session band."""
    if previous is None or not previous.measured:
        return fresh
    if not fresh.measured:
        return previous

    a = float(np.clip(alpha, 0.0, 1.0))
    same = (1.0 - a) * previous.same_max + a * fresh.same_max
    raw = (1.0 - a) * previous.raw_estimate + a * fresh.raw_estimate

    return Band(
        same_max=float(same),
        n_windows=fresh.n_windows,
        raw_estimate=float(raw),
        clamped=previous.clamped or fresh.clamped,
    )
