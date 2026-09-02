"""Stability fixes for the embedding diarization engine.

This wrapper keeps the v13 clustering implementation intact and changes only
three production failure modes observed in real sessions:

1. overlap-tail windows emitted by DiarizationService are de-duplicated;
2. the unexplained-audio watchdog consumes new evidence once instead of
   repeatedly re-counting the whole session after a re-derivation;
3. coherent, strongly separated short interjections can establish a minority
   speaker without weakening the normal 6 s discovery rule.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

from .speaker_engine import (
    ADAPT_CALIB_CAP,
    CENTROID_UPDATE_RATE,
    REDERIVE_MIN_SESSION_SEC,
    REDERIVE_UNEXPLAINED_SEC,
    SpeakerEngine,
    Window,
    l2norm,
)

log = logging.getLogger("voxlive.speaker")


class StableSpeakerEngine(SpeakerEngine):
    """SpeakerEngine with ingestion and minority-speaker stability fixes."""

    def __init__(
        self,
        *args,
        new_identity_min_sec: float = 6.0,
        new_identity_min_windows: int = 4,
        new_identity_short_sec: float = 3.2,
        new_identity_short_windows: int = 3,
        new_identity_strong_dist: float = 0.68,
        growth_check_sec: float = 20.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.new_identity_min_sec = max(0.0, float(new_identity_min_sec))
        self.new_identity_min_windows = max(1, int(new_identity_min_windows))
        self.new_identity_short_sec = max(0.0, float(new_identity_short_sec))
        self.new_identity_short_windows = max(1, int(new_identity_short_windows))
        self.new_identity_strong_dist = float(
            np.clip(float(new_identity_strong_dist), 0.0, 2.0)
        )
        # Retained as an explicit runtime setting instead of being swallowed by
        # **_ignored. It is reserved for periodic growth checks; the current
        # watchdog remains evidence-driven rather than timer-driven.
        self.growth_check_sec = max(0.0, float(growth_check_sec))

        self._window_keys: set[tuple[int, int]] = set()
        self._watchdog_seen: set[tuple[int, int]] = set()
        self._watchdog_unexplained_accum = 0.0
        self._watchdog_trigger_pending = False
        self._last_discovery_evidence_sec = 0.0

    @staticmethod
    def _window_key(w: Window) -> tuple[int, int]:
        # Millisecond quantisation is much finer than the 0.75 s hop while
        # tolerating harmless floating-point reconstruction differences.
        return (int(round(w.start * 1000.0)), int(round(w.end * 1000.0)))

    def add_windows(self, windows: list[Window]) -> None:
        """Add only genuinely new timeline windows.

        DiarizationService intentionally retains a WIN_SEC tail between passes.
        That tail is re-windowed on the next pass, so blindly appending it makes
        historical speech count multiple times and corrupts both clustering and
        the distance-curve turn detector.
        """
        if not windows:
            return
        fresh: list[Window] = []
        for w in windows:
            key = self._window_key(w)
            if key in self._window_keys:
                continue
            self._window_keys.add(key)
            fresh.append(w)
        if fresh:
            super().add_windows(fresh)

    def _needs_rederive(self, trusted: list[Window]) -> bool:
        """Trigger only from newly observed unexplained evidence.

        v13 re-counted every historical far window on every call. If a full
        derivation still returned the same K, the exact same 150+ seconds could
        immediately trigger another derivation, creating the re-clustering loop
        visible in production logs.
        """
        if self._centroid_mat is None or not trusted:
            return False
        if self._trusted_sec(trusted) < REDERIVE_MIN_SESSION_SEC:
            return False

        unseen = [w for w in trusted if self._window_key(w) not in self._watchdog_seen]
        if not unseen:
            self._unexplained_sec = self._watchdog_unexplained_accum
            return False

        X = np.stack([w.embedding for w in unseen])
        dur = np.array([w.duration for w in unseen], dtype=np.float64)
        best = cdist(X, self._centroid_mat, metric="cosine").min(axis=1)
        far = best > self._thr().new_identity_min
        self._watchdog_unexplained_accum += float(dur[far].sum())
        self._watchdog_seen.update(self._window_key(w) for w in unseen)
        self._unexplained_sec = self._watchdog_unexplained_accum

        if self._watchdog_unexplained_accum >= REDERIVE_UNEXPLAINED_SEC:
            self._watchdog_trigger_pending = True
            return True
        return False

    def _derive_clusters(self, trusted: list[Window], final: bool = False) -> bool:
        ok = super()._derive_clusters(trusted, final=final)
        if ok and self._watchdog_trigger_pending:
            # A successful full derivation has already consumed all evidence up
            # to this point. Re-arm only when genuinely new windows arrive.
            self._watchdog_seen = {self._window_key(w) for w in trusted}
            self._watchdog_unexplained_accum = 0.0
            self._unexplained_sec = 0.0
            self._watchdog_trigger_pending = False
        return ok

    def _discover(
        self, X: np.ndarray, dur: np.ndarray, cents: np.ndarray
    ) -> Optional[np.ndarray]:
        """Discover either a normal-duration or strongly separated short voice."""
        self._last_discovery_evidence_sec = 0.0
        if self.speaker_mode == "fixed" or len(cents) >= self.cap:
            return None

        far_thr = self._thr().new_identity_min
        best = cdist(X, cents, metric="cosine").min(axis=1)

        normal = best > far_thr
        short = best >= max(far_thr, self.new_identity_strong_dist)

        candidates: list[tuple[np.ndarray, int, float, float]] = []
        normal_sec = float(dur[normal].sum())
        if int(normal.sum()) >= self.new_identity_min_windows and normal_sec >= self.new_identity_min_sec:
            candidates.append((normal, self.new_identity_min_windows, self.new_identity_min_sec, far_thr))

        short_sec = float(dur[short].sum())
        if int(short.sum()) >= self.new_identity_short_windows and short_sec >= self.new_identity_short_sec:
            candidates.append(
                (
                    short,
                    self.new_identity_short_windows,
                    self.new_identity_short_sec,
                    max(far_thr, self.new_identity_strong_dist),
                )
            )

        # Prefer the stricter short path when both are available; it uses a
        # stronger centroid-separation guard and therefore cannot make normal
        # discovery more permissive.
        candidates.sort(key=lambda item: item[3], reverse=True)

        for mask, min_windows, min_sec, final_sep in candidates:
            Xf, wf = X[mask], dur[mask]
            cand = l2norm((Xf * (wf / wf.sum())[:, None]).sum(axis=0))

            # The candidate must be a coherent voice, not merely a collection
            # of unrelated sounds that all differ from known speakers.
            agree = cdist(Xf, cand[None, :], metric="cosine").ravel() <= far_thr
            if int(agree.sum()) < min_windows:
                continue
            evidence_sec = float(wf[agree].sum())
            if evidence_sec < min_sec:
                continue

            Xa, wa = Xf[agree], wf[agree]
            cand = l2norm((Xa * (wa / wa.sum())[:, None]).sum(axis=0))
            if float(cdist(cand[None, :], cents, metric="cosine").min()) < final_sep:
                continue

            self._last_discovery_evidence_sec = evidence_sec
            return cand
        return None

    def _adapt(self, trusted: list[Window]) -> None:
        """Base adaptation with accurate logging for short discoveries."""
        if self._centroid_mat is None:
            return

        cents = self._centroid_mat.copy()
        X = np.stack([w.embedding for w in trusted])
        dur = np.array([w.duration for w in trusted], dtype=np.float64)
        starts = np.array([w.start for w in trusted], dtype=np.float64)

        if self.calibrate and len(X) >= 2:
            idx = (
                np.linspace(0, len(X) - 1, ADAPT_CALIB_CAP).round().astype(int)
                if len(X) > ADAPT_CALIB_CAP
                else np.arange(len(X))
            )
            idx = np.unique(idx)
            self._recalibrate(squareform(pdist(X[idx], metric="cosine")))

        band = self._thr()
        lab, margin, best = self._nearest(X, cents)
        clean = (margin >= band.ambiguous_margin) & (best <= band.outlier_max)
        if clean.sum() >= len(cents):
            target = self._centroids(
                X[clean], dur[clean], lab[clean], len(cents), fallback=cents
            )
            a = CENTROID_UPDATE_RATE
            cents = np.stack(
                [l2norm((1.0 - a) * cents[i] + a * target[i]) for i in range(len(cents))]
            )

        fresh = self._discover(X, dur, cents)
        if fresh is not None:
            cents = np.vstack([cents, fresh])
            log.info(
                "new speaker identity discovered from %.1fs of coherent audio "
                "unlike any known voice (band %s) -> %d speaker(s)",
                self._last_discovery_evidence_sec,
                band.as_log(),
                len(cents),
            )

        lab, _, _ = self._nearest(X, cents)
        spans = [
            (
                float(starts[lab == c].min()) if np.any(lab == c) else 0.0,
                float(dur[lab == c].sum()),
            )
            for c in range(len(cents))
        ]
        self._register(cents, spans)
