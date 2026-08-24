"""
speaker_engine.py — session-global speaker identity (v11).

WHAT WAS WRONG IN v10
=====================

1. "Speaker 17" FOR A FOUR-PERSON PODCAST
   v10's `_stabilize` mapped this pass's clusters onto the previous pass's
   labels with a Hungarian assignment, and any cluster that failed to match got

       nxt = max(max(old_ids), self._max_sid) + 1

   `_max_sid` only ever went up. So every time the clustering wobbled — a
   cluster appearing for one pass and vanishing the next, which happens
   constantly while audio streams in — a brand new speaker NUMBER was burned
   forever. The log shows exactly that ladder: 1..5, then 12, then 16, 17, 18,
   then 23, 24. Nothing was wrong with the audio; the ID allocator leaked.

   Worse, matching was done on *label overlap between passes*, so identity
   depended on the previous pass's mistakes rather than on the voices.

   v11 keeps a REGISTRY of identities keyed by CENTROID, matches this pass's
   clusters to it by voice distance (not by yesterday's labels), and renders
   DENSE display numbers 1..N ordered by first appearance. A number can never
   exceed the number of people in the room, because it *is* an index into the
   registry.

2. FOUR PEOPLE CLUSTERED AS EIGHT OR NINE
   Three separate causes, all fixed here:

   a) A FIXED DISTANCE CUT (`SPLIT_DISTANCE = 0.60`). One person across seven
      minutes of animated speech spreads well past 0.60 in cosine distance —
      vocal effort, loudness and proximity move an embedding as much as
      identity does. A fixed cut therefore shatters real speakers. v11 chooses
      K by weighted silhouette over K = 2..cap and vetoes any K whose two
      closest centroids sit inside `same_speaker_max` (that veto is what still
      protects the one-speaker case).

   b) BOUNDARY WINDOWS. With WIN_SEC 1.5 and HOP 0.75, every turn change
      produces windows containing two voices. Their embeddings land between
      two centroids, and a handful of them is enough to seed a phantom cluster
      that looks perfectly self-consistent. v11 measures each window's MARGIN
      (distance to nearest centroid vs. second nearest) and excludes ambiguous
      windows from centroid estimation, keeping centroids built from clean
      single-voice audio only.

   c) AN ABSOLUTE PRUNING FLOOR (`min_cluster_sec = 3.0`). Three seconds is
      nothing in a 400-second session, so junk clusters always survived. The
      floor is now RELATIVE: a real participant owns a couple of percent of
      the speech at minimum, a phantom never does.

   Window geometry also moved from 1.5 s to 2.0 s, with trust at 1.8 s.
   Same-speaker distance tightens sharply with window length, and intra-speaker
   spread is precisely what was causing the splits.

3. FLICKER PARAGRAPHS
   A two-second phantom run produced a one-line paragraph with timestamps that
   overlapped its neighbours. Timeline votes are now weighted by window
   confidence, and any run shorter than MIN_RUN_SEC is absorbed into whichever
   neighbour holds it more strongly.

4. COST GROWING WITHOUT BOUND
   v10 ran a full pdist + linkage over every window in the session on every
   pass — 1554 windows by the end of a 7-minute session, roughly a thousand
   times. v11 subsamples uniformly over TIME for the linkage step (coverage is
   preserved, so a speaker holding 3 % of the speech keeps 3 % of the sample),
   and only re-derives clusters once enough new trusted audio has arrived.
   In between it does the cheap part — assign to known centroids, rebuild the
   timeline — which is what actually moves labels on screen.

PUBLIC SURFACE is unchanged: add_windows / recluster / timeline / label_for /
speaker_count / n_windows / reset, plus WIN_SEC, Window, speech_regions and
slice_windows. `recluster()` gained an optional `force` argument.

WHAT v12 CHANGES (supervisor review, P0)
=======================================

v11 could show Speaker 2 and then take it away again. The review traced the
exact path, and it is not a threshold that needs nudging:

    2 speakers detected
           -> 4 s of new trusted audio arrives
           -> the WHOLE session is reclustered from scratch
           -> _choose_k vetoes K=2 because dmin < SAME_SPEAKER_MAX
           -> best_k falls back to its initialiser, 1
           -> every window in the session is reassigned to one centroid
           -> Speaker 2 disappears and the history is relabelled

Two independent things were wrong.

1. EXPECTED_SPEAKERS WAS ONLY EVER A CEILING.
   "2" meant "at most two", so K=1 stayed reachable for a recording the user
   had already told us has two people in it. v12 splits that into an explicit
   MODE:

       speaker_mode="auto"   -> estimate K (expected_speakers is a ceiling,
                                exactly as before — this is still the default,
                                and every v11 behaviour is preserved)
       speaker_mode="fixed"  -> K = expected_speakers, full stop. The
                                separation veto and the relative-size prune
                                are both bypassed, because both of them exist
                                to REMOVE clusters and in fixed mode there is
                                nothing to remove.

   The review's caveat is real and is not papered over here: fixed K on a
   recording where only one person actually speaks WILL split that person in
   two. That is a UI contract ("assume exactly N"), not a bug, and the
   frontend now says so.

2. LIVE RECLUSTERING WAS FREE TO REWRITE HISTORY.
   Even in auto mode, re-deciding K every four seconds means an established
   identity can be deleted by a pass with a marginally better silhouette.
   v12 introduces ESTABLISHMENT:

       < ESTABLISH_SEC of trusted audio   provisional; K may still move
       >= ESTABLISH_SEC                   identities are ESTABLISHED

   Once established, a live pass no longer re-derives K. It does the
   conservative thing instead: nudge each centroid toward its new membership
   with an EMA (CENTROID_UPDATE_RATE), assign incoming windows to the nearest
   existing identity, and — in auto mode only — admit a genuinely new voice
   when NEW_IDENTITY_MIN_SEC of audio sits further than same_speaker_max from
   EVERY known centroid AND agrees with itself.

   Whole-session re-derivation still happens exactly once more, on
   `recluster(force=True)`, which the service calls from `finalize()` when the
   user actually stops. So:

       LIVE  -> stable
       FINAL -> refined

   rather than v11's "LIVE -> constantly rewritten".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist, squareform

from .calibration import Band, blend, estimate_band, knn_spread
from .changepoint import (
    PEAK_FLOOR_FRAC,
    detect_change_points,
    mark_straddling,
)

log = logging.getLogger("voxlive.speaker")

# ---- window geometry -------------------------------------------------------
# 2.0 s carries noticeably more identity than 1.5 s and noticeably less
# phonetic content. TRUST_SEC is the floor for "may define a speaker"; shorter
# windows are still labelled, they just don't get a vote in who exists.
WIN_SEC = 2.00
HOP_SEC = 0.75
TRUST_SEC = 1.80
FRAME_SEC = 0.25

# ---- calibration -----------------------------------------------------------
# Two cluster centroids closer than this are the same person. This is the only
# distance threshold left, and it is applied between CENTROIDS (averages over
# many seconds), never between individual short windows — which is why it can
# be a constant at all.
SAME_SPEAKER_MAX = 0.50

# Gate for re-associating a cluster with an identity seen in earlier passes.
IDENTITY_MATCH_MAX = 0.55

# A window whose two best centroids are within this distance of each other is
# ambiguous — most likely it straddles a turn boundary. Labelled, but never
# allowed to shape a centroid.
AMBIG_MARGIN = 0.05

# A window further than this from every centroid resembles nobody present. It
# is still given the nearest label (so the transcript stays continuous) but it
# barely votes.
OUTLIER_MAX = 0.75

# Neighbours consulted by the core filter, and the distance a genuine voice
# window must achieve to them. If your five nearest neighbours in the whole
# session average further away than SAME_SPEAKER_MAX, nobody here sounds like
# you: you are a music sting, a cough, or a chair scrape, not a speaker.
CORE_NEIGHBOURS = 5

# Absolute ceiling, whatever .env says. Nine "speakers" for four people is not
# a configuration the system should be able to reach.
HARD_MAX_SPEAKERS = 8

# Windows fed to the linkage step. Uniform over time, so proportions hold.
CLUSTER_SAMPLE_CAP = 800

# Runs shorter than this are absorbed into a neighbour instead of becoming a
# paragraph of their own.
MIN_RUN_SEC = 0.80

# New trusted audio required before clusters are re-derived from scratch.
RECLUSTER_AFTER_SEC = 4.0

# ---- v12: identity establishment -------------------------------------------
# Trusted speech required before the speaker structure stops being provisional.
# Below this the engine is still guessing from a couple of windows and must be
# allowed to change its mind; above it, the structure is evidence-backed and a
# live pass may no longer overturn it.
#
# In FIXED mode this is also the gate on forcing K. Forcing K=2 against three
# seconds of one person talking would manufacture two identities out of one
# voice and then lock them in for the session — the exact failure the review
# warns about in its caveat. Waiting means the split is made against real
# evidence.
ESTABLISH_SEC = 8.0

# How far a single live pass may move an established centroid. An identity is
# an average over many seconds; one four-second pass is allowed to nudge it,
# never to redefine it.
CENTROID_UPDATE_RATE = 0.15

# AUTO mode only. Trusted audio that must sit further than `same_speaker_max`
# from EVERY known centroid — and agree with itself — before a new identity is
# admitted after establishment. Deliberately larger than RECLUSTER_AFTER_SEC:
# discovery should need more evidence than a routine pass carries.
NEW_IDENTITY_MIN_SEC = 6.0
NEW_IDENTITY_MIN_WINDOWS = 4

# ---- v13: the unexplained-audio watchdog -----------------------------------
# Trusted audio that fits NO established centroid before a full re-derivation is
# forced, overriding establishment.
#
# WHY THIS EXISTS
# ---------------
# `_established` freezes K after ESTABLISH_SEC of trusted audio. That audio is
# whatever happened to arrive first, and in a real recording the first eight
# seconds are very often ONE person — someone presses record and starts talking
# before anyone else says anything. The count then freezes at 1, and the only
# route to a second identity is `_discover`, which is deliberately narrow: it
# needs a coherent block of audio far from every centroid, and by the time
# enough of it exists the EMA in `_adapt` has already dragged the single
# centroid toward the mixture of both voices, so nothing looks far any more.
#
# The watchdog closes that trap. It does not tune a threshold; it notices that
# the model is failing to explain the audio and rebuilds it. Establishment
# still does its job — stopping a marginally better silhouette from deleting a
# speaker every four seconds — but it can no longer outrank evidence.
REDERIVE_UNEXPLAINED_SEC = 5.0

# Total trusted audio required before a forced re-derivation may fire, so a
# session cannot re-cluster on its second pass over a stray noise burst.
REDERIVE_MIN_SESSION_SEC = 10.0

# v13: windows sampled when re-estimating the band on the cheap (established)
# path. 300 windows is 45k pairwise distances — microseconds — while the full
# CLUSTER_SAMPLE_CAP of 800 would be 320k on every single pass.
ADAPT_CALIB_CAP = 300


def l2norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


@dataclass
class Window:
    start: float
    end: float
    embedding: np.ndarray
    prev: int = -1  # last display id assigned to this window
    # v13: False when a detected speaker change falls inside this window, so
    # the embedding is a mixture of two voices rather than one person's
    # fingerprint. Set by `_mark_turn_boundaries`; see changepoint.py.
    pure: bool = True

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def trusted(self) -> bool:
        """May this window help decide WHO EXISTS?

        Long enough to carry identity, and not straddling a turn change. A
        window that fails either test is still LABELLED — dropping it would
        leave a hole in the transcript's speaker timeline — it simply does not
        get a vote on how many people are in the room.
        """
        return self.pure and self.duration >= TRUST_SEC


@dataclass
class Identity:
    """One person, persisted across passes.

    `key` is internal and never shown. `display` is the 0-based number the UI
    sees, recomputed every pass as an index into the registry ordered by first
    appearance — so it is dense by construction.
    """

    key: int
    centroid: np.ndarray
    duration: float
    first_seen: float
    display: int = 0


class SpeakerEngine:
    def __init__(
        self,
        expected_speakers: int = 0,
        max_speakers: int = 6,
        speaker_mode: str = "auto",
        same_speaker_max: float = SAME_SPEAKER_MAX,
        min_cluster_sec: float = 4.0,
        min_cluster_frac: float = 0.02,
        median_frames: int = 5,
        identity_match_max: float = IDENTITY_MATCH_MAX,
        recluster_after_sec: float = RECLUSTER_AFTER_SEC,
        establish_sec: float = ESTABLISH_SEC,
        calibrate: bool = True,
        separation_margin: float = 0.0,
        detect_turns: bool = True,
        split_distance: float = 0.0,  # accepted for compatibility, unused
        **_ignored,
    ):
        self.expected_speakers = max(0, int(expected_speakers))
        self.max_speakers = max(1, int(max_speakers))
        self.same_speaker_max = float(same_speaker_max)
        self.min_cluster_sec = float(min_cluster_sec)
        self.min_cluster_frac = float(min_cluster_frac)
        self.median_frames = int(median_frames) | 1
        self.identity_match_max = float(identity_match_max)
        self.recluster_after_sec = float(recluster_after_sec)
        self.establish_sec = float(establish_sec)

        # ---- v13: session-adaptive distance band ---------------------------
        # `same_speaker_max` above is now only a FALLBACK, used when there is
        # too little audio to estimate anything. With calibration on, the
        # threshold that decides whether two centroids are two people is
        # measured from this recording. See calibration.py for why a constant
        # cannot work: cosine distance moves with the microphone and the room,
        # so 0.50 is right for some recordings and catastrophically wrong for
        # others — and when it is wrong in the low direction, EVERY K >= 2 is
        # vetoed and the whole session collapses onto one speaker.
        self.calibrate = bool(calibrate)
        self.separation_margin = float(separation_margin) or 0.0
        self._band: Optional[Band] = None

        # ---- v12: mode ----------------------------------------------------
        # "auto"  -> estimate K; expected_speakers is a CEILING (v11 behaviour,
        #            and still the default, so nothing that did not ask for
        #            fixed K changes).
        # "fixed" -> K IS expected_speakers. Requires a count to fix to; asking
        #            for fixed mode without one is a configuration mistake, and
        #            silently obeying it would give auto behaviour under a name
        #            that promises otherwise.
        mode = (speaker_mode or "auto").strip().lower()
        if mode not in ("auto", "fixed"):
            log.warning("unknown speaker_mode=%r — falling back to auto", speaker_mode)
            mode = "auto"
        if mode == "fixed" and self.expected_speakers <= 0:
            log.warning(
                "speaker_mode=fixed needs expected_speakers >= 1 "
                "(got %d) — falling back to auto",
                self.expected_speakers,
            )
            mode = "auto"
        self.speaker_mode = mode

        # In AUTO, EXPECTED_SPEAKERS is a ceiling, not a quota: with cap=4 and
        # one person talking, auto-K still returns 1. In FIXED the cap and the
        # target are the same number by definition.
        cap = self.expected_speakers or self.max_speakers
        self.cap = max(1, min(int(cap), HARD_MAX_SPEAKERS))
        if cap > HARD_MAX_SPEAKERS:
            log.warning("speaker ceiling %d clamped to %d", cap, HARD_MAX_SPEAKERS)
        if self.speaker_mode == "fixed":
            self.expected_speakers = self.cap

        self.windows: list[Window] = []
        self._identities: list[Identity] = []
        self._centroid_mat: Optional[np.ndarray] = None
        self._display_of_cluster: np.ndarray = np.zeros(0, dtype=int)
        self._timeline: list[tuple[float, float, int]] = []
        self._next_key = 0
        self._new_trusted_sec = 0.0

        # v12: True once there is enough trusted audio for the speaker
        # structure to count as evidence rather than a guess. After this, live
        # passes adapt centroids but never re-decide K.
        self._established = False

        # v13: trusted audio that no current centroid explains. Drives the
        # watchdog above.
        self._unexplained_sec = 0.0

        # v13: speaker change times found from the window embeddings. Used to
        # demote straddling windows out of the trusted set, and handed to
        # `split_points()` so the transcript is cut at turn changes even when
        # the frame vote did not move.
        self._change_points: list[float] = []
        self.detect_turns = bool(detect_turns)

        log.info(
            "SpeakerEngine v13: mode=%s, %s, cap=%d, establish=%.1fs, "
            "calibration=%s",
            self.speaker_mode,
            (
                f"K={self.expected_speakers} (forced)"
                if self.speaker_mode == "fixed"
                else "K estimated"
            ),
            self.cap,
            self.establish_sec,
            "on" if self.calibrate else f"OFF (constant {self.same_speaker_max})",
        )

    # --------------------------------------------------------------- ingest

    # ------------------------------------------------------- distance scale

    def _thr(self) -> Band:
        """The thresholds in force right now.

        Returns the calibrated band when one exists, otherwise a Band built
        from the static `same_speaker_max`. The Band's derived properties are
        chosen so that a static band of 0.50 reproduces v12's constants
        exactly — ambiguous margin 0.05, outlier 0.75, identity match 0.55 —
        which means turning calibration off restores the previous behaviour
        rather than approximating it.
        """
        if self.calibrate and self._band is not None:
            return self._band
        return Band(
            same_max=self.same_speaker_max,
            n_windows=0,
            raw_estimate=self.same_speaker_max,
            clamped=False,
        )

    def _recalibrate(self, dist_square: np.ndarray) -> Band:
        """Re-estimate the same-speaker band from a square distance matrix.

        Smoothed into the running value so one noisy pass cannot move the
        threshold far enough to change K — that instability is exactly what the
        v12 establishment logic exists to prevent, and a jumpy threshold would
        reintroduce it through the back door.
        """
        if not self.calibrate:
            return self._thr()
        kwargs = {}
        if self.separation_margin > 0:
            kwargs["margin"] = self.separation_margin
        fresh = estimate_band(
            dist_square,
            fallback=self.same_speaker_max,
            neighbours=CORE_NEIGHBOURS,
            **kwargs,
        )
        self._band = blend(self._band, fresh)
        return self._band

    # ------------------------------------------------------ turn boundaries

    def _mark_turn_boundaries(self) -> None:
        """Find speaker changes and demote the windows that straddle them.

        Runs on every pass because the boundary set grows with the session, and
        because a window at the buffer edge on one pass has neighbours on the
        next. It costs one dot product per window against a window two
        positions later — the embeddings already exist, so there is no model
        work here at all.
        """
        if not self.detect_turns or len(self.windows) < 8:
            return
        spans = [(w.start, w.end) for w in self.windows]
        embs = np.stack([w.embedding for w in self.windows])
        # The absolute floor is tied to the measured same-speaker band: a
        # boundary must separate its two sides by more than two windows of one
        # person typically differ. Without it, the curve's local maxima turn a
        # monologue into a dozen phantom turns and the transcript into
        # fragments.
        self._change_points = detect_change_points(
            spans, embs, min_height=PEAK_FLOOR_FRAC * self._thr().same_max
        )
        straddling = mark_straddling(spans, self._change_points)
        for w, bad in zip(self.windows, straddling):
            w.pure = not bool(bad)

    def change_points(self) -> list[float]:
        """Detected speaker change times, ascending.

        `DiarizationService.split_points` unions these with the timeline's own
        changes. That matters for the transcript: the frame vote is smoothed
        and short runs are absorbed, so a real turn change can leave no trace
        in the timeline while still being clearly visible in the distance
        curve. Without this, the two speakers end up in one paragraph and
        `unsplit_chunks` logs it after the fact.
        """
        return list(self._change_points)

    def add_windows(self, windows: list[Window]) -> None:
        if not windows:
            return
        self.windows.extend(windows)
        self.windows.sort(key=lambda w: w.start)
        self._new_trusted_sec += sum(w.duration for w in windows if w.trusted)

    def n_windows(self) -> int:
        return len(self.windows)

    # ------------------------------------------------------------ main pass

    def recluster(self, force: bool = False) -> bool:
        """Re-derive speakers and the timeline. True if the timeline changed.

        `force=True` is FINALISATION — the user stopped recording, every window
        of the session exists, and a whole-session pass is now plain offline
        diarization. That is the one place a full re-derivation is still
        allowed to change the number of identities.

        Everything else is LIVE, and the policy is the review's:

            not established yet   provisional; keep re-deriving, K may move
            established           never re-decide K; adapt centroids, assign
                                  incoming windows to existing identities,
                                  discover a new one only on strong evidence

        v11 ran a full re-derivation every RECLUSTER_AFTER_SEC regardless, so
        four seconds of audio could delete a speaker that ninety seconds of
        audio had established. That is the collapse this method now prevents.
        """
        self._mark_turn_boundaries()

        trusted = [w for w in self.windows if w.trusted]
        if len(trusted) < 2:
            # Every window straddles a turn change, or there are too few. Fall
            # back to length alone rather than refusing to cluster: a wrong
            # speaker count beats no labels at all.
            trusted = [w for w in self.windows if w.duration >= TRUST_SEC]
        if len(trusted) < 2:
            return False

        if force:
            before = len(self._identities)
            if not self._derive_clusters(trusted, final=True):
                return False
            self._new_trusted_sec = 0.0
            after = len(self._identities)
            if before and after != before:
                # Worth a loud line: this is the one pass permitted to change
                # the answer, so if the transcript renumbers at the very end,
                # this log says why.
                log.warning(
                    "final pass revised the speaker count %d -> %d "
                    "(whole-session re-clustering)",
                    before,
                    after,
                )
            self._established = True
            return self._label_and_build()

        if not self._established:
            # Provisional phase. Behaves exactly like v11 — including K moving
            # between passes — because with a few seconds of audio it SHOULD.
            if self._centroid_mat is None or (
                self._new_trusted_sec >= self.recluster_after_sec
            ):
                if not self._derive_clusters(trusted):
                    return False
                self._new_trusted_sec = 0.0
                # v13: never establish against an UNMEASURED distance scale.
                # With calibration on, the first pass or two run on the static
                # fallback because there are too few windows to estimate from.
                # Freezing K on that verdict is how a session ends up
                # permanently certain there is one speaker.
                measured = (not self.calibrate) or self._thr().measured
                if self._trusted_sec(trusted) >= self.establish_sec and measured:
                    self._established = True
                    log.info(
                        "speaker identities ESTABLISHED after %.1fs of trusted "
                        "audio: %d speaker(s), mode=%s. Live passes will now "
                        "adapt centroids instead of re-deciding K.",
                        self._trusted_sec(trusted),
                        len(self._identities),
                        self.speaker_mode,
                    )
            return self._label_and_build()

        # Established. The expensive, destructive part is simply not run —
        # unless the watchdog says the current model no longer describes the
        # audio, which outranks establishment.
        if self._needs_rederive(trusted):
            log.warning(
                "%.1fs of trusted audio fits no established speaker — forcing "
                "a full re-derivation (was %d speaker(s))",
                self._unexplained_sec,
                len(self._identities),
            )
            self._new_trusted_sec = 0.0
            if self._derive_clusters(trusted):
                self._unexplained_sec = 0.0
                return self._label_and_build()

        if self._new_trusted_sec >= self.recluster_after_sec:
            self._new_trusted_sec = 0.0
            self._adapt(trusted)
        return self._label_and_build()

    def _needs_rederive(self, trusted: list[Window]) -> bool:
        """Is there enough trusted audio that no established identity explains?

        Measured against the CURRENT centroids, so it stops firing by itself
        once a re-derivation has produced identities that cover the audio —
        there is no separate 'watchdog satisfied' state to keep in step.
        """
        if self._centroid_mat is None or not trusted:
            return False
        if self._trusted_sec(trusted) < REDERIVE_MIN_SESSION_SEC:
            return False
        X = np.stack([w.embedding for w in trusted])
        dur = np.array([w.duration for w in trusted], dtype=np.float64)
        best = cdist(X, self._centroid_mat, metric="cosine").min(axis=1)
        self._unexplained_sec = float(dur[best > self._thr().new_identity_min].sum())
        return self._unexplained_sec >= REDERIVE_UNEXPLAINED_SEC

    @staticmethod
    def _trusted_sec(trusted: list[Window]) -> float:
        return float(sum(w.duration for w in trusted))

    # ---------------------------------------------------- cluster discovery

    def _derive_clusters(self, trusted: list[Window], final: bool = False) -> bool:
        X = np.stack([w.embedding for w in trusted])
        dur = np.array([w.duration for w in trusted], dtype=np.float64)
        starts = np.array([w.start for w in trusted], dtype=np.float64)

        sample = self._sample(len(trusted))
        Xs, ws = X[sample], dur[sample]

        # Strip non-speech and one-off windows BEFORE the dendrogram is built.
        # This is not cosmetic: average linkage peels outliers off at the TOP of
        # the tree, so with junk present the first splits separate noise from
        # speech instead of separating people, and no choice of K can recover
        # the speaker structure from that tree.
        Dsq_s = squareform(pdist(Xs, metric="cosine"))
        band = self._recalibrate(Dsq_s)
        # band=None with calibration off keeps the v12 core rule byte for
        # byte, so `DIARIZATION_CALIBRATE=false` is a true revert.
        core = self._core_mask(Dsq_s, band if self.calibrate else None)
        if core.sum() < max(4, self.cap * 2):
            core = np.ones(len(Xs), dtype=bool)
        Xc, wc = Xs[core], ws[core]

        Dc = pdist(Xc, metric="cosine")
        Z = linkage(Dc, method="average")

        # v12: FIXED mode does not ask how many speakers there are. The user
        # already answered that question, and `_choose_k`'s separation veto is
        # exactly what turned that answer back into "maybe one".
        forced = self._forced_k(self._trusted_sec(trusted), len(Xc), final=final)
        if forced:
            k = forced
            lab_c = (
                fcluster(Z, k, criterion="maxclust") - 1
                if k > 1
                else np.zeros(len(Xc), dtype=int)
            )
            sil, dmin = self._diagnostics(squareform(Dc), Xc, wc, lab_c, k)
        else:
            k, sil, dmin = self._choose_k(Z, squareform(Dc), Xc, wc, band)
            if k <= 1:
                lab_c = np.zeros(len(Xc), dtype=int)
            else:
                lab_c = fcluster(Z, k, criterion="maxclust") - 1
        cents = self._centroids(Xc, wc, lab_c, max(1, k))

        # Refine on the FULL trusted set, twice, ignoring ambiguous and
        # nobody-shaped windows so neither turn boundaries nor noise can drag a
        # centroid toward its neighbour.
        for _ in range(2):
            lab, margin, best = self._nearest(X, cents)
            clean = (margin >= band.ambiguous_margin) & (best <= band.outlier_max)
            if clean.sum() < len(cents) * 2:
                break
            cents = self._centroids(
                X[clean], dur[clean], lab[clean], len(cents), fallback=cents
            )

        lab, margin, best = self._nearest(X, cents)
        clean = (margin >= band.ambiguous_margin) & (best <= band.outlier_max)
        if forced:
            # Both `_prune` and `_enforce_cap` exist to DELETE clusters, and a
            # deletion here is precisely how a forced K=2 would decay back into
            # K=1. A cluster that looks too small is not evidence that the
            # second person is imaginary; it is evidence that they have not
            # said much yet.
            cents, lab = self._enforce_exact_k(cents, lab, dur, X, forced)
        else:
            cents, lab = self._prune(cents, lab, dur * clean, X)
            cents, lab = self._enforce_cap(cents, lab, dur, X)

        spans = [
            (
                float(starts[lab == c].min()) if np.any(lab == c) else 0.0,
                float(dur[lab == c].sum()),
            )
            for c in range(len(cents))
        ]
        self._register(cents, spans)

        log.info(
            "clusters: %d trusted window(s) (%d sampled) -> %d speaker(s) "
            "[%s] (silhouette=%.3f, closest centroids=%.3f, band %s)",
            len(trusted),
            len(Xs),
            len(cents),
            f"K={forced} forced" if forced else "K estimated",
            sil,
            dmin,
            band.as_log(),
        )
        return True

    def _forced_k(self, trusted_sec: float, n_core: int, final: bool = False) -> int:
        """The K that must be used, or 0 to let `_choose_k` decide.

        FIXED mode holds back until `establish_sec` of trusted speech exists.
        Splitting three seconds of one voice into two identities and then
        locking them in is worse than a few seconds of provisional auto
        behaviour — and the review's caveat about fixed K splitting a single
        speaker is at its sharpest when there is barely any audio to judge on.
        """
        if self.speaker_mode != "fixed" or self.expected_speakers <= 0:
            return 0
        if not final and not self._established and trusted_sec < self.establish_sec:
            return 0
        # Never ask for more clusters than there are points to build them from.
        return max(1, min(self.expected_speakers, n_core))

    @staticmethod
    def _diagnostics(
        Dsq: np.ndarray, X: np.ndarray, w: np.ndarray, lab: np.ndarray, k: int
    ) -> tuple[float, float]:
        """Silhouette and closest-centroid distance for a K we did not choose.

        Purely observational — nothing branches on these in fixed mode. They go
        into the log line because `closest centroids=` is how you SEE that two
        speakers in a recording sit at 0.43 and would have been vetoed at the
        0.50 threshold in auto mode.
        """
        if k < 2 or len(np.unique(lab)) < 2:
            return 0.0, 1.0
        cents = SpeakerEngine._centroids(X, w, lab, k)
        dmin = float(pdist(cents, metric="cosine").min())
        return SpeakerEngine._silhouette(Dsq, lab, w, k), dmin

    def _enforce_exact_k(
        self,
        cents: np.ndarray,
        lab: np.ndarray,
        dur: np.ndarray,
        X: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Leave the session with exactly `k` centroids — no more, no fewer.

        Too many: keep the k that own the most speech and re-assign the rest.
        Too few (a refinement pass emptied a cluster): re-seed the missing
        centroid from the windows that fit the survivors WORST, which is the
        best available guess at the person who has not been heard from much.
        """
        if len(cents) > k:
            totals = np.array([dur[lab == c].sum() for c in range(len(cents))])
            keep = sorted(np.argsort(-totals)[:k].tolist())
            return self._remap(cents, keep, dur, X)

        while len(cents) < k and len(X) > len(cents):
            _, _, best = self._nearest(X, cents)
            seed = int(np.argmax(best))
            if best[seed] <= 1e-9:
                break
            cents = np.vstack([cents, l2norm(X[seed])])
            log.debug("fixed K: re-seeded a missing centroid (d=%.3f)", best[seed])

        lab, _, _ = self._nearest(X, cents)
        return cents, lab

    def _sample(self, n: int) -> np.ndarray:
        """Uniform-over-time subsample. self.windows is already time-sorted, so
        a linspace stride covers the whole session and preserves each speaker's
        share of it."""
        if n <= CLUSTER_SAMPLE_CAP:
            return np.arange(n)
        return np.unique(np.linspace(0, n - 1, CLUSTER_SAMPLE_CAP).round().astype(int))

    def _choose_k(
        self,
        Z: np.ndarray,
        Dsq: np.ndarray,
        X: np.ndarray,
        w: np.ndarray,
        band: Optional[Band] = None,
    ) -> tuple[int, float, float]:
        """Pick K by weighted silhouette among Ks whose centroids are all at
        least `band.same_max` apart.

        The separation veto is what makes K=1 reachable: if every split
        produces two centroids that are the same voice, no K>=2 is valid and
        the answer is one speaker. Selection is still silhouette-driven, so a
        K=2 split that happens to blur two pairs of speakers does not stop K=4
        from being chosen.

        WHAT CHANGED IN v13, AND WHY IT IS THE WHOLE FIX
        ------------------------------------------------
        The veto used to compare against a CONSTANT 0.50. Two different people
        on one shared microphone in a small room sit at 0.40-0.48 — the same
        two people on separate headsets sit at 0.75. Nothing about the speakers
        changes between those recordings; the acoustics do. With the constant,
        the first recording has NO valid K >= 2, `best_k` keeps its initialiser
        of 1, and every word in the session is attributed to one person. That
        is the reported symptom, and `tests/test_speaker_collapse.py` asserted
        it as intended behaviour.

        The threshold is now measured from the recording itself (see
        calibration.py). The veto still exists and still protects the
        one-speaker case — a monologue's within-speaker spread rises with the
        band, so a bipartition of one voice stays under it — but it no longer
        depends on the room matching whatever room the constant was tuned in.
        """
        thr = (band or self._thr()).same_max
        cap = min(self.cap, len(X) - 1)
        best_k, best_sil, best_dmin = 1, 0.0, 1.0
        found = False
        vetoed: list[tuple[int, float]] = []
        for k in range(2, cap + 1):
            lab = fcluster(Z, k, criterion="maxclust") - 1
            if len(np.unique(lab)) < k:
                continue
            cents = self._centroids(X, w, lab, k)
            dmin = float(pdist(cents, metric="cosine").min())
            if dmin < thr:
                vetoed.append((k, dmin))
                continue
            sil = self._silhouette(Dsq, lab, w, k)
            if not found or sil > best_sil:
                found, best_k, best_sil, best_dmin = True, k, sil, dmin
        if not found and vetoed:
            # Every split looked like one voice. That is a legitimate answer
            # (a monologue), but it is also what a mis-set threshold looks
            # like, so record the near miss rather than collapsing silently.
            k_near, d_near = max(vetoed, key=lambda kv: kv[1])
            log.info(
                "no K>=2 cleared the separation band (%.3f); closest was K=%d "
                "at %.3f -> reporting one speaker",
                thr,
                k_near,
                d_near,
            )
        return best_k, best_sil, best_dmin

    @staticmethod
    def _silhouette(Dsq: np.ndarray, lab: np.ndarray, w: np.ndarray, k: int) -> float:
        n = len(lab)
        mean_to = np.full((n, k), np.inf)
        totals = np.zeros(k)
        for c in range(k):
            wc = np.where(lab == c, w, 0.0)
            s = wc.sum()
            totals[c] = s
            if s > 1e-9:
                mean_to[:, c] = (Dsq @ wc) / s

        own = mean_to[np.arange(n), lab]
        own_tot = totals[lab]
        rest = np.maximum(own_tot - w, 1e-9)
        a = np.where(own_tot - w > 1e-9, own * own_tot / rest, 0.0)

        other = mean_to.copy()
        other[np.arange(n), lab] = np.inf
        b = other.min(axis=1)

        denom = np.maximum(a, b)
        s = np.where(np.isfinite(b) & (denom > 1e-12), (b - a) / denom, 0.0)
        return float((s * w).sum() / max(w.sum(), 1e-9))

    @staticmethod
    def _centroids(
        X: np.ndarray,
        w: np.ndarray,
        lab: np.ndarray,
        k: int,
        fallback: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        out = []
        for c in range(k):
            m = lab == c
            if not np.any(m) or w[m].sum() <= 0:
                out.append(
                    fallback[c] if fallback is not None else l2norm(X.mean(axis=0))
                )
                continue
            wc = w[m] / w[m].sum()
            out.append(l2norm((X[m] * wc[:, None]).sum(axis=0)))
        return np.stack(out)

    @staticmethod
    def _nearest(
        X: np.ndarray, cents: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Nearest centroid per row, the margin to the runner-up, and the
        distance to the winner.

        The margin is the confidence signal used everywhere else: a small margin
        means "this window sounds like two people", which is exactly what a
        window spanning a turn change looks like. The winning distance separates
        that from "this window sounds like nobody", which is noise.
        """
        d = cdist(X, cents, metric="cosine")
        lab = np.argmin(d, axis=1)
        best = d[np.arange(len(X)), lab]
        if cents.shape[0] < 2:
            return lab, np.full(len(X), 1.0), best
        part = np.partition(d, 1, axis=1)
        return lab, part[:, 1] - part[:, 0], best

    @staticmethod
    def _core_mask(Dsq: np.ndarray, band: Optional[Band] = None) -> np.ndarray:
        """Keep windows that have neighbours who sound like them.

        A voice recurs: any genuine speaker window has several other windows of
        the same person within same-speaker range. Music, laughter, coughs and
        chair scrapes do not — they are mutually unrelated, so their nearest
        neighbours are far away. Removing them here is what lets the dendrogram
        spend its top splits on people.

        v13: the cut-off scales with the calibrated band instead of being the
        same 0.50 that broke the veto. It is also floored at the 70th
        percentile of the observed spread, so a recording whose within-speaker
        spread is genuinely wide cannot have most of its speech classified as
        junk — losing 30% of windows from the dendrogram is recoverable, losing
        80% is not.
        """
        n = len(Dsq)
        m = min(CORE_NEIGHBOURS, n - 1)
        if m < 1:
            return np.ones(n, dtype=bool)
        knn = knn_spread(Dsq, m)
        if band is None:
            return knn <= SAME_SPEAKER_MAX
        ceiling = max(band.core_max, float(np.percentile(knn, 70.0)))
        return knn <= ceiling

    def _prune(
        self,
        cents: np.ndarray,
        lab: np.ndarray,
        dur: np.ndarray,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Drop clusters too small to be a participant, RELATIVE to the session.

        `dur` here is CLEAN duration — ambiguous and noise-shaped windows are
        zeroed by the caller — so a cluster assembled entirely from turn
        boundaries measures zero seconds and cannot buy its way past the floor.

        A person in a conversation holds at least a couple of percent of the
        speech. A cluster made of turn boundaries and coughs holds a few
        seconds no matter how long the session runs, which is why v10's fixed
        3-second floor stopped working after the first minute.
        """
        total = float(dur.sum())
        floor = max(self.min_cluster_sec, self.min_cluster_frac * total)
        totals = np.array([dur[lab == c].sum() for c in range(len(cents))])
        counts = np.array([int((lab == c).sum()) for c in range(len(cents))])

        keep = [c for c in range(len(cents)) if totals[c] >= floor and counts[c] >= 3]
        if not keep:
            keep = [int(np.argmax(totals))]
        if len(keep) == len(cents):
            return cents, lab

        log.debug(
            "pruned %d cluster(s) under %.1fs (durations=%s)",
            len(cents) - len(keep),
            floor,
            np.round(totals, 1).tolist(),
        )
        return self._remap(cents, keep, dur, X)

    def _enforce_cap(
        self,
        cents: np.ndarray,
        lab: np.ndarray,
        dur: np.ndarray,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(cents) <= self.cap:
            return cents, lab
        totals = np.array([dur[lab == c].sum() for c in range(len(cents))])
        keep = sorted(np.argsort(-totals)[: self.cap].tolist())
        return self._remap(cents, keep, dur, X)

    def _remap(
        self,
        cents: np.ndarray,
        keep: list[int],
        dur: np.ndarray,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Re-assign everything to the surviving centroids, then re-estimate
        them from their new membership. Discarded audio is absorbed by the
        nearest real speaker — never left to spawn its own."""
        band = self._thr()
        kept = cents[keep]
        lab, margin, best = self._nearest(X, kept)
        clean = (margin >= band.ambiguous_margin) & (best <= band.outlier_max)
        if clean.sum() >= len(kept) * 2:
            kept = self._centroids(
                X[clean], dur[clean], lab[clean], len(kept), fallback=kept
            )
            lab, _, _ = self._nearest(X, kept)
        return kept, lab

    # ------------------------------------------------------------- identity

    def _register(self, cents: np.ndarray, spans: list[tuple[float, float]]) -> None:
        """Attach this pass's clusters to persistent identities BY VOICE.

        v10 matched on how windows were labelled last pass, so a single bad
        pass propagated forever and every unmatched cluster burned a fresh
        number. Matching centroid-to-centroid instead means an identity is
        recognised because it sounds the same, and display numbers are indices
        into this registry — dense, and bounded by the number of people.
        """
        taken: dict[int, int] = {}
        if self._identities:
            gate = (
                self._thr().identity_match_max
                if (self.calibrate and self._band is not None)
                else self.identity_match_max
            )
            prev = np.stack([i.centroid for i in self._identities])
            d = cdist(cents, prev, metric="cosine")
            cost = np.where(d <= gate, d, 1e3)
            for r, c in zip(*linear_sum_assignment(cost)):
                if cost[r, c] < 1e3:
                    taken[int(r)] = int(c)

        fresh: list[Identity] = []
        for r in range(len(cents)):
            start, dur = spans[r]
            if r in taken:
                old = self._identities[taken[r]]
                fresh.append(
                    Identity(
                        key=old.key,
                        centroid=cents[r],
                        duration=dur,
                        first_seen=min(old.first_seen, start),
                    )
                )
            else:
                self._next_key += 1
                fresh.append(
                    Identity(
                        key=self._next_key,
                        centroid=cents[r],
                        duration=dur,
                        first_seen=start,
                    )
                )

        # Dense display numbers, ordered by when the person first spoke — so
        # Speaker 1 is whoever opened the recording. first_seen is quantised to
        # whole seconds and ties broken by discovery order, because otherwise a
        # 200 ms shift in one identity's earliest window could renumber the
        # whole transcript and force a pointless refresh.
        order = sorted(
            range(len(fresh)),
            key=lambda i: (round(fresh[i].first_seen), fresh[i].key),
        )
        for display, i in enumerate(order):
            fresh[i].display = display

        self._identities = fresh
        self._centroid_mat = cents
        self._display_of_cluster = np.array([i.display for i in fresh], dtype=int)

    # --------------------------------------------------- live adaptation (v12)

    def _adapt(self, trusted: list[Window]) -> None:
        """The live pass, once identities are established.

        This is the whole point of v12. It does the three things the review
        asked for and nothing else:

            1. re-estimate each established centroid CONSERVATIVELY (EMA), so a
               person who moves closer to the microphone is tracked without one
               pass being able to redefine who they are;
            2. leave K alone — no `_choose_k`, no prune, no cap enforcement, so
               there is no code path here that can delete Speaker 2;
            3. in AUTO mode only, admit a new identity when the evidence is
               strong (see `_discover`).

        Note what is absent: no linkage, no silhouette, no K search. An
        established session's live cost is now an argmin over centroids.
        """
        if self._centroid_mat is None:
            return

        cents = self._centroid_mat.copy()
        X = np.stack([w.embedding for w in trusted])
        dur = np.array([w.duration for w in trusted], dtype=np.float64)
        starts = np.array([w.start for w in trusted], dtype=np.float64)

        # Keep the band current even though K is frozen. A speaker leaning back
        # from the microphone widens the within-speaker spread over minutes; if
        # the band did not follow, `_discover` would start reading that drift as
        # a new person. ADAPT_CALIB_CAP bounds the O(n^2) cost — this method is
        # the cheap path and must stay cheap.
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
                [
                    l2norm((1.0 - a) * cents[i] + a * target[i])
                    for i in range(len(cents))
                ]
            )

        fresh = self._discover(X, dur, cents)
        if fresh is not None:
            cents = np.vstack([cents, fresh])
            log.info(
                "new speaker identity discovered from %.1fs of audio unlike "
                "any known voice (band %s) -> %d speaker(s)",
                NEW_IDENTITY_MIN_SEC,
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
        # `_register` matches centroid-to-centroid, and the EMA guarantees each
        # one barely moved, so every established identity re-matches itself and
        # keeps its display number. That is what makes labels stable on screen.
        self._register(cents, spans)

    def _discover(
        self, X: np.ndarray, dur: np.ndarray, cents: np.ndarray
    ) -> Optional[np.ndarray]:
        """A genuinely new voice, or None.

        The review's condition is "consistently far from ALL known speaker
        centroids", and `consistently` is doing real work: a handful of windows
        far from everyone is what a cough, a door, or a turn boundary looks
        like. So a candidate must clear three bars, not one:

            far        every window further than same_speaker_max from EVERY
                       established centroid
            enough     NEW_IDENTITY_MIN_SEC of it, over several windows
            coherent   the far windows must sound like EACH OTHER, not merely
                       unlike us — noise is unlike everything including itself

        FIXED mode never discovers: the user stated the count.
        """
        if self.speaker_mode == "fixed":
            return None
        if len(cents) >= self.cap:
            return None

        far_thr = self._thr().new_identity_min
        best = cdist(X, cents, metric="cosine").min(axis=1)
        far = best > far_thr
        if int(far.sum()) < NEW_IDENTITY_MIN_WINDOWS:
            return None
        if float(dur[far].sum()) < NEW_IDENTITY_MIN_SEC:
            return None

        Xf, wf = X[far], dur[far]
        cand = l2norm((Xf * (wf / wf.sum())[:, None]).sum(axis=0))

        # Coherence: keep only the far windows that agree with the candidate,
        # then re-check that what remains is still substantial.
        agree = cdist(Xf, cand[None, :], metric="cosine").ravel() <= far_thr
        if int(agree.sum()) < NEW_IDENTITY_MIN_WINDOWS:
            return None
        if float(wf[agree].sum()) < NEW_IDENTITY_MIN_SEC:
            return None

        Xa, wa = Xf[agree], wf[agree]
        cand = l2norm((Xa * (wa / wa.sum())[:, None]).sum(axis=0))

        # Final guard: the candidate must still be a different person from
        # everyone already registered, measured centroid-to-centroid.
        if float(cdist(cand[None, :], cents, metric="cosine").min()) < far_thr:
            return None
        return cand

    # ------------------------------------------------- labelling & timeline

    def _label_and_build(self) -> bool:
        if self._centroid_mat is None or not self.windows:
            return False

        X = np.stack([w.embedding for w in self.windows])
        cluster, margin, best = self._nearest(X, self._centroid_mat)
        labels = self._display_of_cluster[cluster]

        for w, sid in zip(self.windows, labels):
            w.prev = int(sid)

        prev_timeline = self._timeline
        self._timeline = self._build_timeline(labels, margin, best)

        changed = self._timeline != prev_timeline
        if changed:
            spk = sorted({s for _, _, s in self._timeline})
            log.info(
                "timeline: %d window(s) -> %d speaker(s) %s",
                len(self.windows),
                len(spk),
                [s + 1 for s in spk],
            )
        return changed

    def _build_timeline(
        self, labels: np.ndarray, margin: np.ndarray, best: np.ndarray
    ) -> list[tuple[float, float, int]]:
        if not self.windows:
            return []
        t0 = min(w.start for w in self.windows)
        t1 = max(w.end for w in self.windows)
        n = max(1, int(np.ceil((t1 - t0) / FRAME_SEC)))
        ids = sorted(set(int(x) for x in labels))
        index = {s: i for i, s in enumerate(ids)}
        votes = np.zeros((n, len(ids)), dtype=np.float64)

        for w, lab, m, d1 in zip(self.windows, labels, margin, best):
            a = int((w.start - t0) / FRAME_SEC)
            b = int(np.ceil((w.end - t0) / FRAME_SEC))
            a, b = max(0, a), min(n, b)
            if b <= a:
                continue
            # Confidence: an ambiguous window (two centroids nearly tied) is
            # usually two voices at once. It still votes, quietly.
            conf = float(np.clip(m / 0.15, 0.25, 1.0))
            if d1 > self._thr().outlier_max:
                # Resembles nobody in the room. Keep the label for continuity,
                # but do not let it decide a turn.
                conf *= 0.3
            if not w.trusted:
                conf *= 0.6
            centre = (a + b - 1) / 2.0
            half = max(1.0, (b - a) / 2.0)
            f = np.arange(a, b)
            wgt = (0.25 + 0.75 * (1.0 - np.abs(f - centre) / half)) * conf
            votes[a:b, index[int(lab)]] += wgt

        active = votes.sum(axis=1) > 0
        frames = np.full(n, -1, dtype=int)
        frames[active] = np.array(ids)[np.argmax(votes[active], axis=1)]
        frames = self._median_filter(frames)
        frames = self._absorb_short_runs(frames, votes, ids)

        runs: list[tuple[float, float, int]] = []
        i = 0
        while i < n:
            if frames[i] < 0:
                i += 1
                continue
            j = i
            while j + 1 < n and frames[j + 1] == frames[i]:
                j += 1
            runs.append((t0 + i * FRAME_SEC, t0 + (j + 1) * FRAME_SEC, int(frames[i])))
            i = j + 1
        return runs

    def _median_filter(self, frames: np.ndarray) -> np.ndarray:
        k = self.median_frames
        if k < 3 or len(frames) < k:
            return frames
        out = frames.copy()
        r = k // 2
        for i in range(r, len(frames) - r):
            if frames[i] < 0:
                continue
            win = frames[i - r : i + r + 1]
            win = win[win >= 0]
            if len(win) < 3:
                continue
            vals, counts = np.unique(win, return_counts=True)
            out[i] = int(vals[int(np.argmax(counts))])
        return out

    def _absorb_short_runs(
        self, frames: np.ndarray, votes: np.ndarray, ids: list[int]
    ) -> np.ndarray:
        """Absorb sub-MIN_RUN_SEC runs into a neighbour.

        Nobody takes a turn for 0.5 s. A run that short is a labelling wobble,
        and left alone it becomes its own paragraph with timestamps overlapping
        the lines either side of it.
        """
        min_frames = max(1, int(round(MIN_RUN_SEC / FRAME_SEC)))
        if min_frames < 2:
            return frames
        index = {s: i for i, s in enumerate(ids)}

        for _ in range(4):
            spans = []
            i = 0
            n = len(frames)
            while i < n:
                if frames[i] < 0:
                    i += 1
                    continue
                j = i
                while j + 1 < n and frames[j + 1] == frames[i]:
                    j += 1
                spans.append((i, j, int(frames[i])))
                i = j + 1
            if len(spans) < 2:
                return frames

            victim = None
            for pos, (a, b, lab) in enumerate(spans):
                if (b - a + 1) >= min_frames:
                    continue
                if victim is None or (b - a) < (spans[victim][1] - spans[victim][0]):
                    victim = pos
            if victim is None:
                return frames

            a, b, lab = spans[victim]
            options = []
            if victim > 0:
                options.append(spans[victim - 1][2])
            if victim + 1 < len(spans):
                options.append(spans[victim + 1][2])
            if not options:
                return frames
            # Give the frames to whichever neighbour the votes actually prefer.
            best = max(options, key=lambda s: votes[a : b + 1, index[s]].sum())
            frames[a : b + 1] = best
        return frames

    # ------------------------------------------------------------- queries

    def timeline(self) -> list[tuple[float, float, int]]:
        return list(self._timeline)

    def label_for(self, start: float, end: float) -> Optional[int]:
        if not self._timeline or end <= start:
            return None
        totals: dict[int, float] = {}
        for a, b, s in self._timeline:
            ov = min(end, b) - max(start, a)
            if ov > 0:
                totals[s] = totals.get(s, 0.0) + ov
        return max(totals, key=totals.get) if totals else None

    def speaker_count(self) -> int:
        return len({s for _, _, s in self._timeline})

    def identities(self) -> list[dict]:
        return [
            {"speaker": i.display + 1, "seconds": round(i.duration, 1)}
            for i in sorted(self._identities, key=lambda x: x.display)
        ]

    def established(self) -> bool:
        """True once the speaker structure is evidence-backed rather than a
        guess. Live passes stop re-deciding K at this point."""
        return self._established

    def reset(self) -> None:
        """Forget every speaker. Used by the explicit NEW RECORDING control.

        Deliberately not called on silence: a six-second thinking pause in an
        interview is not a new recording, and resetting there would invent a
        fresh set of identities mid-conversation.
        """
        self.windows.clear()
        self._identities.clear()
        self._centroid_mat = None
        self._display_of_cluster = np.zeros(0, dtype=int)
        self._timeline.clear()
        self._next_key = 0
        self._new_trusted_sec = 0.0
        self._established = False
        self._unexplained_sec = 0.0
        self._change_points.clear()
        # The next recording may be a different microphone, a different room or
        # a different distance from it, so the previous recording's distance
        # scale is not evidence about this one.
        self._band = None


# ---------------------------------------------------------------------------
# Speech masking — never embed silence.
# ---------------------------------------------------------------------------


def speech_regions(
    pcm: np.ndarray,
    sample_rate: int,
    offset: float = 0.0,
    aggressiveness: int = 2,
    min_speech_sec: float = 0.30,
    bridge_sec: float = 0.20,
) -> list[tuple[float, float]]:
    """Return contiguous speech spans (absolute seconds) using webrtcvad.

    Short gaps are bridged: a 100 ms stop between two words is not a turn
    boundary, and cutting there would produce windows too short to embed.
    """
    import webrtcvad

    vad = webrtcvad.Vad(aggressiveness)
    frame = int(sample_rate * 0.03)  # 30 ms
    pcm16 = np.clip(pcm * 32767.0, -32768, 32767).astype(np.int16)

    flags: list[bool] = []
    for i in range(0, len(pcm16) - frame + 1, frame):
        try:
            flags.append(vad.is_speech(pcm16[i : i + frame].tobytes(), sample_rate))
        except Exception:  # noqa: BLE001
            flags.append(False)

    spans: list[list[float]] = []
    for i, f in enumerate(flags):
        if not f:
            continue
        a, b = i * 0.03, (i + 1) * 0.03
        if spans and a - spans[-1][1] <= bridge_sec:
            spans[-1][1] = b
        else:
            spans.append([a, b])

    return [(offset + a, offset + b) for a, b in spans if b - a >= min_speech_sec]


def slice_windows(
    pcm: np.ndarray,
    sample_rate: int,
    regions: list[tuple[float, float]],
    offset: float,
) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    """Cut speech regions into WIN_SEC windows at HOP_SEC.

    Returns the raw waveforms plus their absolute (start, end) times, ready to
    hand to Embedder.embed_batch as ONE batch.
    """
    waves: list[np.ndarray] = []
    spans: list[tuple[float, float]] = []
    for a, b in regions:
        dur = b - a
        if dur < 0.90:
            continue
        t = a
        while t < b - 0.05:
            e = min(t + WIN_SEC, b)
            if e - t < 0.90:
                break
            i0 = int((t - offset) * sample_rate)
            i1 = int((e - offset) * sample_rate)
            if i1 > i0:
                waves.append(pcm[i0:i1])
                spans.append((t, e))
            if e >= b - 0.05:
                break
            t += HOP_SEC
    return waves, spans
