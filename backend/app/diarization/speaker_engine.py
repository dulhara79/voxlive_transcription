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
# Two cluster centroids further apart than this are UNCONDITIONALLY different
# people. It is applied between CENTROIDS (averages over many seconds), never
# between individual short windows — which is why it can be a constant at all.
#
# v13 (supervisor review §9): this stopped being a VETO and became a fast path.
# In v12 a candidate K was thrown away whenever its two closest centroids sat
# inside 0.50, which is the documented failure mode of the review's "Video 2":
# two genuinely different people who happen to sound alike land at 0.43 and get
# merged. See `_separation_verdict` for the test that replaced the veto.
SAME_SPEAKER_MAX = 0.50

# Absolute floor under the relative test below. Under this distance the two
# centroids are the same voice no matter how tight the clusters are, and the
# relative test is not consulted. A monologue split in two lands at ~0.11.
SAME_SPEAKER_FLOOR = 0.35

# The relative separation test (§9). Two clusters are two PEOPLE when the gap
# between their centroids is larger than the clusters are wide:
#
#     d(c_i, c_j)  >=  SEPARATION_RATIO * (radius_i + radius_j)
#
# where radius is the weighted mean distance from a cluster's members to its
# own centroid. Measured on the review's own geometry:
#
#     one person split in two   d=0.11  r=0.12+0.19   ratio 0.37   rejected
#     the 0.43 "similar" pair   d=0.44  r=0.18+0.19   ratio 1.19   accepted
#     two clear speakers        d=0.72  r=0.19+0.19   ratio 1.95   accepted
#
# This is what lets the engine keep protecting the one-speaker case (the whole
# reason the veto existed) while no longer merging real people at 0.43.
SEPARATION_RATIO = 1.10

# A candidate K in which any cluster holds fewer windows than this is not a
# speaker structure — it is one good cluster plus debris. Without this guard a
# singleton cluster has radius 0.0, which makes the ratio above meaningless.
CANDIDATE_MIN_WINDOWS = 3

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

# v13 (§10): the short-speaker path. A news reporter who says eight seconds and
# leaves cannot clear 6.0 s of TRUSTED windows, and in v12 they were silently
# absorbed into whoever they sounded least unlike. They are admitted here on a
# smaller amount of audio, but only against a much harder distance bar and a
# strict internal-coherence bar — which is exactly the combination a music
# sting, a jingle or a turn boundary cannot produce, because noise is unlike
# everything INCLUDING ITSELF.
NEW_IDENTITY_SHORT_SEC = 3.2
NEW_IDENTITY_SHORT_WINDOWS = 3
NEW_IDENTITY_STRONG_DIST = 0.68

# The far windows backing a new identity must agree with each other at least
# this well. Enforced on both the normal and the short path.
NEW_IDENTITY_MAX_RADIUS = 0.40

# v13 (§12): AUTO mode only. Trusted audio accumulated after establishment
# before the engine re-examines whether the session has grown a speaker the
# incremental discovery path missed. The re-examination may only ever ADD
# identities — see `_growth_check`.
GROWTH_CHECK_SEC = 20.0


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

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def trusted(self) -> bool:
        return self.duration >= TRUST_SEC


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
        same_speaker_floor: float = SAME_SPEAKER_FLOOR,
        separation_ratio: float = SEPARATION_RATIO,
        new_identity_min_sec: float = NEW_IDENTITY_MIN_SEC,
        new_identity_min_windows: int = NEW_IDENTITY_MIN_WINDOWS,
        new_identity_short_sec: float = NEW_IDENTITY_SHORT_SEC,
        new_identity_short_windows: int = NEW_IDENTITY_SHORT_WINDOWS,
        new_identity_strong_dist: float = NEW_IDENTITY_STRONG_DIST,
        growth_check_sec: float = GROWTH_CHECK_SEC,
        split_distance: float = 0.0,  # accepted for compatibility, unused
        **_ignored,
    ):
        self.expected_speakers = max(0, int(expected_speakers))
        self.max_speakers = max(1, int(max_speakers))
        self.same_speaker_max = float(same_speaker_max)
        # v13 §9 — the relative separation test that replaced the hard veto.
        self.same_speaker_floor = min(float(same_speaker_floor), self.same_speaker_max)
        self.separation_ratio = float(separation_ratio)
        # v13 §10 — new-identity evidence, normal path and short-speaker path.
        self.new_identity_min_sec = float(new_identity_min_sec)
        self.new_identity_min_windows = int(new_identity_min_windows)
        self.new_identity_short_sec = float(new_identity_short_sec)
        self.new_identity_short_windows = int(new_identity_short_windows)
        self.new_identity_strong_dist = float(new_identity_strong_dist)
        # v13 §12 — post-establishment upward-only re-examination.
        self.growth_check_sec = float(growth_check_sec)
        self.min_cluster_sec = float(min_cluster_sec)
        self.min_cluster_frac = float(min_cluster_frac)
        self.median_frames = int(median_frames) | 1
        self.identity_match_max = float(identity_match_max)
        self.recluster_after_sec = float(recluster_after_sec)
        self.establish_sec = float(establish_sec)

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
        # v13 §12: trusted audio since the last upward-only growth check.
        self._since_growth_sec = 0.0

        # v12: True once there is enough trusted audio for the speaker
        # structure to count as evidence rather than a guess. After this, live
        # passes adapt centroids but never re-decide K.
        self._established = False

        # §5 of the review: the runtime configuration must be READABLE in the
        # log, because the whole 2-speaker investigation came down to a .env
        # value nobody could see at run time.
        log.info("SpeakerEngine v13 runtime config: %s", self.describe())

    def describe(self) -> str:
        """One-line statement of what this engine will actually do.

        Deliberately spells out the CONSEQUENCE of the mode rather than only
        the settings, so a log reader does not have to remember which of
        expected_speakers/max_speakers is load-bearing in which mode.
        """
        if self.speaker_mode == "fixed":
            what = (
                f"K FORCED to exactly {self.expected_speakers} — "
                f"Speaker {self.expected_speakers + 1} can never be created"
            )
        else:
            what = f"K ESTIMATED from the audio, 1..{self.cap}"
        return (
            f"speaker_mode={self.speaker_mode}, "
            f"expected_speakers={self.expected_speakers}, "
            f"max_speakers={self.max_speakers}, cap={self.cap}, "
            f"establish={self.establish_sec:.1f}s, "
            f"same_speaker_max={self.same_speaker_max:.2f}, "
            f"separation_floor={self.same_speaker_floor:.2f}, "
            f"separation_ratio={self.separation_ratio:.2f} -> {what}"
        )

    # --------------------------------------------------------------- ingest

    def add_windows(self, windows: list[Window]) -> None:
        if not windows:
            return
        self.windows.extend(windows)
        self.windows.sort(key=lambda w: w.start)
        added = sum(w.duration for w in windows if w.trusted)
        self._new_trusted_sec += added
        self._since_growth_sec += added

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
        trusted = [w for w in self.windows if w.trusted]
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
                if self._trusted_sec(trusted) >= self.establish_sec:
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

        # Established. The expensive, destructive part is simply not run.
        if self._new_trusted_sec >= self.recluster_after_sec:
            self._new_trusted_sec = 0.0
            self._adapt(trusted)
        return self._label_and_build()

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
        core = self._core_mask(squareform(pdist(Xs, metric="cosine")))
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
            table = []
        else:
            k, sil, dmin, table = self._choose_k(Z, squareform(Dc), Xc, wc)
            if k <= 1:
                lab_c = np.zeros(len(Xc), dtype=int)
            else:
                lab_c = fcluster(Z, k, criterion="maxclust") - 1
            # §6: this is the line that answers "why did it say 2?".
            log.info("auto-K candidates: %s", self._format_table(table, k))
        cents = self._centroids(Xc, wc, lab_c, max(1, k))

        # Refine on the FULL trusted set, twice, ignoring ambiguous and
        # nobody-shaped windows so neither turn boundaries nor noise can drag a
        # centroid toward its neighbour.
        for _ in range(2):
            lab, margin, best = self._nearest(X, cents)
            clean = (margin >= AMBIG_MARGIN) & (best <= OUTLIER_MAX)
            if clean.sum() < len(cents) * 2:
                break
            cents = self._centroids(
                X[clean], dur[clean], lab[clean], len(cents), fallback=cents
            )

        lab, margin, best = self._nearest(X, cents)
        clean = (margin >= AMBIG_MARGIN) & (best <= OUTLIER_MAX)
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
            "[%s] (silhouette=%.3f, closest centroids=%.3f)",
            len(trusted),
            len(Xs),
            len(cents),
            f"K={forced} forced" if forced else "K estimated",
            sil,
            dmin,
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

    @staticmethod
    def _radii(X: np.ndarray, w: np.ndarray, lab: np.ndarray, cents: np.ndarray):
        """Weighted mean distance from each cluster's members to its centroid.

        This is the cluster's WIDTH. The separation test below compares the gap
        between two centroids against the sum of the two widths, which is the
        difference between "two people who sound similar" (a real gap between
        two tight clusters) and "one person cut in half" (no gap worth the
        name between two halves of one wide cluster).
        """
        out = np.zeros(len(cents))
        for c in range(len(cents)):
            m = lab == c
            if not np.any(m) or w[m].sum() <= 0:
                continue
            d = cdist(X[m], cents[c][None, :], metric="cosine").ravel()
            out[c] = float((d * w[m]).sum() / w[m].sum())
        return out

    def _separation_verdict(
        self, X: np.ndarray, w: np.ndarray, lab: np.ndarray, cents: np.ndarray
    ) -> tuple[bool, str, float, float]:
        """Are these K clusters K different PEOPLE?

        Returns (ok, reason, dmin, ratio). The order of the tests matters:

            1. dmin >= same_speaker_max      unconditionally different people
                                             (v11/v12 behaviour, unchanged)
            2. dmin <  same_speaker_floor    unconditionally the same person
            3. otherwise                     the RELATIVE test — the gap must
                                             exceed the combined width of the
                                             two closest clusters

        Step 3 is the whole of §9. v12 had only step 1 and treated everything
        below it as the same speaker, which is why two real people at 0.43 were
        merged. Step 2 keeps the one-speaker protection that the veto was there
        to provide in the first place.
        """
        D = squareform(pdist(cents, metric="cosine"))
        np.fill_diagonal(D, np.inf)
        i, j = np.unravel_index(int(np.argmin(D)), D.shape)
        dmin = float(D[i, j])

        rad = self._radii(X, w, lab, cents)
        width = float(rad[i] + rad[j])
        ratio = dmin / width if width > 1e-9 else float("inf")

        if dmin >= self.same_speaker_max:
            return True, "separated", dmin, ratio
        if dmin < self.same_speaker_floor:
            return (
                False,
                f"same voice (d={dmin:.3f} < {self.same_speaker_floor:.2f})",
                dmin,
                ratio,
            )
        if ratio >= self.separation_ratio:
            return True, "separated (relative)", dmin, ratio
        return (
            False,
            f"clusters wider than the gap (ratio={ratio:.2f} < "
            f"{self.separation_ratio:.2f})",
            dmin,
            ratio,
        )

    def _choose_k(
        self, Z: np.ndarray, Dsq: np.ndarray, X: np.ndarray, w: np.ndarray
    ) -> tuple[int, float, float, list[dict]]:
        """Pick K by weighted silhouette among the SEPARABLE candidates.

        Also returns the full candidate table (§6). Until v13 the log said only
        `-> 2 speaker(s)`, so when the engine chose 2 for a five-person news
        broadcast there was no way to tell whether K=3 lost on silhouette, was
        thrown out by the separation rule, or was never evaluated at all. Every
        candidate now records why it was kept or dropped.
        """
        cap = min(self.cap, len(X) - 1)
        best_k, best_sil, best_dmin = 1, 0.0, 1.0
        found = False
        table: list[dict] = []

        for k in range(2, cap + 1):
            lab = fcluster(Z, k, criterion="maxclust") - 1
            row: dict = {"k": k}
            table.append(row)

            if len(np.unique(lab)) < k:
                row["rejected"] = "degenerate (linkage produced fewer clusters)"
                continue

            counts = [int((lab == c).sum()) for c in range(k)]
            durs = [float(w[lab == c].sum()) for c in range(k)]
            row["windows"] = counts
            row["seconds"] = [round(d, 1) for d in durs]

            if min(counts) < CANDIDATE_MIN_WINDOWS:
                row["rejected"] = (
                    f"cluster of {min(counts)} window(s) < {CANDIDATE_MIN_WINDOWS}"
                )
                continue

            cents = self._centroids(X, w, lab, k)
            ok, why, dmin, ratio = self._separation_verdict(X, w, lab, cents)
            row["dmin"] = round(dmin, 3)
            row["ratio"] = round(ratio, 2)
            if not ok:
                row["rejected"] = why
                continue

            sil = self._silhouette(Dsq, lab, w, k)
            row["silhouette"] = round(sil, 3)
            row["accepted"] = why
            if not found or sil > best_sil:
                found, best_k, best_sil, best_dmin = True, k, sil, dmin

        return best_k, best_sil, best_dmin, table

    @staticmethod
    def _format_table(table: list[dict], selected: int) -> str:
        """The candidate table as one readable log line (§6/§21)."""
        parts = []
        for row in table:
            k = row["k"]
            if "rejected" in row:
                parts.append(f"K={k} REJECTED: {row['rejected']}")
            else:
                parts.append(
                    f"K={k} sil={row.get('silhouette', 0.0):.3f} "
                    f"dmin={row.get('dmin', 0.0):.3f} "
                    f"ratio={row.get('ratio', 0.0):.2f} "
                    f"windows={row.get('windows')} secs={row.get('seconds')}"
                )
        parts.append(f"SELECTED K={selected}")
        return " | ".join(parts)

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

    def _core_mask(self, Dsq: np.ndarray) -> np.ndarray:
        """Keep windows that have neighbours who sound like them.

        A voice recurs: any genuine speaker window has other windows of the
        same person within same-speaker range. Music, laughter, coughs and
        chair scrapes do not — they are mutually unrelated, so their nearest
        neighbours are far away. Removing them here is what lets the dendrogram
        spend its top splits on people.

        v13 (§10) — THE FIVE-NEIGHBOUR MEAN PUNISHED SHORT SPEAKERS.
        The test averaged the FIVE nearest neighbours, so a reporter with three
        trusted windows was scored on two of their own windows plus three
        windows belonging to other people:

            (2 * 0.22 + 3 * 0.78) / 5 = 0.556  >  0.50   -> deleted as noise

        A brief but genuine speaker was therefore stripped out before the
        dendrogram was even built, and no value of K could recover them. A
        window now also qualifies on its TWO nearest neighbours, against a
        tighter bar. Noise still fails: a cough has no near neighbour at all,
        which is the property the filter was actually written to exploit.
        """
        n = len(Dsq)
        m = min(CORE_NEIGHBOURS, n - 1)
        if m < 1:
            return np.ones(n, dtype=bool)
        d = Dsq.copy()
        np.fill_diagonal(d, np.inf)
        srt = np.sort(d, axis=1)
        broad = srt[:, :m].mean(axis=1) <= self.same_speaker_max
        near = min(2, m)
        tight = srt[:, :near].mean(axis=1) <= 0.85 * self.same_speaker_max
        return broad | tight

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

        # v13 (§10). A relative floor asks "does this cluster own a couple of
        # percent of the speech?", and in a 400-second news broadcast an
        # eight-second reporter owns 2 % on a good day. The floor was written
        # to delete turn-boundary debris, not people, so a cluster that looks
        # small may still stay IF it looks like a person: clearly separated
        # from everyone else, and internally coherent. Debris is neither —
        # boundary windows sit BETWEEN two centroids by construction, and noise
        # does not agree with itself.
        rad = self._radii(X, dur, lab, cents)
        cd = squareform(pdist(cents, metric="cosine")) if len(cents) > 1 else None

        def distinct(c: int) -> bool:
            if cd is None:
                return False
            others = np.delete(cd[c], c)
            return float(others.min()) >= self.same_speaker_max

        keep = []
        for c in range(len(cents)):
            if totals[c] >= floor and counts[c] >= 3:
                keep.append(c)
                continue
            if (
                counts[c] >= self.new_identity_short_windows
                and totals[c] >= self.new_identity_short_sec
                and rad[c] <= NEW_IDENTITY_MAX_RADIUS
                and distinct(c)
            ):
                keep.append(c)
                log.info(
                    "prune exemption: cluster %d kept on %.1fs / %d window(s) — "
                    "coherent (radius=%.3f) and distinct from every other "
                    "speaker. A brief speaker is not debris.",
                    c,
                    totals[c],
                    counts[c],
                    rad[c],
                )
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
        kept = cents[keep]
        lab, margin, best = self._nearest(X, kept)
        clean = (margin >= AMBIG_MARGIN) & (best <= OUTLIER_MAX)
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
            prev = np.stack([i.centroid for i in self._identities])
            d = cdist(cents, prev, metric="cosine")
            cost = np.where(d <= self.identity_match_max, d, 1e3)
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

        lab, margin, best = self._nearest(X, cents)
        clean = (margin >= AMBIG_MARGIN) & (best <= OUTLIER_MAX)
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
            log.info("speaker count is now %d", len(cents))

        # §12: the incremental path above only ever sees a voice that is far
        # from every centroid RIGHT NOW. A speaker who was folded into someone
        # else during establishment is invisible to it forever, because they
        # are no longer far from anything — they ARE part of a centroid. The
        # growth check re-asks the whole question periodically, and is allowed
        # to answer only in one direction.
        if self._since_growth_sec >= self.growth_check_sec:
            grown = self._growth_check(trusted, cents)
            if grown is not None:
                cents = grown

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

    def _growth_check(
        self, trusted: list[Window], cents: np.ndarray
    ) -> Optional[np.ndarray]:
        """Periodic, UPWARD-ONLY re-examination of the speaker count (§12).

        The review asks for a count that can evolve 1 -> 2 -> 3 -> 4 as a
        session goes on, and for established identities never to be deleted by
        a later pass (§11). Those two requirements are only compatible if the
        re-examination is asymmetric, so this one is:

            new K <= current K                       -> discarded, nothing moves
            an established centroid loses its match  -> discarded, nothing moves
            new K > current K and all matched        -> adopted

        The result is that a whole-session re-derivation can ADD Speaker 4 but
        can never take Speaker 2 away, which is exactly the collapse v12 was
        written to stop. A pass that wants FEWER speakers is not wrong — it is
        just not trusted live, and `recluster(force=True)` at finalisation is
        where it gets its say.
        """
        self._since_growth_sec = 0.0
        if self.speaker_mode != "auto" or len(cents) >= self.cap:
            return None

        X = np.stack([w.embedding for w in trusted])
        dur = np.array([w.duration for w in trusted], dtype=np.float64)

        sample = self._sample(len(trusted))
        Xs, ws = X[sample], dur[sample]
        core = self._core_mask(squareform(pdist(Xs, metric="cosine")))
        if core.sum() < max(4, self.cap * 2):
            core = np.ones(len(Xs), dtype=bool)
        Xc, wc = Xs[core], ws[core]
        if len(Xc) <= len(cents) + 1:
            return None

        Dc = pdist(Xc, metric="cosine")
        Z = linkage(Dc, method="average")
        k, sil, dmin, table = self._choose_k(Z, squareform(Dc), Xc, wc)

        if k <= len(cents):
            log.debug(
                "growth check: no new speaker (K=%d vs %d established) | %s",
                k,
                len(cents),
                self._format_table(table, k),
            )
            return None

        lab_c = fcluster(Z, k, criterion="maxclust") - 1
        grown = self._centroids(Xc, wc, lab_c, k)
        for _ in range(2):
            lab, margin, best = self._nearest(X, grown)
            clean = (margin >= AMBIG_MARGIN) & (best <= OUTLIER_MAX)
            if clean.sum() < len(grown) * 2:
                break
            grown = self._centroids(
                X[clean], dur[clean], lab[clean], len(grown), fallback=grown
            )

        # The monotone guard. Every established identity must still be findable
        # in the new structure, one-to-one — otherwise this is not "we found
        # another person", it is a re-partition, and re-partitions are what
        # renumber a transcript out from under the user.
        cost = cdist(cents, grown, metric="cosine")
        rows, cols = linear_sum_assignment(cost)
        worst = float(max(cost[r, c] for r, c in zip(rows, cols)))
        if worst > self.identity_match_max:
            log.info(
                "growth check: K=%d looked better than %d but an established "
                "identity would have moved %.3f (> %.2f) — DISCARDED, the "
                "existing speakers stand.",
                k,
                len(cents),
                worst,
                self.identity_match_max,
            )
            return None

        log.info(
            "growth check ADOPTED: %d -> %d speaker(s) after %.0fs of further "
            "audio (silhouette=%.3f, closest centroids=%.3f). Every existing "
            "identity survived (worst move %.3f). | %s",
            len(cents),
            k,
            self.growth_check_sec,
            sil,
            dmin,
            worst,
            self._format_table(table, k),
        )
        return grown

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

        v13 adds a SHORT path (§10) and, more importantly, a DIAGNOSTIC (§22).
        Every rejection now says which bar was missed and by how much, so a
        reporter who should have become Speaker 3 and did not leaves a line
        saying `windows=3 < 4` rather than nothing at all.

        FIXED mode never discovers: the user stated the count.
        """
        if self.speaker_mode == "fixed":
            return None
        if len(cents) >= self.cap:
            log.debug(
                "candidate_new_speaker: not evaluated — already at the cap of "
                "%d identities (raise MAX_SPEAKERS to go further)",
                self.cap,
            )
            return None

        d_all = cdist(X, cents, metric="cosine")
        best = d_all.min(axis=1)
        far = best > self.same_speaker_max
        n_far, sec_far = int(far.sum()), float(dur[far].sum())

        def reject(reason: str, **extra) -> None:
            bits = " ".join(f"{k}={v}" for k, v in extra.items())
            log.info(
                "candidate_new_speaker REJECTED (%s): far_windows=%d far_sec=%.1f "
                "known_speakers=%d %s",
                reason,
                n_far,
                sec_far,
                len(cents),
                bits,
            )

        if n_far == 0:
            return None

        # The strongest evidence available about this candidate, computed once
        # so it can be logged whichever way the decision goes.
        Xf, wf = X[far], dur[far]
        cand = l2norm((Xf * (wf / wf.sum())[:, None]).sum(axis=0))
        agree = (
            cdist(Xf, cand[None, :], metric="cosine").ravel() <= self.same_speaker_max
        )
        n_agree = int(agree.sum())
        sec_agree = float(wf[agree].sum())
        if n_agree:
            Xa, wa = Xf[agree], wf[agree]
            cand = l2norm((Xa * (wa / wa.sum())[:, None]).sum(axis=0))
            radius = float(
                (cdist(Xa, cand[None, :], metric="cosine").ravel() * wa).sum()
                / max(wa.sum(), 1e-9)
            )
        else:
            radius = 1.0

        d_to_known = cdist(cand[None, :], cents, metric="cosine").ravel()
        d_min_known = float(d_to_known.min())
        dists = ", ".join(
            f"S{i + 1}={d:.3f}" for i, d in enumerate(np.round(d_to_known, 3))
        )

        # The normal path (plenty of audio) and the short path (little audio,
        # but unmistakably somebody else and unmistakably self-consistent).
        normal = (
            n_agree >= self.new_identity_min_windows
            and sec_agree >= self.new_identity_min_sec
        )
        short = (
            n_agree >= self.new_identity_short_windows
            and sec_agree >= self.new_identity_short_sec
            and d_min_known >= self.new_identity_strong_dist
        )

        if not (normal or short):
            reject(
                "insufficient evidence",
                agree_windows=n_agree,
                agree_sec=round(sec_agree, 1),
                need=f"{self.new_identity_min_windows}w/"
                f"{self.new_identity_min_sec:.1f}s "
                f"or {self.new_identity_short_windows}w/"
                f"{self.new_identity_short_sec:.1f}s at d>="
                f"{self.new_identity_strong_dist:.2f}",
                distances=dists,
                internal_radius=round(radius, 3),
            )
            return None

        if radius > NEW_IDENTITY_MAX_RADIUS:
            # Unlike everybody INCLUDING ITSELF: music, applause, a jingle, or
            # a run of turn-boundary windows. This is the guard that keeps the
            # short path from manufacturing phantom speakers.
            reject(
                "not internally consistent",
                internal_radius=round(radius, 3),
                limit=NEW_IDENTITY_MAX_RADIUS,
                agree_windows=n_agree,
                agree_sec=round(sec_agree, 1),
                distances=dists,
            )
            return None

        if d_min_known < self.same_speaker_max:
            reject(
                "too close to a known speaker after refinement",
                distances=dists,
                nearest=round(d_min_known, 3),
                limit=self.same_speaker_max,
            )
            return None

        log.info(
            "candidate_new_speaker ACCEPTED (%s path): windows=%d duration=%.1fs "
            "internal_radius=%.3f distances=[%s] -> Speaker %d",
            "normal" if normal else "short",
            n_agree,
            sec_agree,
            radius,
            dists,
            len(cents) + 1,
        )
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
            if d1 > OUTLIER_MAX:
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
        self._since_growth_sec = 0.0
        self._established = False


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
