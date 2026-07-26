"""
speaker_clustering.py — session-global speaker clustering (v7).

PURE numpy/scipy. No torch, no pyannote. This is the part that actually
decides "who is talking", and it is deliberately importable and unit-testable
on synthetic embeddings.

WHY THIS REPLACES THE v5/v6 APPROACH
------------------------------------
The old design assigned a speaker the moment a segment arrived, by comparing
its embedding to a running centroid and thresholding the cosine distance
(dist <= 0.55 -> match, else new speaker). That fails for three structural
reasons, none of which are fixable by moving the threshold:

  1. ONE BAD FIRST EMBEDDING POISONS A CENTROID FOR THE WHOLE SESSION.
     A running mean has no way to recover: every later comparison is made
     against a corrupted anchor, so errors compound instead of averaging out.

  2. EMBEDDING DISTANCE IS DURATION-DEPENDENT.
     Speaker embeddings from <1s of audio are dominated by phonetic content,
     not voice identity. Same-speaker distances on 0.6s clips routinely exceed
     cross-speaker distances on 3s clips. A single global threshold therefore
     cannot be correct for both, which is what produces "9 speakers for 2
     people": short segments spawn phantoms.

  3. A GREEDY ONLINE DECISION IS NEVER REVISITED.
     Offline diarization is accurate because it clusters ALL the evidence at
     once. Online-greedy throws that away.

THE FIX: two tiers.

  TIER 1 (instant, provisional) — nearest-centroid assignment so the UI has a
  label immediately. Conservative: when the speaker count K is known it can
  NEVER invent a speaker, and short turns can never create one.

  TIER 2 (background, authoritative) — periodically re-cluster EVERY embedding
  in the session with agglomerative clustering, duration-weighted, with small
  clusters pruned and reassigned. The result overwrites the provisional labels
  and is pushed to the UI as a `refresh` message. Accuracy converges toward
  offline quality as the session goes on, instead of degrading.

Label churn is prevented by Hungarian matching each new clustering against the
previous one, so "Speaker 1" stays the same human across refreshes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist

log = logging.getLogger("voxlive.cluster")

# Below this, an embedding is treated as evidence-poor: it still receives a
# label, but it never creates a speaker and never updates a centroid.
DEFAULT_RELIABLE_SEC = 1.0


def l2norm(v: np.ndarray) -> np.ndarray:
    """L2-normalize so that cosine distance == 0.5 * squared euclidean."""
    v = np.asarray(v, dtype=np.float64).ravel()
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


@dataclass
class Turn:
    """One contiguous stretch of speech by one voice, in session time."""

    turn_id: int
    start: float
    end: float
    embedding: np.ndarray  # L2-normalized
    speaker: int = -1  # stable speaker id (0-based)
    provisional: bool = True  # True until a global recluster has seen it
    meta: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def reliable(self) -> bool:
        return self.duration >= self.meta.get("reliable_sec", DEFAULT_RELIABLE_SEC)


class SpeakerClusterer:
    """Session-global speaker assignment with periodic self-correction.

    Typical use from the audio pipeline::

        c = SpeakerClusterer(expected_speakers=2)
        turn = c.add(embedding, start=12.4, end=15.1)   # instant label
        if c.should_recluster():
            if c.recluster():                            # labels changed
                emit_refresh(c.turns)
    """

    def __init__(
        self,
        expected_speakers: int = 0,
        max_speakers: int = 6,
        match_threshold: float = 0.55,
        new_speaker_margin: float = 0.15,
        min_new_speaker_sec: float = 1.5,
        min_cluster_sec: float = 3.0,
        reliable_sec: float = DEFAULT_RELIABLE_SEC,
        recluster_every_turns: int = 6,
        smooth_max_sec: float = 1.2,
        smooth_margin: float = 0.10,
    ):
        self.expected_speakers = max(0, int(expected_speakers))
        self.max_speakers = max(1, int(max_speakers))
        self.match_threshold = float(match_threshold)
        self.new_speaker_margin = float(new_speaker_margin)
        self.min_new_speaker_sec = float(min_new_speaker_sec)
        self.min_cluster_sec = float(min_cluster_sec)
        self.reliable_sec = float(reliable_sec)
        self.recluster_every_turns = max(1, int(recluster_every_turns))
        self.smooth_max_sec = float(smooth_max_sec)
        self.smooth_margin = float(smooth_margin)

        self.turns: list[Turn] = []
        self._centroids: dict[int, np.ndarray] = {}
        self._weights: dict[int, float] = {}
        self._next_speaker = 0
        self._turns_since_recluster = 0

    # ---------------- capacity ----------------

    @property
    def _cap(self) -> int:
        """Hard ceiling on how many identities may exist."""
        return (
            self.expected_speakers if self.expected_speakers > 0 else self.max_speakers
        )

    def n_speakers(self) -> int:
        return len({t.speaker for t in self.turns if t.speaker >= 0})

    # ---------------- tier 1: instant provisional label ----------------

    def add(self, embedding, start: float, end: float, **meta) -> Turn:
        emb = l2norm(embedding)
        turn = Turn(
            turn_id=len(self.turns),
            start=float(start),
            end=float(end),
            embedding=emb,
            meta={"reliable_sec": self.reliable_sec, **meta},
        )
        turn.speaker = self._assign_provisional(turn)
        self.turns.append(turn)
        self._turns_since_recluster += 1
        return turn

    def _assign_provisional(self, turn: Turn) -> int:
        if not self._centroids:
            return self._create_speaker(turn)

        ids = sorted(self._centroids)
        mat = np.stack([self._centroids[i] for i in ids])
        dists = cdist(turn.embedding[None, :], mat, metric="cosine")[0]
        best_i = int(np.argmin(dists))
        best_id, best_dist = ids[best_i], float(dists[best_i])

        # A new identity requires ALL of: room under the cap, enough audio to
        # trust the embedding, and a distance clearly beyond the match radius.
        room = len(self._centroids) < self._cap
        long_enough = turn.duration >= self.min_new_speaker_sec
        far_enough = best_dist > self.match_threshold + self.new_speaker_margin

        if room and long_enough and far_enough:
            log.debug(
                "new speaker: dur=%.2fs dist=%.3f > %.3f",
                turn.duration,
                best_dist,
                self.match_threshold + self.new_speaker_margin,
            )
            return self._create_speaker(turn)

        if turn.reliable:
            self._update_centroid(best_id, turn)
        log.debug(
            "assign: dur=%.2fs dist=%.3f -> Speaker %d%s",
            turn.duration,
            best_dist,
            best_id + 1,
            "" if turn.reliable else " (short, centroid not updated)",
        )
        return best_id

    def _create_speaker(self, turn: Turn) -> int:
        sid = self._next_speaker
        self._next_speaker += 1
        self._centroids[sid] = turn.embedding.copy()
        self._weights[sid] = turn.duration if turn.reliable else 0.0
        return sid

    def _update_centroid(self, sid: int, turn: Turn) -> None:
        """Duration-weighted running mean. Longer turns move the anchor more."""
        w_old = self._weights.get(sid, 0.0)
        w_new = turn.duration
        total = w_old + w_new
        if total <= 0:
            return
        blended = (self._centroids[sid] * w_old + turn.embedding * w_new) / total
        self._centroids[sid] = l2norm(blended)
        self._weights[sid] = total

    # ---------------- tier 2: global self-correction ----------------

    def should_recluster(self) -> bool:
        return (
            self._turns_since_recluster >= self.recluster_every_turns
            and len(self.turns) >= 2
        )

    def recluster(self) -> bool:
        """Re-derive every label from all evidence. True if anything changed."""
        self._turns_since_recluster = 0
        if len(self.turns) < 2:
            return False

        old = [t.speaker for t in self.turns]

        anchors = [t for t in self.turns if t.reliable]
        if len(anchors) < 2:
            return False  # not enough trustworthy audio to cluster on yet

        X = np.stack([t.embedding for t in anchors])
        labels = self._cluster(X, anchors)
        labels = self._prune_and_reassign(labels, anchors)

        centroids = self._centroids_from(labels, anchors)
        if not centroids:
            return False

        # Short/unreliable turns did not vote; they now inherit the nearest
        # centroid derived from turns we DO trust.
        full = self._assign_all(centroids)
        full = self._smooth(full, centroids)
        full = self._stabilize(full, old)

        for t, sid in zip(self.turns, full):
            t.speaker = int(sid)
            t.provisional = False

        self._rebuild_state(centroids, full)
        changed = full != old
        if changed:
            n_moved = sum(1 for a, b in zip(old, full) if a != b)
            log.info(
                "recluster: %d/%d turn(s) relabelled, %d speaker(s)",
                n_moved,
                len(full),
                len(set(full)),
            )
        return changed

    def _cluster(self, X: np.ndarray, anchors: list[Turn]) -> np.ndarray:
        """Agglomerative clustering. Average linkage on cosine distance."""
        Z = linkage(pdist(X, metric="cosine"), method="average")
        if self.expected_speakers > 0:
            k = min(self.expected_speakers, len(anchors))
            return fcluster(Z, k, criterion="maxclust") - 1
        lab = fcluster(Z, self.match_threshold, criterion="distance") - 1
        # Respect the ceiling even in auto mode.
        if len(set(lab)) > self.max_speakers:
            lab = fcluster(Z, self.max_speakers, criterion="maxclust") - 1
        return lab

    def _prune_and_reassign(
        self, labels: np.ndarray, anchors: list[Turn]
    ) -> np.ndarray:
        """Kill phantom speakers: clusters holding too little total speech.

        A real participant accumulates seconds. A clustering artefact holds one
        or two short turns. When the speaker count is known we never prune,
        because the cluster count is already pinned to K.
        """
        if self.expected_speakers > 0:
            return labels
        labels = labels.copy()
        totals: dict[int, float] = {}
        for lab, t in zip(labels, anchors):
            totals[int(lab)] = totals.get(int(lab), 0.0) + t.duration
        survivors = [c for c, d in totals.items() if d >= self.min_cluster_sec]
        if not survivors:
            survivors = [max(totals, key=totals.get)]
        if len(survivors) == len(totals):
            return labels

        keep = np.stack(
            [
                l2norm(
                    np.mean(
                        [
                            anchors[i].embedding
                            for i in range(len(anchors))
                            if labels[i] == c
                        ],
                        axis=0,
                    )
                )
                for c in survivors
            ]
        )
        for i, lab in enumerate(labels):
            if int(lab) in survivors:
                continue
            d = cdist(anchors[i].embedding[None, :], keep, metric="cosine")[0]
            labels[i] = survivors[int(np.argmin(d))]
            log.debug("pruned phantom cluster %d -> %d", lab, labels[i])
        return labels

    def _centroids_from(self, labels, anchors) -> dict[int, np.ndarray]:
        out: dict[int, np.ndarray] = {}
        for c in sorted(set(int(x) for x in labels)):
            members = [
                (anchors[i].embedding, anchors[i].duration)
                for i in range(len(anchors))
                if int(labels[i]) == c
            ]
            if not members:
                continue
            vecs = np.stack([m[0] for m in members])
            w = np.array([m[1] for m in members], dtype=np.float64)
            w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1 / len(w))
            out[c] = l2norm((vecs * w[:, None]).sum(axis=0))
        return out

    def _assign_all(self, centroids: dict[int, np.ndarray]) -> list[int]:
        ids = sorted(centroids)
        mat = np.stack([centroids[i] for i in ids])
        X = np.stack([t.embedding for t in self.turns])
        d = cdist(X, mat, metric="cosine")
        return [ids[int(i)] for i in np.argmin(d, axis=1)]

    def _smooth(self, labels: list[int], centroids: dict[int, np.ndarray]) -> list[int]:
        """Temporal prior: conversation is sticky.

        A 0.5s island of Speaker 2 wedged between two Speaker 1 turns is far
        more likely to be an embedding error than a real half-second
        interjection. Flip it only when the acoustic evidence was marginal
        anyway (the two candidate distances were within `smooth_margin`).
        """
        if len(labels) < 3:
            return labels
        ids = sorted(centroids)
        mat = np.stack([centroids[i] for i in ids])
        out = list(labels)
        for i in range(1, len(labels) - 1):
            prev_l, cur_l, next_l = out[i - 1], labels[i], labels[i + 1]
            if prev_l != next_l or cur_l == prev_l:
                continue
            if self.turns[i].duration > self.smooth_max_sec:
                continue
            d = cdist(self.turns[i].embedding[None, :], mat, metric="cosine")[0]
            d_cur = d[ids.index(cur_l)]
            d_nbr = d[ids.index(prev_l)]
            if d_nbr - d_cur <= self.smooth_margin:
                out[i] = prev_l
                log.debug("smoothed turn %d: %d -> %d", i, cur_l, prev_l)
        return out

    def _stabilize(self, new: list[int], old: list[int]) -> list[int]:
        """Keep display names stable across refreshes.

        Clustering ids are arbitrary; without this, a recluster could rename
        every participant and the transcript would look scrambled even when
        the grouping improved. Hungarian-match new clusters to the previous
        labelling by shared speech duration.
        """
        new_ids = sorted(set(new))
        old_ids = sorted({o for o in old if o >= 0})
        if not old_ids:
            return new

        cost = np.zeros((len(new_ids), len(old_ids)))
        for i, n in enumerate(new_ids):
            for j, o in enumerate(old_ids):
                shared = sum(
                    self.turns[k].duration
                    for k in range(len(new))
                    if new[k] == n and old[k] == o
                )
                cost[i, j] = -shared

        rows, cols = linear_sum_assignment(cost)
        mapping = {new_ids[r]: old_ids[c] for r, c in zip(rows, cols) if cost[r, c] < 0}
        nxt = max(old_ids) + 1
        for n in new_ids:
            if n not in mapping:
                mapping[n] = nxt
                nxt += 1
        self._next_speaker = max(self._next_speaker, nxt)
        return [mapping[n] for n in new]

    def _rebuild_state(self, centroids, labels) -> None:
        """Re-anchor the online centroids on the corrected grouping.

        This is what stops errors compounding: tier 1 restarts from a clean
        anchor after every tier-2 pass.
        """
        self._centroids, self._weights = {}, {}
        for sid in sorted(set(labels)):
            members = [
                (t.embedding, t.duration)
                for t, l in zip(self.turns, labels)
                if l == sid and t.reliable
            ]
            if not members:
                continue
            vecs = np.stack([m[0] for m in members])
            w = np.array([m[1] for m in members], dtype=np.float64)
            wn = w / w.sum() if w.sum() > 0 else np.full(len(w), 1 / len(w))
            self._centroids[sid] = l2norm((vecs * wn[:, None]).sum(axis=0))
            self._weights[sid] = float(w.sum())
        self._next_speaker = max(
            self._next_speaker, max(self._centroids, default=-1) + 1
        )

    # ---------------- output ----------------

    def speaker_name(self, sid: int) -> str:
        return f"Speaker {sid + 1}"

    def reset(self) -> None:
        self.turns.clear()
        self._centroids.clear()
        self._weights.clear()
        self._next_speaker = 0
        self._turns_since_recluster = 0
