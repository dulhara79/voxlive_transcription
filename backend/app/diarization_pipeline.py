# """
# Pipeline diarizer (v7) — automatic speaker-count detection, production grade.

# WHY v6 STILL MISLABELED IN PRACTICE: v6 fixed the assignment mechanics
# (joint Hungarian matching, weighted centroids, global re-clustering with
# retroactive correction) but the SPEAKER COUNT was still decided by fixed
# cosine-distance cutoffs (0.55 match / 0.70 create, and a fixed-threshold
# dendrogram cut in the recluster). On clean audio WeSpeaker distances are
# bimodal and any cutoff works; on real broadcast/code-switched audio with
# short turns and music beds, same-speaker distances drift into 0.5-0.7 —
# exactly where the cutoffs sit. Result: phantom splits or wrong merges no
# matter how the threshold is tuned.

# v7 REMOVES THE THRESHOLD FROM THE COUNT DECISION ENTIRELY:

#   1. MODEL-SELECTED SPEAKER COUNT. The periodic global re-cluster now tries
#      every candidate count k = 1..cap on the average-linkage dendrogram and
#      picks the k that maximises a DURATION-WEIGHTED SILHOUETTE score (how
#      well-separated the clustering is, judged by the session's OWN distance
#      distribution — no absolute cutoff involved). Below a minimum
#      separation score the answer is "one speaker". A small parsimony margin
#      prefers fewer speakers on near-ties.

#   2. PHANTOM ABSORPTION BY MASS. Any cluster carrying less than
#      _MIN_CLUSTER_W seconds of total voice evidence is dissolved into its
#      nearest surviving cluster. Phantom speakers are, by construction, low
#      mass — this removes them mechanically instead of hoping a threshold
#      excludes them.

#   3. ADAPTIVE ONLINE THRESHOLDS. After each re-cluster with >=2 speakers,
#      the online match threshold is re-derived from the measured
#      intra-cluster spread vs. the closest inter-centroid distance of THIS
#      session's audio (clamped to [0.35, 0.75]). The hardcoded 0.55 is only
#      the starting point; the system calibrates itself to the microphone,
#      codec and speakers actually present.

#   4. FASTER EARLY CONVERGENCE. Re-clustering runs every 2 segments until
#      the bank has enough evidence (24 embeddings), then every 4. Combined
#      with the retroactive "refresh" push (main.py), the on-screen labels
#      converge to offline-quality within the first ~30-60 seconds.

#   5. CONSERVATIVE ONLINE CREATION. Live assignment still creates new
#      identities when a voice is clearly far from every known one, but the
#      re-cluster is now the AUTHORITY on how many speakers exist — online
#      mistakes in either direction are corrected retroactively.

#   EXPECTED_SPEAKERS is now OPTIONAL. If set (>0) it acts as a hard cap and
#   candidate counts run 1..K (a session where only some of the K spoke so
#   far is still clustered correctly). Unset (0), the count is detected
#   automatically up to MAX_SPEAKERS.

# Carried over from v6: joint Hungarian per-segment assignment, duration-
# weighted unit centroids, multi-span embedding averaging, temporal-continuity
# prior, exclusive (non-overlapping) turn detection via
# pyannote/speaker-diarization-community-1, WeSpeaker identity embeddings,
# session embedding bank + {emb_id: new_label} retroactive remap contract.

# REQUIRED SETUP (unchanged): pyannote.audio>=4.0, scipy>=1.11, one-time
# acceptance of BOTH gated HF repos while logged in:
#     https://hf.co/pyannote/speaker-diarization-community-1
#     https://hf.co/pyannote/wespeaker-voxceleb-resnet34-LM
# .env: HUGGINGFACE_TOKEN=hf_...   DIARIZATION_MODE=pipeline
#       EXPECTED_SPEAKERS=0        (auto-detect; >0 = hard cap)

# API (unchanged from v6 — main.py needs no edits):
#     diarize_spans(pcm, sr) -> list[(start_s, end_s, label:int, emb_id|None)]
#     maybe_recluster()      -> dict[emb_id, new_label] | None
# """

# import asyncio
# import logging
# from collections import defaultdict

# import numpy as np
# from scipy.cluster.hierarchy import fcluster, linkage
# from scipy.optimize import linear_sum_assignment
# from scipy.spatial.distance import pdist, squareform

# log = logging.getLogger("voxlive.diarize.pipeline")

# _PIPELINES: dict = {}  # device -> community-1 Pipeline (shared)
# _EMBED_INFERENCE: dict = {}  # device -> Inference wrapping the WeSpeaker model

# _MIN_EMBED_SEC = 0.5  # spans shorter than this produce unreliable embeddings
# _MIN_CREATE_SEC = 1.0  # min speech before a voice may CREATE a new identity
# _MERGE_GAP_SEC = 0.4  # adjacent same-speaker spans closer than this merge
# _MAX_SPANS_PER_EMB = 3  # average the embeddings of up to N longest spans
# _CENTROID_MAX_W = 5.0  # cap one segment's influence on a centroid (seconds)
# _SLOT_MAX_W = 60.0  # cap a slot's accumulated inertia
# _CONTINUITY_BONUS = 0.04  # cost discount for "same speaker kept talking"
# _SEG_MAX_SPEAKERS = 4  # per-segment (<=10s) active-speaker ceiling for the
# #                        turn-detection pipeline; the SESSION count is
# #                        governed by the re-cluster, not by this.
# _BANK_MAX = 240  # session embedding bank size (oldest dropped)
# _RECLUSTER_EVERY = 4  # steady-state cadence (segments)
# _RECLUSTER_EVERY_EARLY = 2  # cadence while evidence is still thin
# _EARLY_BANK = 24  # bank size below which the early cadence applies
# _RECLUSTER_MIN_EMBS = 4
# _MIN_SIL = 0.24  # weighted-silhouette floor: below this = one speaker.
# #                  Set empirically: spurious splits of single-voice noisy
# #                  banks score <=~0.22; genuine two-voice splits score
# #                  >=~0.27 even when same-speaker distances drift to ~0.6.
# _K_PREFER_MARGIN = 0.02  # parsimony: a larger k must beat by this margin
# _MIN_CLUSTER_W = 1.6  # clusters with less voice evidence (s) are absorbed
# _MIN_CLUSTER_FRAC = 0.05  # ...or less than this share of total evidence
# _THR_LO, _THR_HI = 0.35, 0.75  # clamp for the adaptive match threshold
# _BIG = 10.0  # "impossible" assignment cost


# def _from_pretrained_compat(cls, name: str, hf_token: str | None, **kw):
#     """pyannote.audio renamed from_pretrained()'s auth kwarg from
#     `use_auth_token` (<=3.x) to `token` (4.x). Try current name first."""
#     try:
#         return cls.from_pretrained(name, token=hf_token or None, **kw)
#     except TypeError:
#         return cls.from_pretrained(name, use_auth_token=hf_token or None, **kw)


# def _get_pipeline(device: str, hf_token: str | None):
#     if device not in _PIPELINES:
#         import torch
#         from pyannote.audio import Pipeline

#         log.info(
#             "loading pyannote/speaker-diarization-community-1 (device=%s)…", device
#         )
#         try:
#             pipe = _from_pretrained_compat(
#                 Pipeline, "pyannote/speaker-diarization-community-1", hf_token
#             )
#         except Exception as e:  # noqa: BLE001
#             raise RuntimeError(
#                 "Could not load pyannote/speaker-diarization-community-1. "
#                 "Fix: (1) HUGGINGFACE_TOKEN in .env, (2) accept the "
#                 "conditions of BOTH gated models (while logged in): "
#                 "hf.co/pyannote/speaker-diarization-community-1 AND "
#                 "hf.co/pyannote/wespeaker-voxceleb-resnet34-LM, (3) "
#                 "pyannote.audio>=4.0 (pip install -U pyannote.audio)."
#             ) from e
#         pipe.to(torch.device(device))
#         _PIPELINES[device] = pipe
#     return _PIPELINES[device]


# def _get_embed_inference(device: str, hf_token: str | None):
#     if device not in _EMBED_INFERENCE:
#         import torch
#         from pyannote.audio import Inference, Model

#         log.info(
#             "loading pyannote/wespeaker-voxceleb-resnet34-LM (device=%s)…", device
#         )
#         try:
#             model = _from_pretrained_compat(
#                 Model, "pyannote/wespeaker-voxceleb-resnet34-LM", hf_token
#             )
#         except Exception as e:  # noqa: BLE001
#             raise RuntimeError(
#                 "Could not load pyannote/wespeaker-voxceleb-resnet34-LM. "
#                 "If this model page shows a user-conditions gate, accept it "
#                 "once while logged in to Hugging Face, then retry."
#             ) from e
#         _EMBED_INFERENCE[device] = Inference(
#             model, window="whole", device=torch.device(device)
#         )
#     return _EMBED_INFERENCE[device]


# def _weighted_silhouette(D, labels, w) -> float:
#     """Duration-weighted silhouette score of a labeling over a distance
#     matrix. Judges cluster separation by THIS session's own distance
#     distribution — no absolute threshold. Singletons contribute 0 (neutral),
#     so a lone noisy embedding can't fake 'perfect separation'."""
#     n = len(labels)
#     uniq = sorted(set(labels))
#     if len(uniq) < 2:
#         return -1.0
#     members = {c: [j for j in range(n) if labels[j] == c] for c in uniq}
#     s_sum, w_sum = 0.0, 0.0
#     for i in range(n):
#         own = [j for j in members[labels[i]] if j != i]
#         w_sum += w[i]
#         if not own:
#             continue  # singleton -> 0 contribution
#         wa = sum(w[j] for j in own)
#         a = sum(D[i, j] * w[j] for j in own) / wa
#         b = _BIG
#         for c in uniq:
#             if c == labels[i]:
#                 continue
#             wb = sum(w[j] for j in members[c])
#             if wb <= 0:
#                 continue
#             b = min(b, sum(D[i, j] * w[j] for j in members[c]) / wb)
#         m = max(a, b)
#         s_sum += ((b - a) / m if m > 0 else 0.0) * w[i]
#     return s_sum / w_sum if w_sum > 0 else -1.0


# class PipelineDiarizer:
#     """
#     Per-connection global speaker registry + per-segment turn detection,
#     with automatic speaker-count detection via periodic global re-clustering.

#     Instantiate ONE PER CONNECTION: speaker memory is session state.
#     """

#     def __init__(
#         self,
#         expected_speakers: int = 0,
#         max_speakers: int = 10,
#         threshold: float = 0.55,
#         new_speaker_margin: float = 0.15,
#         device: str = "cpu",
#         hf_token: str | None = None,
#     ):
#         import torch

#         self.torch = torch
#         self.K = expected_speakers if expected_speakers > 0 else 0
#         self.cap = self.K if self.K else max(1, max_speakers)
#         self.match_thr = threshold  # STARTING point; adapts after recluster
#         self.margin = max(0.0, new_speaker_margin)
#         self.new_thr = threshold + self.margin
#         self.pipeline = _get_pipeline(device, hf_token)
#         self.embed = _get_embed_inference(device, hf_token)

#         # slots[i] = {"label": int, "centroid": unit np.float32 vec, "w": float}
#         self.slots: list[dict] = []
#         # bank[i] = {"id": int, "emb": vec, "w": float(dur-cap), "label": int}
#         self.bank: list[dict] = []
#         self._next_emb_id = 1
#         self._last_label = 1  # sticky label for unembeddable segments
#         self._last_emb_id: int | None = None
#         self._segs_since_recluster = 0
#         self._bank_dirty = False

#     # ------------------------------------------------------------------ API

#     async def diarize_spans(self, pcm_bytes: bytes, sample_rate: int):
#         return await asyncio.to_thread(self._diarize, pcm_bytes, sample_rate)

#     def maybe_recluster(self):
#         """Call once per diarized segment (from a thread — CPU work).
#         Re-clusters the whole session bank on a cadence (fast while evidence
#         is thin, slower once stable) and rebuilds slots/centroids/count.
#         Returns {emb_id: new_label} for chunks whose speaker changed."""
#         self._segs_since_recluster += 1
#         every = (
#             _RECLUSTER_EVERY_EARLY if len(self.bank) < _EARLY_BANK else _RECLUSTER_EVERY
#         )
#         if self._segs_since_recluster < every:
#             return None
#         if len(self.bank) < _RECLUSTER_MIN_EMBS or not self._bank_dirty:
#             return None
#         self._segs_since_recluster = 0
#         self._bank_dirty = False
#         return self._recluster()

#     # ------------------------------------------------- embedding utilities

#     def _embed_span(self, audio, sr: int, s: float, e: float):
#         """WeSpeaker embedding for one span. Unit-norm np.float32 or None."""
#         if e - s < _MIN_EMBED_SEC:
#             return None
#         clip = audio[int(s * sr) : int(e * sr)]
#         if len(clip) < int(_MIN_EMBED_SEC * sr):
#             return None
#         wav = self.torch.from_numpy(clip).unsqueeze(0)
#         vec = self.embed({"waveform": wav, "sample_rate": sr})
#         vec = np.asarray(vec, dtype=np.float32).reshape(-1)
#         n = float(np.linalg.norm(vec))
#         if not np.isfinite(n) or n < 1e-6:
#             return None
#         return vec / n

#     def _embed_local(self, audio, sr: int, spans):
#         """Average the embeddings of up to _MAX_SPANS_PER_EMB longest usable
#         spans of one local speaker (duration-weighted) — lower variance than
#         a single-span embedding."""
#         usable = sorted(spans, key=lambda t: t[1] - t[0], reverse=True)
#         acc, tot = None, 0.0
#         for s, e in usable[:_MAX_SPANS_PER_EMB]:
#             v = self._embed_span(audio, sr, s, e)
#             if v is None:
#                 continue
#             w = e - s
#             acc = v * w if acc is None else acc + v * w
#             tot += w
#         if acc is None:
#             return None
#         n = float(np.linalg.norm(acc))
#         return (acc / n).astype(np.float32) if n > 1e-9 else None

#     def _bank_add(self, emb, w: float, label: int) -> int:
#         eid = self._next_emb_id
#         self._next_emb_id += 1
#         self.bank.append(
#             {"id": eid, "emb": emb.astype(np.float32), "w": float(w), "label": int(label)}
#         )
#         if len(self.bank) > _BANK_MAX:
#             self.bank.pop(0)
#         self._bank_dirty = True
#         return eid

#     def _mint_label(self) -> int:
#         used = {s["label"] for s in self.slots}
#         lab = 1
#         while lab in used:
#             lab += 1
#         return lab

#     # -------------------------------------------- joint (Hungarian) matching

#     def _assign_joint(self, locals_):
#         """locals_: [(local_lab, emb, dur, first_start)], all with embeddings.
#         Returns {local_lab: (global_label, emb_id)}. Assigns all local
#         speakers to global slots simultaneously (min total cost), with
#         'create new speaker' as explicitly priced pseudo-slots."""
#         n_loc = len(locals_)
#         n_slot = len(self.slots)
#         n_new = min(n_loc, max(0, self.cap - n_slot))

#         cost = np.full((n_loc, n_slot + max(n_new, 0)), _BIG, dtype=np.float64)
#         dists = np.full((n_loc, max(n_slot, 1)), _BIG, dtype=np.float64)
#         earliest = min(range(n_loc), key=lambda i: locals_[i][3]) if n_loc else -1

#         for i, (lab, emb, dur, _fs) in enumerate(locals_):
#             for j, slot in enumerate(self.slots):
#                 d = 1.0 - float(np.dot(emb, slot["centroid"]))
#                 dists[i, j] = d
#                 c = d
#                 # Conversation prior: the temporally-first voice of a segment
#                 # is usually whoever was talking across the VAD split.
#                 if i == earliest and slot["label"] == self._last_label:
#                     c -= _CONTINUITY_BONUS
#                 cost[i, j] = c
#             # Creating a new identity: while slots < K (known count) creation
#             # is EXPECTED and only needs to beat the match threshold; in auto
#             # mode it must beat match+margin. Either way the re-cluster is
#             # the authority — online mistakes are corrected retroactively.
#             create_cost = self.match_thr if self.K else self.new_thr
#             if dur < _MIN_CREATE_SEC:
#                 create_cost = _BIG  # too little speech to found an identity
#             for k in range(n_new):
#                 cost[i, n_slot + k] = create_cost

#         out: dict = {}
#         assigned_rows: set = set()
#         if cost.shape[1] > 0:
#             rows, cols = linear_sum_assignment(cost)
#         else:
#             rows, cols = [], []

#         for i, j in zip(rows, cols):
#             lab, emb, dur, _fs = locals_[i]
#             w = min(dur, _CENTROID_MAX_W)
#             assigned_rows.add(i)

#             if j < n_slot and cost[i, j] < _BIG:
#                 slot = self.slots[j]
#                 d = dists[i, j]
#                 if d <= self.match_thr:
#                     c = slot["centroid"] * slot["w"] + emb * w
#                     n = float(np.linalg.norm(c))
#                     if n > 1e-9:
#                         slot["centroid"] = (c / n).astype(np.float32)
#                     slot["w"] = min(slot["w"] + w, _SLOT_MAX_W)
#                     tag = "match"
#                 else:
#                     tag = "forced"  # centroid untouched; recluster may fix
#                 eid = self._bank_add(emb, w, slot["label"])
#                 log.info(
#                     "diarize: dur=%.1fs dist=%.3f (thr=%.2f) -> Speaker %d (%s)",
#                     dur, d, self.match_thr, slot["label"], tag,
#                 )
#                 out[lab] = (slot["label"], eid)

#             elif j >= n_slot and cost[i, j] < _BIG:
#                 label = self._mint_label()
#                 self.slots.append({"label": label, "centroid": emb.copy(), "w": w})
#                 eid = self._bank_add(emb, w, label)
#                 log.info(
#                     "diarize: dur=%.1fs -> NEW Speaker %d (%d/%d slots)",
#                     dur, label, len(self.slots), self.cap,
#                 )
#                 out[lab] = (label, eid)

#             else:
#                 out[lab] = self._fallback_assign(lab, emb, dur, dists, i)

#         # More local voices than available columns (cap reached): force each
#         # leftover onto its nearest slot. Bank it anyway — recluster can fix.
#         for i in range(n_loc):
#             if i not in assigned_rows:
#                 lab, emb, dur, _fs = locals_[i]
#                 out[lab] = self._fallback_assign(lab, emb, dur, dists, i)
#         return out

#     def _fallback_assign(self, lab, emb, dur, dists, i):
#         if self.slots:
#             j = int(np.argmin(dists[i]))
#             label = self.slots[j]["label"]
#             log.info(
#                 "diarize: dur=%.1fs dist=%.3f -> Speaker %d (forced, cap/short)",
#                 dur, dists[i, j], label,
#             )
#         else:
#             label = self._last_label
#         eid = self._bank_add(emb, min(dur, _CENTROID_MAX_W), label)
#         return (label, eid)

#     # -------------------------------------------------- global re-clustering

#     def _select_count(self, Z, D, w) -> np.ndarray:
#         """Choose the speaker count by model selection: try k = 2..cap cuts
#         of the dendrogram, score each with the duration-weighted silhouette,
#         prefer fewer speakers on near-ties, and fall back to ONE speaker when
#         even the best split isn't meaningfully separated."""
#         n = D.shape[0]
#         best_k, best_s, best_labels = 1, -1.0, np.ones(n, dtype=int)
#         for k in range(2, min(self.cap, n) + 1):
#             labels = fcluster(Z, t=k, criterion="maxclust")
#             s = _weighted_silhouette(D, labels, w)
#             if s > best_s + _K_PREFER_MARGIN:
#                 best_k, best_s, best_labels = k, s, labels
#         if best_k > 1 and best_s < _MIN_SIL:
#             log.info(
#                 "recluster: best split k=%d scored %.3f < %.2f -> one speaker",
#                 best_k, best_s, _MIN_SIL,
#             )
#             return np.ones(n, dtype=int)
#         if best_k > 1:
#             log.info("recluster: selected k=%d (silhouette=%.3f)", best_k, best_s)
#         return best_labels

#     def _absorb_phantoms(self, labels, X, w) -> np.ndarray:
#         """Dissolve clusters carrying < _MIN_CLUSTER_W seconds of voice
#         evidence into their nearest surviving cluster. Phantom speakers are
#         low-mass by construction — this removes them mechanically."""
#         labels = labels.copy()
#         floor = max(_MIN_CLUSTER_W, _MIN_CLUSTER_FRAC * float(np.sum(w)))
#         while True:
#             uniq = sorted(set(labels))
#             if len(uniq) <= 1:
#                 return labels
#             mass = {c: sum(w[i] for i in range(len(labels)) if labels[i] == c)
#                     for c in uniq}
#             weakest = min(uniq, key=lambda c: mass[c])
#             if mass[weakest] >= floor:
#                 return labels
#             cents = {}
#             for c in uniq:
#                 if c == weakest:
#                     continue
#                 vec = np.zeros(X.shape[1])
#                 for i in range(len(labels)):
#                     if labels[i] == c:
#                         vec += X[i] * w[i]
#                 nrm = np.linalg.norm(vec)
#                 cents[c] = vec / nrm if nrm > 1e-9 else vec
#             for i in range(len(labels)):
#                 if labels[i] == weakest:
#                     labels[i] = min(
#                         cents, key=lambda c: 1.0 - float(X[i] @ cents[c])
#                     )
#             log.info(
#                 "recluster: absorbed phantom cluster (%.1fs of evidence)",
#                 mass[weakest],
#             )

#     def _adapt_thresholds(self, labels, D, w):
#         """Re-derive the online match threshold from THIS session's measured
#         separation: place it between the intra-cluster spread and the closest
#         inter-centroid distance, clamped to [_THR_LO, _THR_HI]. The 0.55
#         default is only the cold-start value."""
#         uniq = sorted(set(labels))
#         if len(uniq) < 2:
#             return
#         intra_num, intra_den = 0.0, 0.0
#         cents = {}
#         n = len(labels)
#         for c in uniq:
#             idx = [i for i in range(n) if labels[i] == c]
#             for a_i in range(len(idx)):
#                 for b_i in range(a_i + 1, len(idx)):
#                     ww = w[idx[a_i]] * w[idx[b_i]]
#                     intra_num += D[idx[a_i], idx[b_i]] * ww
#                     intra_den += ww
#             cents[c] = idx
#         if intra_den <= 0:
#             return
#         intra = intra_num / intra_den
#         # closest pair of cluster centroids (cosine distance)
#         cent_vecs = {}
#         dim = int(self.bank[0]["emb"].shape[0])
#         for c, idx in cents.items():
#             vec = np.zeros(dim, dtype=np.float64)
#             for i in idx:
#                 vec += self.bank[i]["emb"].astype(np.float64) * w[i]
#             nrm = np.linalg.norm(vec)
#             if nrm > 1e-9:
#                 cent_vecs[c] = vec / nrm
#         inter = _BIG
#         keys = sorted(cent_vecs)
#         for a_i in range(len(keys)):
#             for b_i in range(a_i + 1, len(keys)):
#                 inter = min(
#                     inter,
#                     1.0 - float(cent_vecs[keys[a_i]] @ cent_vecs[keys[b_i]]),
#                 )
#         if inter <= intra:  # not separable enough to trust an adaptation
#             return
#         new_thr = float(np.clip(intra + 0.35 * (inter - intra), _THR_LO, _THR_HI))
#         if abs(new_thr - self.match_thr) >= 0.02:
#             log.info(
#                 "recluster: adaptive threshold %.2f -> %.2f "
#                 "(intra=%.3f, inter=%.3f)",
#                 self.match_thr, new_thr, intra, inter,
#             )
#         self.match_thr = new_thr
#         self.new_thr = new_thr + self.margin

#     def _recluster(self):
#         """Re-cluster the entire session bank from scratch: count selection
#         by silhouette, phantom absorption by mass, stable label naming,
#         centroid rebuild, adaptive threshold update. Returns {emb_id:
#         new_label} for every past chunk whose speaker changed."""
#         X = np.stack([b["emb"] for b in self.bank]).astype(np.float64)
#         w = np.array([b["w"] for b in self.bank], dtype=np.float64)

#         d = pdist(X, metric="cosine")
#         D = squareform(d)
#         Z = linkage(d, method="average")

#         labels = self._select_count(Z, D, w)
#         labels = self._absorb_phantoms(labels, X, w)
#         self._adapt_thresholds(labels, D, w)

#         clusters: dict = defaultdict(list)
#         for idx, c in enumerate(labels):
#             clusters[int(c)].append(idx)

#         # Keep on-screen names stable: each cluster claims the label its
#         # members mostly carried (weighted vote); heaviest cluster picks
#         # first; unclaimable clusters mint the lowest free label.
#         order = sorted(clusters, key=lambda c: -sum(w[i] for i in clusters[c]))
#         taken: set = set()
#         name: dict = {}
#         for c in order:
#             votes: dict = defaultdict(float)
#             for i in clusters[c]:
#                 votes[self.bank[i]["label"]] += w[i]
#             for lab, _v in sorted(votes.items(), key=lambda kv: -kv[1]):
#                 if lab not in taken:
#                     name[c] = lab
#                     taken.add(lab)
#                     break
#             else:
#                 lab = 1
#                 while lab in taken:
#                     lab += 1
#                 name[c] = lab
#                 taken.add(lab)

#         new_slots: list = []
#         remap: dict = {}
#         for c, idxs in clusters.items():
#             vec = np.zeros(X.shape[1], dtype=np.float64)
#             tw = 0.0
#             for i in idxs:
#                 vec += X[i] * w[i]
#                 tw += w[i]
#             n = float(np.linalg.norm(vec))
#             centroid = (
#                 (vec / n).astype(np.float32)
#                 if n > 1e-9
#                 else X[idxs[0]].astype(np.float32)
#             )
#             new_slots.append(
#                 {"label": name[c], "centroid": centroid, "w": float(min(tw, _SLOT_MAX_W))}
#             )
#             for i in idxs:
#                 b = self.bank[i]
#                 if b["label"] != name[c]:
#                     remap[b["id"]] = name[c]
#                     b["label"] = name[c]

#         old_n = len(self.slots)
#         self.slots = sorted(new_slots, key=lambda s: s["label"])

#         # keep the sticky label consistent with the corrected history
#         if self._last_emb_id is not None:
#             for b in self.bank:
#                 if b["id"] == self._last_emb_id:
#                     self._last_label = b["label"]
#                     break

#         if remap or len(self.slots) != old_n:
#             log.info(
#                 "recluster: %d embeddings -> %d speaker(s) (was %d), "
#                 "%d past chunk(s) relabeled",
#                 len(self.bank), len(self.slots), old_n, len(remap),
#             )
#         return remap or None

#     # ---------------------------------------------- per-segment diarization

#     def _diarize(self, pcm_bytes: bytes, sample_rate: int):
#         dur = len(pcm_bytes) / 2 / sample_rate

#         if dur < _MIN_EMBED_SEC:  # too short for reliable turn detection
#             return [(0.0, dur, self._last_label, None)]

#         audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
#         wav = self.torch.from_numpy(audio).unsqueeze(0)

#         try:
#             output = self.pipeline(
#                 {"waveform": wav, "sample_rate": sample_rate},
#                 # A <=10s clip realistically holds few ACTIVE voices; keeping
#                 # this tight stops the turn detector from over-fragmenting.
#                 # The SESSION speaker count is decided by the re-cluster.
#                 max_speakers=min(self.cap, _SEG_MAX_SPEAKERS),
#             )
#         except Exception as e:  # noqa: BLE001
#             log.error("pipeline failed on %.1fs segment: %s", dur, e)
#             return [(0.0, dur, self._last_label, None)]

#         # Non-overlapping mode: exactly one speaker active at a time — clean,
#         # disjoint spans we can slice per-speaker chunks from directly.
#         exclusive = getattr(output, "exclusive_speaker_diarization", None)
#         if exclusive is None:  # defensive fallback for older/odd builds
#             exclusive = output.speaker_diarization

#         spans_by_label: dict = defaultdict(list)
#         for turn, local_lab in exclusive:
#             spans_by_label[local_lab].append((turn.start, turn.end))

#         if not spans_by_label:  # no speech (music/noise) -> sticky label
#             return [(0.0, dur, self._last_label, None)]

#         durs = {
#             lab: sum(e - s for s, e in spans) for lab, spans in spans_by_label.items()
#         }
#         first_start = {
#             lab: min(s for s, _e in spans) for lab, spans in spans_by_label.items()
#         }

#         # One robust embedding per local speaker (multi-span average).
#         locals_ = []
#         for lab, spans in spans_by_label.items():
#             v = self._embed_local(audio, sample_rate, spans)
#             if v is not None:
#                 locals_.append((lab, v, durs[lab], first_start[lab]))
#         locals_.sort(key=lambda t: -t[2])

#         mapping: dict = {}
#         if locals_:
#             assigned = self._assign_joint(locals_)
#             for lab, (glabel, eid) in assigned.items():
#                 mapping[lab] = (glabel, eid)

#         # Local voices with no usable embedding inherit a label from this
#         # segment (or the sticky label) and carry no emb_id.
#         fallback = next(
#             (mapping[l][0] for l in spans_by_label if l in mapping), self._last_label
#         )
#         for lab in spans_by_label:
#             if lab not in mapping:
#                 mapping[lab] = (fallback, None)

#         raw = sorted(
#             (s, e, mapping[lab][0], mapping[lab][1])
#             for lab, spans in spans_by_label.items()
#             for s, e in spans
#         )
#         merged: list[list] = []
#         for s, e, glab, eid in raw:
#             if merged and merged[-1][2] == glab and s - merged[-1][1] < _MERGE_GAP_SEC:
#                 merged[-1][1] = max(merged[-1][1], e)
#                 if merged[-1][3] is None:
#                     merged[-1][3] = eid
#             else:
#                 merged.append([s, e, glab, eid])

#         self._last_label = merged[-1][2]
#         if merged[-1][3] is not None:
#             self._last_emb_id = merged[-1][3]
#         if len(merged) > 1:
#             log.info(
#                 "diarize: segment %.1fs split into %d turn(s): %s",
#                 dur,
#                 len(merged),
#                 " | ".join(f"{s:.1f}-{e:.1f}s Spk{g}" for s, e, g, _ in merged),
#             )
#         return [(s, e, g, eid) for s, e, g, eid in merged]