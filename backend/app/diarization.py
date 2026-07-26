# """
# Speaker diarization / identification — pyannote.audio edition (v3).

# WHAT CHANGED vs v2 (and why 2 people became 9 speakers):

#   1. DUAL THRESHOLDS (match vs create). v2 had ONE threshold doing two jobs:
#      "is this the same speaker?" AND "is this a new speaker?". On broadcast
#      audio (music beds, compression) same-speaker distances jitter a lot, so
#      a single cutoff either merges people or spawns phantom speakers.
#      v3 splits it:
#          dist <= match_thr                     -> same speaker (confident)
#          match_thr < dist <= new_thr           -> nearest speaker, but the
#                                                   embedding is NOT stored
#                                                   (ambiguous zone)
#          dist > new_thr                        -> genuinely new voice
#      new_thr = match_thr + DIARIZATION_NEW_SPEAKER_MARGIN (default +0.15).

#   2. ANTI-CHAINING STORE MARGIN. v2 stored every matched embedding. A
#      borderline embedding at dist=0.44 stored into Speaker 1 can later act as
#      a BRIDGE to a genuinely different voice (single-linkage chaining) — the
#      "two people identified as the same speaker" bug. v3 only stores an
#      embedding when dist <= match_thr - 0.05; borderline matches are labeled
#      but never become reference points.

#   3. AUTOMATIC MERGE PASS. v2 could create a spurious "Speaker 7" from one
#      noisy segment and keep it forever. v3 checks (after every store/create)
#      whether any two clusters sit within match_thr AVERAGE-linkage distance
#      of each other; if so, the smaller (by accumulated speech time) merges
#      into the larger and its label disappears from FUTURE output.
#      NOTE: paragraphs already sent to the client keep their old labels — the
#      merge only stops further proliferation.

#   4. NEW SPEAKERS NEED MORE EVIDENCE. min_new_speaker_sec default is now
#      2.0s (was 1.0). One noisy second of audio is not enough evidence to
#      invent a person.

# TUNING IS STILL NOT OPTIONAL. Watch the "diarize:" log lines during a real
# two-person test: same-speaker distances must sit BELOW match_thr,
# cross-speaker distances ABOVE new_thr. Set DIARIZATION_THRESHOLD in the gap.

# BROADCAST-AUDIO CAVEAT: a TV/YouTube programme mixes speech with music beds,
# jingles and audience noise. Music under speech shifts embeddings — that alone
# caused most of the v2 phantom speakers. If you know the speaker count
# (e.g., a 2-person interview), set MAX_SPEAKERS to it: with the ambiguous-zone
# rule, forced assignments no longer pollute clusters, so capping is now safe.

# Modes (DIARIZATION_MODE):
#   off       -> Diarizer             everyone is "Speaker 1"
#   pyannote  -> PyannoteDiarizer     "Speaker 1..N", N capped at MAX_SPEAKERS
#   identify  -> IdentifyingDiarizer  enrolled names + Speaker-N fallback

# HARD LIMITS (architectural, unchanged):
#   - One label PER SEGMENT: overlapping speech / crosstalk inside one VAD
#     segment gets one label. Turn-taking with pauses works; crosstalk doesn't.
#   - Segments < 0.5s reuse the previous speaker's label.
# """

# import asyncio
# import glob
# import logging
# import os

# log = logging.getLogger("voxlive.diarize")

# # ---- module-level caches (shared across connections) ----
# _INFERENCE: dict = {}  # device -> pyannote Inference
# _VOICEPRINTS: dict = {}  # abspath(voiceprints_dir) -> (names, embeddings)

# _MIN_EMBED_SEC = 0.5  # below this, embeddings are noise -> sticky label
# _MAX_EMBS_PER_SPEAKER = 10  # raw embeddings kept per speaker
# _STORE_MARGIN = 0.05  # store an embedding only if dist <= match_thr - this
# #                       (prevents single-linkage chaining between voices)


# def _load_model_compat(name: str, hf_token: str | None):
#     """pyannote.audio renamed the auth kwarg on from_pretrained() from
#     `use_auth_token` (older releases) to `token` (newer releases). Try the
#     current name first, fall back to the old one, so this works across
#     installed versions without pinning."""
#     from pyannote.audio import Model

#     try:
#         return Model.from_pretrained(name, token=hf_token or None)
#     except TypeError:
#         return Model.from_pretrained(name, use_auth_token=hf_token or None)


# def _get_inference(device: str, hf_token: str | None):
#     """Load pyannote/embedding once per device and share it (inference-only)."""
#     if device not in _INFERENCE:
#         import torch
#         from pyannote.audio import Inference

#         log.info("loading pyannote/embedding (device=%s)…", device)
#         try:
#             model = _load_model_compat("pyannote/embedding", hf_token)
#         except Exception as e:  # noqa: BLE001
#             raise RuntimeError(
#                 "Could not load pyannote/embedding (GATED model). Fix: "
#                 "(1) token at hf.co/settings/tokens, (2) accept conditions at "
#                 "hf.co/pyannote/embedding, (3) HUGGINGFACE_TOKEN in .env."
#             ) from e
#         _INFERENCE[device] = Inference(
#             model, window="whole", device=torch.device(device)
#         )
#     return _INFERENCE[device]


# class Diarizer:
#     """No-op diarizer: everything is Speaker 1."""

#     async def assign(self, pcm_bytes: bytes, sample_rate: int) -> str:
#         return "Speaker 1"


# class PyannoteDiarizer(Diarizer):
#     """
#     Online speaker assignment via pyannote/embedding.

#     Clusters are lists of raw normalized embeddings. Assignment uses
#     single-linkage (min distance to any stored embedding); merging uses
#     average-linkage (robust against chaining). See module docstring for the
#     dual-threshold / store-margin / merge design.

#     Instantiate ONE PER CONNECTION: speaker memory is session state.
#     """

#     def __init__(
#         self,
#         threshold: float = 0.45,
#         max_speakers: int = 10,
#         device: str = "cpu",
#         hf_token: str | None = None,
#         min_new_speaker_sec: float = 2.0,
#         new_speaker_margin: float = 0.15,
#     ):
#         import numpy as np  # local imports so the base app doesn't need torch
#         import torch

#         self.np = np
#         self.torch = torch
#         self.match_thr = threshold  # cosine DISTANCE; <= means same speaker
#         self.new_thr = threshold + max(0.0, new_speaker_margin)
#         self.max_speakers = max(1, max_speakers)
#         self.min_new_speaker_sec = min_new_speaker_sec
#         self.inference = _get_inference(device, hf_token)  # shared, cached

#         # Each cluster: {"embs": [np.ndarray], "label": int, "dur": float}
#         # "label" is the display number ("Speaker <label>") and never changes;
#         # "dur" is accumulated speech seconds (used to pick the survivor of a
#         # merge — the voice we've heard more of keeps its name).
#         self.clusters: list[dict] = []
#         self.next_label = 1
#         self._last_label: str | None = None  # sticky label for micro-segments

#     async def assign(self, pcm_bytes: bytes, sample_rate: int) -> str:
#         return await asyncio.to_thread(self._assign, pcm_bytes, sample_rate)

#     # ---- embedding helpers ----

#     def _embed(self, pcm_bytes: bytes, sample_rate: int):
#         np, torch = self.np, self.torch
#         audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
#         wav = torch.from_numpy(audio).unsqueeze(0)  # (1, time)
#         emb = self.inference({"waveform": wav, "sample_rate": sample_rate})
#         emb = np.asarray(emb, dtype=np.float32).reshape(-1)
#         return emb / (np.linalg.norm(emb) + 1e-9)

#     def _embed_file(self, path: str):
#         """Embed an enrollment WAV from disk (any rate/channels -> 16k mono)."""
#         import torchaudio

#         wav, sr = torchaudio.load(path)  # (channels, samples)
#         if wav.shape[0] > 1:
#             wav = wav.mean(dim=0, keepdim=True)
#         if sr != 16000:
#             wav = torchaudio.functional.resample(wav, sr, 16000)
#             sr = 16000
#         np = self.np
#         emb = self.inference({"waveform": wav, "sample_rate": sr})
#         emb = np.asarray(emb, dtype=np.float32).reshape(-1)
#         return emb / (np.linalg.norm(emb) + 1e-9)

#     # ---- cluster maths ----

#     def _closest_cluster(self, emb):
#         """Single-linkage: min cosine distance from emb to ANY stored
#         embedding of each cluster. Returns (best_index, best_distance)."""
#         np = self.np
#         best_i, best = -1, 1e9
#         for i, c in enumerate(self.clusters):
#             for ref in c["embs"]:
#                 d = 1.0 - float(np.dot(emb, ref))
#                 if d < best:
#                     best_i, best = i, d
#         return best_i, best

#     def _avg_linkage(self, a: dict, b: dict) -> float:
#         """Average cosine distance across all cross pairs of two clusters.
#         Robust against chaining (unlike single-linkage) — used ONLY for the
#         merge decision."""
#         np = self.np
#         dists = [1.0 - float(np.dot(x, y)) for x in a["embs"] for y in b["embs"]]
#         return sum(dists) / len(dists)

#     def _maybe_merge(self):
#         """If any two clusters are within match_thr AVERAGE-linkage distance,
#         they are the same voice split in two — merge the smaller (by speech
#         time) into the larger. Repeats until stable."""
#         changed = True
#         while changed and len(self.clusters) > 1:
#             changed = False
#             n = len(self.clusters)
#             for i in range(n):
#                 for j in range(i + 1, n):
#                     d = self._avg_linkage(self.clusters[i], self.clusters[j])
#                     if d <= self.match_thr:
#                         keep, drop = (
#                             (i, j)
#                             if self.clusters[i]["dur"] >= self.clusters[j]["dur"]
#                             else (j, i)
#                         )
#                         kc, dc = self.clusters[keep], self.clusters[drop]
#                         log.info(
#                             "diarize: MERGE Speaker %d -> Speaker %d "
#                             "(avg dist=%.3f <= thr=%.2f); future segments use "
#                             "Speaker %d",
#                             dc["label"],
#                             kc["label"],
#                             d,
#                             self.match_thr,
#                             kc["label"],
#                         )
#                         half = _MAX_EMBS_PER_SPEAKER // 2
#                         kc["embs"] = (kc["embs"][-half:] + dc["embs"][-half:])[
#                             -_MAX_EMBS_PER_SPEAKER:
#                         ]
#                         kc["dur"] += dc["dur"]
#                         del self.clusters[drop]
#                         changed = True
#                         break
#                 if changed:
#                     break

#     def _store(self, cluster: dict, emb, dur: float):
#         cluster["embs"].append(emb)
#         if len(cluster["embs"]) > _MAX_EMBS_PER_SPEAKER:
#             cluster["embs"].pop(0)  # keep the most recent K
#         cluster["dur"] += dur

#     def _label_of(self, emb, fallback: str) -> str:
#         """After a merge, the cluster we stored into may have been absorbed;
#         locate emb by identity to report the SURVIVING label."""
#         for c in self.clusters:
#             if any(ref is emb for ref in c["embs"]):
#                 return f"Speaker {c['label']}"
#         return fallback

#     # ---- assignment ----

#     def _assign_from_emb(self, emb, dur: float) -> str:
#         # First segment of the session.
#         if not self.clusters:
#             self.clusters.append({"embs": [emb], "label": self.next_label, "dur": dur})
#             label = f"Speaker {self.next_label}"
#             self.next_label += 1
#             log.info("diarize: dur=%.1fs first segment -> %s", dur, label)
#             self._last_label = label
#             return label

#         best_i, best = self._closest_cluster(emb)
#         c = self.clusters[best_i]

#         # CONFIDENT MATCH.
#         if best <= self.match_thr:
#             label = f"Speaker {c['label']}"
#             if best <= self.match_thr - _STORE_MARGIN:
#                 # Only clearly-inside matches become reference embeddings —
#                 # borderline ones would enable single-linkage chaining.
#                 self._store(c, emb, dur)
#                 self._maybe_merge()
#                 label = self._label_of(emb, label)
#             log.info(
#                 "diarize: dur=%.1fs dist=%.3f <= match=%.2f -> %s%s",
#                 dur,
#                 best,
#                 self.match_thr,
#                 label,
#                 (
#                     ""
#                     if best <= self.match_thr - _STORE_MARGIN
#                     else " (borderline, not stored)"
#                 ),
#             )
#             self._last_label = label
#             return label

#         # AMBIGUOUS ZONE, or new-speaker creation not allowed -> nearest,
#         # WITHOUT storing (never pollute a cluster with an uncertain voice).
#         can_create = (
#             best > self.new_thr
#             and dur >= self.min_new_speaker_sec
#             and len(self.clusters) < self.max_speakers
#         )
#         if not can_create:
#             label = f"Speaker {c['label']}"
#             reason = (
#                 "ambiguous zone"
#                 if best <= self.new_thr
#                 else (
#                     "cap reached"
#                     if len(self.clusters) >= self.max_speakers
#                     else "segment too short to create"
#                 )
#             )
#             log.info(
#                 "diarize: dur=%.1fs dist=%.3f (match=%.2f / new=%.2f) -> %s "
#                 "(%s, not stored)",
#                 dur,
#                 best,
#                 self.match_thr,
#                 self.new_thr,
#                 label,
#                 reason,
#             )
#             self._last_label = label
#             return label

#         # GENUINELY NEW VOICE.
#         self.clusters.append({"embs": [emb], "label": self.next_label, "dur": dur})
#         label = f"Speaker {self.next_label}"
#         self.next_label += 1
#         log.info(
#             "diarize: dur=%.1fs dist=%.3f > new=%.2f -> NEW %s",
#             dur,
#             best,
#             self.new_thr,
#             label,
#         )
#         self._last_label = label
#         return label

#     def _assign(self, pcm_bytes: bytes, sample_rate: int) -> str:
#         dur = len(pcm_bytes) / 2 / sample_rate
#         if dur < _MIN_EMBED_SEC and self._last_label:
#             return self._last_label
#         return self._assign_from_emb(self._embed(pcm_bytes, sample_rate), dur)


# class IdentifyingDiarizer(PyannoteDiarizer):
#     """
#     Speaker IDENTIFICATION against enrolled voiceprints, with Speaker-N
#     fallback (capped clustering) for anyone not enrolled.

#     ENROLLMENT — a folder of reference clips per person:
#         voiceprints/
#           Dulhara/              clip1.wav  clip2.wav
#           Prof_Thelijjagoda/    intro.wav
#     Folder names become the displayed identity ("_" -> " "). Give each person
#     a few CLEAN clips of ~5-10s recorded on the SAME kind of mic you'll use
#     live — enrolling from studio audio but running on a laptop mic is a
#     channel mismatch and identification accuracy drops hard.

#     Every comparison is logged with its distance so id_threshold can be tuned
#     the same way as the cluster threshold.
#     """

#     def __init__(
#         self,
#         voiceprints_dir: str = "voiceprints",
#         id_threshold: float = 0.45,  # cosine DISTANCE to accept an identity
#         cluster_threshold: float = 0.45,
#         max_speakers: int = 10,
#         device: str = "cpu",
#         hf_token: str | None = None,
#         min_new_speaker_sec: float = 2.0,
#         new_speaker_margin: float = 0.15,
#     ):
#         super().__init__(
#             threshold=cluster_threshold,
#             max_speakers=max_speakers,
#             device=device,
#             hf_token=hf_token,
#             min_new_speaker_sec=min_new_speaker_sec,
#             new_speaker_margin=new_speaker_margin,
#         )
#         self.id_threshold = id_threshold
#         self.names, self.voiceprints = self._get_voiceprints(voiceprints_dir)

#     def _get_voiceprints(self, root: str):
#         key = os.path.abspath(root)
#         if key not in _VOICEPRINTS:
#             _VOICEPRINTS[key] = self._load_voiceprints(root)
#         return _VOICEPRINTS[key]

#     def _load_voiceprints(self, root: str):
#         np = self.np
#         names: list = []
#         prints: list = []

#         if not os.path.isdir(root):
#             log.warning(
#                 "voiceprints dir %r not found — ID mode with 0 enrolled "
#                 "speakers (everyone will be 'Speaker N')",
#                 root,
#             )
#             return names, prints

#         for person in sorted(os.listdir(root)):
#             pdir = os.path.join(root, person)
#             if not os.path.isdir(pdir):
#                 continue
#             clips = sorted(glob.glob(os.path.join(pdir, "*.wav")))
#             embs = []
#             for c in clips:
#                 try:
#                     embs.append(self._embed_file(c))
#                 except Exception as e:  # noqa: BLE001
#                     log.error("could not embed enrollment clip %s: %s", c, e)
#             if not embs:
#                 continue
#             mean = np.mean(embs, axis=0)
#             mean = mean / (np.linalg.norm(mean) + 1e-9)
#             prints.append(mean)
#             names.append(person.replace("_", " "))
#             log.info("enrolled %r from %d clip(s)", person, len(embs))

#         if not names:
#             log.warning("no voiceprints loaded under %r", root)
#         return names, prints

#     async def assign(self, pcm_bytes: bytes, sample_rate: int) -> str:
#         return await asyncio.to_thread(self._identify, pcm_bytes, sample_rate)

#     def _identify(self, pcm_bytes: bytes, sample_rate: int) -> str:
#         np = self.np

#         dur = len(pcm_bytes) / 2 / sample_rate
#         if dur < _MIN_EMBED_SEC and self._last_label:
#             return self._last_label

#         emb = self._embed(pcm_bytes, sample_rate)

#         best_name, best_dist = None, 1e9
#         for name, ref in zip(self.names, self.voiceprints):
#             dist = 1.0 - float(np.dot(emb, ref))
#             if dist < best_dist:
#                 best_name, best_dist = name, dist

#         if best_name is not None and best_dist <= self.id_threshold:
#             log.info(
#                 "identify: %s dist=%.3f thr=%.2f",
#                 best_name,
#                 best_dist,
#                 self.id_threshold,
#             )
#             self._last_label = best_name
#             return best_name

#         label = self._assign_from_emb(emb, dur)
#         log.info(
#             "identify: no enrolled match (closest=%s dist=%.3f) -> %s",
#             best_name,
#             best_dist if best_name else float("nan"),
#             label,
#         )
#         return label
