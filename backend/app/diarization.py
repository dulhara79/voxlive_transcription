"""
Speaker diarization / identification — pyannote.audio edition (v2).

WHAT CHANGED vs v1 (and why everything was "Speaker 1"):
  1. NO MORE DRIFTING CENTROID. v1 kept one running-mean embedding per
     speaker; after a few segments the mean absorbs a second voice and every
     new voice lands within threshold of it -> everyone becomes Speaker 1.
     v2 stores UP TO 10 RAW EMBEDDINGS PER SPEAKER and compares each new
     segment to the closest stored one (single-linkage). Speakers stay
     distinct.
  2. THRESHOLD LOWERED 0.60 -> 0.45. On short live segments, 0.60 merges
     almost any two voices recorded through the same mic/playback chain.
  3. EVERY ASSIGNMENT IS LOGGED with its distance, e.g.
         diarize: dur=3.2s best=Speaker 1 dist=0.512 thr=0.45 -> NEW Speaker 2
     Watch the log while two people speak: same-speaker distances should sit
     BELOW the threshold, cross-speaker distances ABOVE it. Set
     DIARIZATION_THRESHOLD between those two bands. THIS TUNING STEP IS NOT
     OPTIONAL — the right value depends on your mics and room.
  4. A new speaker is only *created* from a segment >= 1.0s (configurable).
     Short blips are too unreliable to justify spawning "Speaker 7".

IMPORTANT TESTING CAVEAT: playing two recordings through the SAME laptop
speakers into the SAME laptop mic makes both voices share one playback
channel — the channel fingerprint can dominate the speaker fingerprint and
squeeze cross-speaker distances toward zero. That setup is the WORST CASE for
any diarizer, commercial ones included. Validate with two real people
speaking directly into the mic (with pauses between turns), or by streaming
the clean audio files into the pipeline directly.

Modes (DIARIZATION_MODE):
  off       -> Diarizer             everyone is "Speaker 1"
  pyannote  -> PyannoteDiarizer     "Speaker 1..N", N capped at MAX_SPEAKERS
  identify  -> IdentifyingDiarizer  enrolled names + Speaker-N fallback

SETUP (once):
  pip install pyannote.audio torch torchaudio
  1. Token: https://hf.co/settings/tokens
  2. Accept gated-model conditions: https://hf.co/pyannote/embedding
  3. .env: HUGGINGFACE_TOKEN=hf_...  DIARIZATION_MODE=pyannote

HARD LIMITS (architectural):
  - One label PER SEGMENT: overlapping speech / crosstalk within one VAD
    segment gets one label. Turn-taking with ~500ms pauses works.
  - Segments < 0.5s reuse the previous speaker's label (embeddings that short
    are noise).
"""

import asyncio
import glob
import logging
import os

log = logging.getLogger("voxlive.diarize")

# ---- module-level caches (shared across connections) ----
_INFERENCE: dict = {}  # device -> pyannote Inference
_VOICEPRINTS: dict = {}  # abspath(voiceprints_dir) -> (names, embeddings)

_MIN_EMBED_SEC = 0.5  # below this, embeddings are noise -> sticky label
_MAX_EMBS_PER_SPEAKER = 10  # raw embeddings kept per speaker (no mean drift)


def _get_inference(device: str, hf_token: str | None):
    """Load pyannote/embedding once per device and share it (inference-only)."""
    if device not in _INFERENCE:
        import torch
        from pyannote.audio import Inference, Model

        log.info("loading pyannote/embedding (device=%s)…", device)
        try:
            model = Model.from_pretrained(
                "pyannote/embedding", use_auth_token=hf_token or None
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "Could not load pyannote/embedding (GATED model). Fix: "
                "(1) token at hf.co/settings/tokens, (2) accept conditions at "
                "hf.co/pyannote/embedding, (3) HUGGINGFACE_TOKEN in .env."
            ) from e
        _INFERENCE[device] = Inference(
            model, window="whole", device=torch.device(device)
        )
    return _INFERENCE[device]


class Diarizer:
    """No-op diarizer: everything is Speaker 1."""

    async def assign(self, pcm_bytes: bytes, sample_rate: int) -> str:
        return "Speaker 1"


class PyannoteDiarizer(Diarizer):
    """
    Online speaker assignment via pyannote/embedding + single-linkage cosine
    matching against stored raw embeddings (no centroid averaging).

    Instantiate ONE PER CONNECTION: speaker memory is session state.
    """

    def __init__(
        self,
        threshold: float = 0.45,
        max_speakers: int = 10,
        device: str = "cpu",
        hf_token: str | None = None,
        min_new_speaker_sec: float = 1.0,
    ):
        import numpy as np  # local imports so the base app doesn't need torch
        import torch

        self.np = np
        self.torch = torch
        self.threshold = threshold  # cosine DISTANCE; below it = same speaker
        self.max_speakers = max(1, max_speakers)
        self.min_new_speaker_sec = min_new_speaker_sec
        self.inference = _get_inference(device, hf_token)  # shared, cached
        # speakers[i] = list of raw normalized embeddings for "Speaker i+1"
        self.speakers: list[list] = []
        self._last_label: str | None = None  # sticky label for micro-segments

    async def assign(self, pcm_bytes: bytes, sample_rate: int) -> str:
        return await asyncio.to_thread(self._assign, pcm_bytes, sample_rate)

    # ---- embedding helpers ----

    def _embed(self, pcm_bytes: bytes, sample_rate: int):
        np, torch = self.np, self.torch
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        wav = torch.from_numpy(audio).unsqueeze(0)  # (1, time)
        emb = self.inference({"waveform": wav, "sample_rate": sample_rate})
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        return emb / (np.linalg.norm(emb) + 1e-9)

    def _embed_file(self, path: str):
        """Embed an enrollment WAV from disk (any rate/channels -> 16k mono)."""
        import torchaudio

        wav, sr = torchaudio.load(path)  # (channels, samples)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
            sr = 16000
        np = self.np
        emb = self.inference({"waveform": wav, "sample_rate": sr})
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        return emb / (np.linalg.norm(emb) + 1e-9)

    # ---- single-linkage matching (no mean drift) ----

    def _closest_speaker(self, emb):
        """Distance of emb to each speaker = min cosine distance to ANY of
        that speaker's stored embeddings. Returns (best_index, best_distance)."""
        np = self.np
        best_i, best_dist = -1, 1e9
        for i, embs in enumerate(self.speakers):
            for ref in embs:
                dist = 1.0 - float(np.dot(emb, ref))
                if dist < best_dist:
                    best_i, best_dist = i, dist
        return best_i, best_dist

    def _remember(self, idx: int, emb):
        embs = self.speakers[idx]
        embs.append(emb)
        if len(embs) > _MAX_EMBS_PER_SPEAKER:
            embs.pop(0)  # keep the most recent K

    def _assign_from_emb(self, emb, dur: float) -> str:
        best_i, best_dist = self._closest_speaker(emb)

        if best_i == -1:  # very first segment of the session
            self.speakers.append([emb])
            label = "Speaker 1"
            log.info("diarize: dur=%.1fs first segment -> %s", dur, label)
            self._last_label = label
            return label

        if best_dist <= self.threshold:
            self._remember(best_i, emb)
            label = f"Speaker {best_i + 1}"
            log.info(
                "diarize: dur=%.1fs best=Speaker %d dist=%.3f thr=%.2f -> %s",
                dur,
                best_i + 1,
                best_dist,
                self.threshold,
                label,
            )
            self._last_label = label
            return label

        # Above threshold -> looks like a NEW voice.
        if len(self.speakers) < self.max_speakers and dur >= self.min_new_speaker_sec:
            self.speakers.append([emb])
            label = f"Speaker {len(self.speakers)}"
            log.info(
                "diarize: dur=%.1fs best=Speaker %d dist=%.3f thr=%.2f -> NEW %s",
                dur,
                best_i + 1,
                best_dist,
                self.threshold,
                label,
            )
            self._last_label = label
            return label

        # Can't create (cap reached, or segment too short to trust) -> nearest.
        self._remember(best_i, emb)
        label = f"Speaker {best_i + 1}"
        log.info(
            "diarize: dur=%.1fs dist=%.3f > thr=%.2f but %s -> %s (forced)",
            dur,
            best_dist,
            self.threshold,
            "cap reached" if len(self.speakers) >= self.max_speakers else "too short",
            label,
        )
        self._last_label = label
        return label

    def _assign(self, pcm_bytes: bytes, sample_rate: int) -> str:
        dur = len(pcm_bytes) / 2 / sample_rate
        if dur < _MIN_EMBED_SEC and self._last_label:
            return self._last_label
        return self._assign_from_emb(self._embed(pcm_bytes, sample_rate), dur)


class IdentifyingDiarizer(PyannoteDiarizer):
    """
    Speaker IDENTIFICATION against enrolled voiceprints, with Speaker-N
    fallback (capped clustering) for anyone not enrolled.

    ENROLLMENT — a folder of reference clips per person:
        voiceprints/
          Dulhara/              clip1.wav  clip2.wav
          Prof_Thelijjagoda/    intro.wav
    Folder names become the displayed identity ("_" -> " "). Give each person
    a few CLEAN clips of ~5-10s recorded on the SAME kind of mic you'll use
    live — enrolling from studio audio but running on a laptop mic is a
    channel mismatch and identification accuracy drops hard.

    Every comparison is logged with its distance so id_threshold can be tuned
    the same way as the cluster threshold.
    """

    def __init__(
        self,
        voiceprints_dir: str = "voiceprints",
        id_threshold: float = 0.45,  # cosine DISTANCE to accept an identity
        cluster_threshold: float = 0.45,
        max_speakers: int = 10,
        device: str = "cpu",
        hf_token: str | None = None,
        min_new_speaker_sec: float = 1.0,
    ):
        super().__init__(
            threshold=cluster_threshold,
            max_speakers=max_speakers,
            device=device,
            hf_token=hf_token,
            min_new_speaker_sec=min_new_speaker_sec,
        )
        self.id_threshold = id_threshold
        self.names, self.voiceprints = self._get_voiceprints(voiceprints_dir)

    def _get_voiceprints(self, root: str):
        key = os.path.abspath(root)
        if key not in _VOICEPRINTS:
            _VOICEPRINTS[key] = self._load_voiceprints(root)
        return _VOICEPRINTS[key]

    def _load_voiceprints(self, root: str):
        np = self.np
        names: list = []
        prints: list = []

        if not os.path.isdir(root):
            log.warning(
                "voiceprints dir %r not found — ID mode with 0 enrolled "
                "speakers (everyone will be 'Speaker N')",
                root,
            )
            return names, prints

        for person in sorted(os.listdir(root)):
            pdir = os.path.join(root, person)
            if not os.path.isdir(pdir):
                continue
            clips = sorted(glob.glob(os.path.join(pdir, "*.wav")))
            embs = []
            for c in clips:
                try:
                    embs.append(self._embed_file(c))
                except Exception as e:  # noqa: BLE001
                    log.error("could not embed enrollment clip %s: %s", c, e)
            if not embs:
                continue
            mean = np.mean(embs, axis=0)
            mean = mean / (np.linalg.norm(mean) + 1e-9)
            prints.append(mean)
            names.append(person.replace("_", " "))
            log.info("enrolled %r from %d clip(s)", person, len(embs))

        if not names:
            log.warning("no voiceprints loaded under %r", root)
        return names, prints

    async def assign(self, pcm_bytes: bytes, sample_rate: int) -> str:
        return await asyncio.to_thread(self._identify, pcm_bytes, sample_rate)

    def _identify(self, pcm_bytes: bytes, sample_rate: int) -> str:
        np = self.np

        dur = len(pcm_bytes) / 2 / sample_rate
        if dur < _MIN_EMBED_SEC and self._last_label:
            return self._last_label

        emb = self._embed(pcm_bytes, sample_rate)

        best_name, best_dist = None, 1e9
        for name, ref in zip(self.names, self.voiceprints):
            dist = 1.0 - float(np.dot(emb, ref))
            if dist < best_dist:
                best_name, best_dist = name, dist

        if best_name is not None and best_dist <= self.id_threshold:
            log.info(
                "identify: %s dist=%.3f thr=%.2f",
                best_name,
                best_dist,
                self.id_threshold,
            )
            self._last_label = best_name
            return best_name

        label = self._assign_from_emb(emb, dur)
        log.info(
            "identify: no enrolled match (closest=%s dist=%.3f) -> %s",
            best_name,
            best_dist if best_name else float("nan"),
            label,
        )
        return label
