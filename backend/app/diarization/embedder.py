"""
embedder.py — batched speaker-embedding extraction.

WHY THIS FILE EXISTS
--------------------
v7 called pyannote's `Inference(model, window="whole")` once per window, in a
Python loop. For a 6-second VAD segment that is 12-20 sequential forward passes
of a ResNet34 on CPU, each carrying full framework overhead: 1-4 seconds of
latency on the critical path, per segment.

The model itself is not the problem. ResNet34 on 1 second of 16 kHz audio is
tiny. The problem is that a batch of 20 windows costs barely more than a batch
of 1, and v7 paid for 20 batches of 1.

This module does three things v7 did not:

  1. BATCHES. Every window of a pass goes through the network in ONE call.
  2. CACHES BY AUDIO IDENTITY. A window's embedding depends only on its samples,
     so re-diarizing a rolling window never re-embeds audio it already saw.
  3. REFUSES SHORT AUDIO. Below `MIN_EMBED_SEC` a speaker embedding encodes
     phonetic content, not voice identity — same-speaker distance on 0.5s clips
     routinely exceeds cross-speaker distance on 3s clips. That single fact is
     the root cause of "9 speakers for 2 people". v7's floor was 0.25s. Here it
     is 0.90s, and the clustering layer only *trusts* windows above 1.5s.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from typing import Optional

import numpy as np

log = logging.getLogger("voxlive.embed")

EMBED_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"

# Hard floor. Anything shorter is not a voice fingerprint, it is a phoneme.
MIN_EMBED_SEC = 0.90

# Max windows per forward pass. Keeps peak memory bounded on small containers.
MAX_BATCH = 64

# v13: windows are grouped into buckets this wide and every member is TRIMMED to
# the shortest in its bucket, so a forward pass never contains zero padding.
#
# WHY THIS MATTERS FOR SPEAKER SEPARATION
# ---------------------------------------
# WeSpeaker pools statistics over the whole time axis. Zero padding is not
# neutral to that pool — it drags every padded embedding toward the same
# "mostly silence" direction, and that direction is shared by every speaker.
# The effect is to shrink the distance between different people, which is
# precisely the failure being fixed elsewhere in this change.
#
# The previous code sorted by length and padded each batch to its longest
# member, which bounds the damage but does not remove it: `slice_windows`
# emits a short tail window at the end of every speech region, so a batch
# routinely mixes a 0.95 s window with 2.00 s ones and pads the short one with
# 52% zeros. Trimming 0.10 s off the long end of a >=0.90 s window costs a
# little speech; padding it with a second of silence costs its identity.
LENGTH_BUCKET_SEC = 0.10

# Cap on remembered embeddings per session. The service re-presents a WIN_SEC
# tail of audio on every pass so the same window really is embedded twice;
# 4096 entries is roughly an hour of 2 s windows at 0.75 s hop.
CACHE_MAX = 4096

_MODEL_CACHE: dict = {}
_LOAD_LOCK = threading.Lock()


def _load(hf_token: str, device: Optional[str] = None):
    """Load (once, process-wide) the raw torch module — not pyannote's
    Inference wrapper, which cannot batch."""
    import torch
    from pyannote.audio import Model

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    key = (EMBED_MODEL, dev)
    with _LOAD_LOCK:
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]

        # On CPU, torch defaults to spawning a thread per core and then fights
        # itself: for batches this small, intra-op parallelism costs more in
        # synchronisation than it saves. Two threads is consistently fastest.
        if dev == "cpu":
            torch.set_num_threads(min(4, max(1, (torch.get_num_threads() or 4) // 2)))

        log.info("loading %s on %s ...", EMBED_MODEL, dev)
        model = Model.from_pretrained(EMBED_MODEL, use_auth_token=hf_token)
        model.eval().to(torch.device(dev))
        _MODEL_CACHE[key] = (model, torch.device(dev))
        log.info("embedder ready (%s, shared across all sessions)", dev)
        return _MODEL_CACHE[key]


def warmup(hf_token: str, device: Optional[str] = None) -> None:
    """Load the model at startup so the first user never waits for it."""
    import torch

    model, dev = _load(hf_token, device)
    # A real forward pass, not just a load: the first inference triggers lazy
    # kernel selection / autotuning that would otherwise hit the first user.
    with torch.inference_mode():
        model(torch.zeros(2, 1, 16000, device=dev))
    log.info("embedder warm")


class Embedder:
    """Stateless w.r.t. speakers; holds only a per-session memo of embeddings.

    One instance per WebSocket session. The heavy weights are shared; this
    object owns nothing but a small dict.
    """

    def __init__(
        self, hf_token: str, device: Optional[str] = None, sample_rate: int = 16000
    ):
        self.hf_token = hf_token
        self.device = device
        self.sample_rate = sample_rate
        # Keyed by the CONTENT of the window, so identical audio presented
        # twice is embedded once. The module docstring has always claimed this
        # cache exists; until v13 it was allocated, cleared by `reset()`, and
        # never read or written by `embed_batch`.
        self._cache: dict[bytes, np.ndarray] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self._model = None
        self._dev = None

    @staticmethod
    def _key(w: np.ndarray) -> bytes:
        return hashlib.blake2b(
            np.ascontiguousarray(w, dtype=np.float32).tobytes(), digest_size=16
        ).digest()

    def _ensure(self):
        if self._model is None:
            self._model, self._dev = _load(self.hf_token, self.device)

    def embed_batch(self, windows: list[np.ndarray]) -> list[Optional[np.ndarray]]:
        """Embed many float32 mono waveforms in as few forward passes as
        possible. Returns one L2-normalised vector (or None) per input, in
        order. None means "too short to be meaningful" — the caller must not
        substitute a zero vector, because a zero vector would silently become
        a cluster member.
        """
        out: list[Optional[np.ndarray]] = [None] * len(windows)
        todo: list[int] = []
        keys: dict[int, bytes] = {}
        min_len = int(MIN_EMBED_SEC * self.sample_rate)

        for i, w in enumerate(windows):
            if w is None or len(w) < min_len:
                continue
            k = self._key(w)
            hit = self._cache.get(k)
            if hit is not None:
                out[i] = hit
                self.cache_hits += 1
                continue
            keys[i] = k
            todo.append(i)
            self.cache_misses += 1

        if not todo:
            return out

        import torch

        self._ensure()

        # Group into length buckets and TRIM each bucket to its shortest
        # member, so no forward pass ever contains a padded row. See
        # LENGTH_BUCKET_SEC for why padding is not neutral here.
        bucket = max(1, int(LENGTH_BUCKET_SEC * self.sample_rate))
        groups: dict[int, list[int]] = {}
        for i in todo:
            groups.setdefault(len(windows[i]) // bucket, []).append(i)

        batches: list[list[int]] = []
        for _, members in sorted(groups.items()):
            for b0 in range(0, len(members), MAX_BATCH):
                batches.append(members[b0 : b0 + MAX_BATCH])

        for idxs in batches:
            width = min(len(windows[i]) for i in idxs)
            if width < min_len:
                continue
            batch = np.empty((len(idxs), 1, width), dtype=np.float32)
            for r, i in enumerate(idxs):
                batch[r, 0, :] = windows[i][:width]

            try:
                with torch.inference_mode():
                    t = torch.from_numpy(batch).to(self._dev)
                    emb = self._model(t)
                    emb = emb.detach().float().cpu().numpy()
            except Exception as exc:  # noqa: BLE001 — never kill the stream
                log.warning("embedding batch failed (%d windows): %s", len(idxs), exc)
                continue

            emb = np.atleast_2d(np.asarray(emb, dtype=np.float64))
            for r, i in enumerate(idxs):
                v = emb[r].ravel()
                n = float(np.linalg.norm(v))
                vec = v / n if n > 1e-9 else None
                out[i] = vec
                if vec is not None and len(self._cache) < CACHE_MAX:
                    self._cache[keys[i]] = vec

        return out

    def embed(self, window: np.ndarray) -> Optional[np.ndarray]:
        return self.embed_batch([window])[0]

    def reset(self) -> None:
        self._cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
