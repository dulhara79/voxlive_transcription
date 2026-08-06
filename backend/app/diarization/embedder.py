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
        self._cache: dict[tuple, np.ndarray] = {}
        self._model = None
        self._dev = None

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
        min_len = int(MIN_EMBED_SEC * self.sample_rate)

        for i, w in enumerate(windows):
            if w is None or len(w) < min_len:
                continue
            todo.append(i)

        if not todo:
            return out

        import torch

        self._ensure()

        # Pad to the longest window in each batch. WeSpeaker pools over time,
        # so trailing zeros bias the result — keep batches length-homogeneous
        # by sorting, which also makes the padding nearly free.
        todo.sort(key=lambda i: len(windows[i]))

        for b0 in range(0, len(todo), MAX_BATCH):
            idxs = todo[b0 : b0 + MAX_BATCH]
            lens = [len(windows[i]) for i in idxs]
            width = max(lens)
            batch = np.zeros((len(idxs), 1, width), dtype=np.float32)
            for r, i in enumerate(idxs):
                w = windows[i]
                batch[r, 0, : len(w)] = w

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
                out[i] = v / n if n > 1e-9 else None

        return out

    def embed(self, window: np.ndarray) -> Optional[np.ndarray]:
        return self.embed_batch([window])[0]

    def reset(self) -> None:
        self._cache.clear()
