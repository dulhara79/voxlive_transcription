"""
diarization_session.py — VoxLive v7 speaker diarization.

Replaces diarization.py + diarization_pipeline.py.

WHAT CHANGED, AND WHY
---------------------
v5 ran the FULL `pyannote/speaker-diarization-community-1` pipeline on every
VAD segment. That pipeline is a whole-file system: internally it does
segmentation -> embedding -> agglomerative clustering. Running it on a 3-second
chunk is wrong in two ways at once:

  * COST. It is the 1-3s of CPU per segment that config.py admits to. On a
    live stream that is the entire latency budget, spent on a model whose
    output you then throw away.
  * CORRECTNESS. Its clustering step has nothing to cluster — one short chunk
    almost always yields exactly one cluster, so the expensive part of the
    pipeline contributes nothing, and cross-segment identity still falls back
    to a separate embedding model plus a distance threshold.

v7 uses ONE model — the WeSpeaker embedder — for both jobs:

  * TURN SPLITTING inside a segment is done by change-point detection over
    sliding-window embeddings. Where a segment's voice actually changes, the
    distance between adjacent 1.5s windows spikes toward cross-speaker range
    (~1.0) while same-speaker neighbours sit near ~0.25. That gap is wide
    enough to cut on, and it costs a handful of forward passes rather than a
    full pipeline invocation.
  * IDENTITY is delegated to SpeakerClusterer, which clusters the whole
    session rather than deciding greedily turn by turn.

This also drops the dependency on `speaker-diarization-community-1`, so only
ONE gated model needs accepting on Hugging Face.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:  # inside the `app` package (production layout)
    from .speaker_clustering import SpeakerClusterer
except ImportError:  # standalone / test harness
    from speaker_clustering import SpeakerClusterer

log = logging.getLogger("voxlive.diarize")

EMBED_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"

# The embedder is STATELESS — it maps audio to a vector and nothing else. All
# per-session state lives in SpeakerClusterer. So one loaded copy is shared by
# every WebSocket connection: loading it per connection would multiply memory
# by the number of concurrent users and make each user's first utterance wait
# on a model load.
_INFERENCE_CACHE: dict = {}


def _get_inference(hf_token: str, device: Optional[str] = None):
    import torch
    from pyannote.audio import Inference, Model

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    key = (EMBED_MODEL, dev)
    if key in _INFERENCE_CACHE:
        return _INFERENCE_CACHE[key]

    log.info("loading %s (device=%s)...", EMBED_MODEL, dev)
    model = Model.from_pretrained(EMBED_MODEL, use_auth_token=hf_token)
    inf = Inference(model, window="whole", device=torch.device(dev))
    _INFERENCE_CACHE[key] = inf
    log.info("embedder ready on %s (shared across sessions)", dev)
    return inf


def warmup_shared(hf_token: str, device: Optional[str] = None) -> None:
    """Call once from the FastAPI lifespan so the first user isn't the one
    who pays for the model load."""
    _get_inference(hf_token, device)


# Sliding-window geometry for intra-segment change-point detection.
WIN_SEC = 1.5
HOP_SEC = 0.40
MIN_SPLIT_SEC = 2.2  # shorter than this, never bother looking for a turn change
MIN_TURN_SEC = 0.70  # never emit a turn shorter than this


@dataclass
class TurnOut:
    """One turn, ready to transcribe and display."""

    speaker: str  # "Speaker 1"
    speaker_id: int
    start: float  # absolute session seconds
    end: float
    pcm: bytes  # this turn's audio only
    turn_id: int


class SessionDiarizer:
    """Per-session diarizer. One instance per WebSocket connection."""

    def __init__(
        self,
        hf_token: str,
        expected_speakers: int = 0,
        max_speakers: int = 6,
        match_threshold: float = 0.55,
        new_speaker_margin: float = 0.15,
        min_new_speaker_sec: float = 1.5,
        split_threshold: float = 0.60,
        recluster_every_turns: int = 6,
        device: Optional[str] = None,
        enabled: bool = True,
    ):
        self.hf_token = hf_token
        self.split_threshold = float(split_threshold)
        self.enabled = enabled
        self._device = device
        self._inference = None

        self.clusterer = SpeakerClusterer(
            expected_speakers=expected_speakers,
            max_speakers=max_speakers,
            match_threshold=match_threshold,
            new_speaker_margin=new_speaker_margin,
            min_new_speaker_sec=min_new_speaker_sec,
            recluster_every_turns=recluster_every_turns,
        )
        self._refresh_pending = False

    # ---------------- model ----------------

    def warmup(self) -> None:
        """Bind this session to the shared embedder (loading it if needed)."""
        if not self.enabled or self._inference is not None:
            return
        self._inference = _get_inference(self.hf_token, self._device)

    def _embed(self, pcm: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
        """Embed a float32 mono waveform. None if too short or too quiet."""
        if self._inference is None:
            self.warmup()
        if len(pcm) < int(0.25 * sample_rate):
            return None
        import torch

        wav = torch.from_numpy(pcm[None, :].astype("float32"))
        try:
            emb = self._inference({"waveform": wav, "sample_rate": sample_rate})
        except (
            Exception
        ) as exc:  # noqa: BLE001 - never kill the stream on one bad window
            log.warning("embedding failed: %s", exc)
            return None
        return np.asarray(emb, dtype=np.float64).ravel()

    # ---------------- turn splitting ----------------

    def _find_boundaries(self, pcm: np.ndarray, sr: int) -> list[float]:
        """Return split offsets (seconds from segment start).

        Adjacent sliding windows are embedded and compared. A local maximum in
        that distance curve, above `split_threshold`, is a voice change.
        """
        dur = len(pcm) / sr
        if dur < MIN_SPLIT_SEC:
            return []

        win, hop = int(WIN_SEC * sr), int(HOP_SEC * sr)
        starts = list(range(0, max(1, len(pcm) - win + 1), hop))
        if len(starts) < 2:
            return []

        embs, centers = [], []
        for s in starts:
            chunk = pcm[s : s + win]
            if float(np.sqrt(np.mean(chunk**2))) < 0.004:  # silence, skip
                continue
            e = self._embed(chunk, sr)
            if e is None:
                continue
            n = np.linalg.norm(e)
            embs.append(e / n if n > 1e-9 else e)
            centers.append((s + win / 2) / sr)
        if len(embs) < 3:
            return []

        E = np.stack(embs)
        d = 1.0 - np.sum(E[:-1] * E[1:], axis=1)  # adjacent cosine distance

        cuts: list[float] = []
        for i in range(1, len(d) - 1):
            if d[i] < self.split_threshold:
                continue
            if d[i] < d[i - 1] or d[i] < d[i + 1]:  # keep local maxima only
                continue
            t = (centers[i] + centers[i + 1]) / 2
            if t < MIN_TURN_SEC or (dur - t) < MIN_TURN_SEC:
                continue
            if cuts and (t - cuts[-1]) < MIN_TURN_SEC:
                continue
            cuts.append(t)
            log.debug("turn boundary at %.2fs (dist=%.3f)", t, d[i])
        return cuts

    # ---------------- main entry ----------------

    def process_segment(
        self, pcm_bytes: bytes, sample_rate: int, segment_start: float
    ) -> list[TurnOut]:
        """Split one VAD segment into turns and label each one.

        `segment_start` is this segment's offset in session seconds. Returns
        one TurnOut per turn — main.py should transcribe each SEPARATELY so a
        two-voice segment never becomes one mislabelled paragraph.
        """
        if not self.enabled:
            return [
                TurnOut(
                    "Speaker 1",
                    0,
                    segment_start,
                    segment_start + len(pcm_bytes) / 2 / sample_rate,
                    pcm_bytes,
                    0,
                )
            ]

        pcm = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        dur = len(pcm) / sample_rate
        if dur <= 0:
            return []

        cuts = self._find_boundaries(pcm, sample_rate)
        bounds = [0.0, *cuts, dur]
        if cuts:
            log.info(
                "segment %.1fs split into %d turn(s) at %s",
                dur,
                len(bounds) - 1,
                ", ".join(f"{c:.2f}s" for c in cuts),
            )

        out: list[TurnOut] = []
        for a, b in zip(bounds, bounds[1:]):
            i0, i1 = int(a * sample_rate), int(b * sample_rate)
            chunk = pcm[i0:i1]
            if len(chunk) < int(0.2 * sample_rate):
                continue
            emb = self._embed(chunk, sample_rate)
            if emb is None:
                continue
            turn = self.clusterer.add(emb, segment_start + a, segment_start + b)
            out.append(
                TurnOut(
                    speaker=self.clusterer.speaker_name(turn.speaker),
                    speaker_id=turn.speaker,
                    start=turn.start,
                    end=turn.end,
                    pcm=pcm_bytes[i0 * 2 : i1 * 2],
                    turn_id=turn.turn_id,
                )
            )

        if self.clusterer.should_recluster() and self.clusterer.recluster():
            self._refresh_pending = True
        return out

    def take_refresh(self) -> Optional[list[TurnOut]]:
        """Pop the corrected transcript, if the last recluster changed labels.

        main.py should call this after each segment; when it returns a list,
        rebuild paragraphs from it and send schemas.refresh_msg(...).
        """
        if not self._refresh_pending:
            return None
        self._refresh_pending = False
        return [
            TurnOut(
                speaker=self.clusterer.speaker_name(t.speaker),
                speaker_id=t.speaker,
                start=t.start,
                end=t.end,
                pcm=b"",
                turn_id=t.turn_id,
            )
            for t in self.clusterer.turns
        ]

    def finalize(self) -> Optional[list[TurnOut]]:
        """Force a last global recluster on stop — the most accurate pass."""
        if not self.enabled:
            return None
        if self.clusterer.recluster():
            self._refresh_pending = False
            return [
                TurnOut(
                    self.clusterer.speaker_name(t.speaker),
                    t.speaker,
                    t.start,
                    t.end,
                    b"",
                    t.turn_id,
                )
                for t in self.clusterer.turns
            ]
        return None

    def reset(self) -> None:
        self.clusterer.reset()
        self._refresh_pending = False


# ---------------------------------------------------------------------------
# Backwards-compatible name.
#
# v5's main.py did `from .diarization import Diarizer`. This alias makes that
# import line resolve, but it does NOT make the old CALL SITES work: the v5
# class had a different constructor and a different per-segment method. Treat
# this as a bridge while main.py is updated, not as a drop-in.
# ---------------------------------------------------------------------------
Diarizer = SessionDiarizer
