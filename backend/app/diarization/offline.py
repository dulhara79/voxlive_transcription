"""
offline.py — whole-file diarization, for the FINAL pass and for the bake-off.

WHY
---
Review point 11: after `stop`, the complete recording is in hand. Running the
rolling 20/90-second window one more time and calling that a "final pass" wastes
the one advantage the final pass has. `session/state.py` even documents the
finalize call as "plain offline diarization ... the best labelling the system
can produce", which for the Sortformer backend was not true — `finalize()` just
called `_pass()` again.

Review point 4 separates the two regimes properly:

    LIVE    streaming model, bounded latency, rolling window
    FINAL   offline model, whole recording, one inference, no stitching

Review point 13 adds pyannote Community-1 as the first alternative to benchmark
for that final slot.

So this module holds whole-file diarizers behind one interface. They are
candidates in `diag_diarization.py`, and one of them can be wired in as the
final pass via SORTFORMER_FINAL_PASS.

WHAT IS AND IS NOT VERIFIED HERE
--------------------------------
The model ids, the loading calls and the output shapes below were taken from
the current model cards, not from memory:

  * nvidia/diar_sortformer_4spk-v1 — the NON-streaming Sortformer. Same
    `SortformerEncLabelModel.from_pretrained(...)` / `.diarize(audio=[array],
    batch_size=, sample_rate=)` surface as the streaming model already in
    `sortformer.py`. Max 4 speakers; degrades at 5+. Licensed CC-BY-NC-4.0 —
    NOT the CC-BY-4.0 of the streaming model. **That is a commercial-use
    question for SLT and it is not mine to answer.** Check it before this ships
    to anything but your own laptop.
    The card also notes a practical length limit set by GPU memory (roughly 12
    minutes on a 48 GB RTX A6000), which is why the final pass has a duration
    ceiling rather than silently attempting an hour.

  * pyannote/speaker-diarization-community-1 — `Pipeline.from_pretrained(id,
    token=...)`, called with a path or `{"waveform": tensor, "sample_rate":
    int}`, returning `output.speaker_diarization` and
    `output.exclusive_speaker_diarization`. Gated: accept the user conditions
    on the model page first. No speaker ceiling, and accepts `num_speakers` /
    `min_speakers` / `max_speakers`.

NOTHING HERE HAS BEEN RUN. This container has no GPU, no checkpoints and no HF
token, so every function below is written against the documented API and is
UNVERIFIED AGAINST A REAL MODEL. The first time you run `diag_diarization.py`
you are also testing this file. Treat a failure here as a bug in this file
before you treat it as a finding about the model.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence

import numpy as np

log = logging.getLogger("voxlive.diar.offline")

Segment = tuple[float, float, int]

# The offline (non-streaming) Sortformer. Different checkpoint AND different
# licence from the streaming model in sortformer.py — see the module docstring.
OFFLINE_SORTFORMER_MODEL_ID = "nvidia/diar_sortformer_4spk-v1"
PYANNOTE_COMMUNITY1_ID = "pyannote/speaker-diarization-community-1"

_CACHE: dict = {}


def pcm_to_float(pcm_bytes: bytes) -> np.ndarray:
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0


# --------------------------------------------------------------------- helpers


def _normalise_nemo_output(out) -> list[Segment]:
    """NeMo returns either 'start end speaker' strings or tuples depending on
    version. `sortformer.py` already accepts both; do the same here rather than
    pinning a version in one file and not the other."""
    segs: list[Segment] = []
    for item in (out[0] if out else []):
        if isinstance(item, str):
            parts = item.replace(",", " ").split()
            if len(parts) < 3:
                continue
            a, b, spk = float(parts[0]), float(parts[1]), parts[2]
        else:
            a, b, spk = float(item[0]), float(item[1]), item[2]
        idx = int(str(spk).replace("speaker_", "").strip() or 0)
        segs.append((a, b, idx))
    segs.sort()
    return segs


def _normalise_pyannote_annotation(annotation) -> list[Segment]:
    """pyannote labels speakers as strings ('SPEAKER_00'); the rest of this
    codebase uses ints. Map them in order of first appearance, which is also
    Sortformer's convention, so the two are at least comparable by eye."""
    turns = []
    try:
        for turn, speaker in annotation:
            turns.append((float(turn.start), float(turn.end), str(speaker)))
    except (TypeError, ValueError):
        # Older pyannote Annotation objects iterate differently.
        for segment, _track, label in annotation.itertracks(yield_label=True):
            turns.append((float(segment.start), float(segment.end), str(label)))

    turns.sort()
    order: dict[str, int] = {}
    out: list[Segment] = []
    for a, b, label in turns:
        if label not in order:
            order[label] = len(order)
        out.append((a, b, order[label]))
    return out


# ------------------------------------------------------------------ backends


def diarize_offline_sortformer(
    pcm_bytes: bytes,
    sample_rate: int = 16000,
    max_speakers: Optional[int] = None,
    device: Optional[str] = None,
    model_id: str = OFFLINE_SORTFORMER_MODEL_ID,
    **_ignored,
) -> list[Segment]:
    """One inference over the whole recording. No window, no stitching.

    Deliberately does NOT apply the STREAM_CFG from `sortformer.py`. Those
    settings configure the streaming model's speaker cache and chunking; the
    offline model has no streaming path to configure and setting them would be
    at best meaningless and at worst wrong.
    """
    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    key = ("nemo-offline", model_id, dev)
    if key not in _CACHE:
        log.info("loading %s on %s (offline mode) ...", model_id, dev)
        model = SortformerEncLabelModel.from_pretrained(model_id, map_location=dev)
        model.eval()
        _CACHE[key] = model
    model = _CACHE[key]

    audio = pcm_to_float(pcm_bytes)
    out = model.diarize(audio=[audio], batch_size=1, sample_rate=sample_rate)
    segs = _normalise_nemo_output(out)

    # The 4-speaker ceiling is a property of the checkpoint, not a setting. A
    # request for more is a misunderstanding worth logging rather than a knob.
    if max_speakers and max_speakers > 4:
        log.warning(
            "offline Sortformer caps at 4 speakers; max_speakers=%d cannot be "
            "honoured by this checkpoint",
            max_speakers,
        )
    return segs


def diarize_pyannote_community1(
    pcm_bytes: bytes,
    sample_rate: int = 16000,
    max_speakers: Optional[int] = None,
    num_speakers: Optional[int] = None,
    device: Optional[str] = None,
    hf_token: str = "",
    model_id: str = PYANNOTE_COMMUNITY1_ID,
    exclusive: bool = True,
    **_ignored,
) -> list[Segment]:
    """pyannote Community-1 over the whole recording.

    `exclusive=True` uses `exclusive_speaker_diarization`, where only one
    speaker is active at any instant. That is the right choice for THIS system:
    `TranscriptStore.label_for()` assigns each transcript chunk the speaker
    with maximum overlap, so it can only consume one speaker per instant
    anyway, and the exclusive output exists precisely to make that join clean.

    The cost is that overlapped speech is resolved for you rather than
    reported. If you are evaluating how the system handles cross-talk, set
    exclusive=False and look at the regular output instead.
    """
    import torch
    from pyannote.audio import Pipeline

    key = ("pyannote", model_id)
    if key not in _CACHE:
        log.info("loading %s ...", model_id)
        pipeline = Pipeline.from_pretrained(model_id, token=hf_token or None)
        if pipeline is None:
            raise RuntimeError(
                f"{model_id} did not load. Accept the user conditions on the "
                "model page and pass a HUGGINGFACE_TOKEN with read access."
            )
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            pipeline.to(torch.device(dev))
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "could not move %s to %s: %s (staying on CPU)", model_id, dev, exc
            )
        _CACHE[key] = pipeline
    pipeline = _CACHE[key]

    waveform = torch.from_numpy(pcm_to_float(pcm_bytes)).unsqueeze(0)  # (1, samples)
    options: dict = {}
    if num_speakers:
        options["num_speakers"] = int(num_speakers)
    elif max_speakers:
        options["max_speakers"] = int(max_speakers)

    output = pipeline({"waveform": waveform, "sample_rate": sample_rate}, **options)

    annotation = getattr(output, "speaker_diarization", output)
    if exclusive:
        annotation = getattr(output, "exclusive_speaker_diarization", annotation)
    return _normalise_pyannote_annotation(annotation)


# ------------------------------------------------------------------ registry

# `diag_diarization.py` iterates this. Adding a fourth candidate — the Gemini
# diarized transcription of review point 5 — means adding one function with
# this signature and one line here. It is deliberately NOT in the live path:
# review point 6 is explicit that a generative whole-file call should not be
# the low-latency diarizer, and review point 10 is explicit that it must not
# decide final labels on its own either. Run it here, reconcile it in
# `reconcile.py`, and look at the agreement score before trusting it anywhere.

OFFLINE_BACKENDS: dict[str, Callable[..., list[Segment]]] = {
    "sortformer_offline": diarize_offline_sortformer,
    "pyannote_community1": diarize_pyannote_community1,
}


def available() -> list[str]:
    return sorted(OFFLINE_BACKENDS)


def run(name: str, pcm_bytes: bytes, **kwargs) -> list[Segment]:
    if name not in OFFLINE_BACKENDS:
        raise KeyError(f"unknown offline backend {name!r}; have {available()}")
    return OFFLINE_BACKENDS[name](pcm_bytes, **kwargs)
