"""
scoring.py — score a speaker timeline against a hand-annotated one.

WHY THIS IS A SEPARATE MODULE FROM THE CLI
------------------------------------------
The only way to know whether diarization is getting better is to measure it
against truth. That measurement is arithmetic, and arithmetic can be unit
tested; loading WeSpeaker and running it over a WAV cannot be, not quickly.
So the maths lives here with tests beside it, and `eval_diarization.py` is a
thin wrapper that produces two timelines and hands them over.

This also means you can score a timeline that came from somewhere else — a
different tool, a manual transcript, an older branch — without touching the
diarizer.

WHAT IS MEASURED
----------------
The standard decomposition, on a fixed frame grid:

    MISS        reference has a speaker here, the system labelled nothing
    FALSE ALARM system labelled a speaker here, the reference has silence
    CONFUSION   both have a speaker, but not the same one after mapping

    DER = (miss + false_alarm + confusion) / total reference speech

WHAT IS DELIBERATELY *NOT* MEASURED, AND WHY IT MATTERS HERE
------------------------------------------------------------
OVERLAPPING SPEECH IS NOT SCORED AS OVERLAP. Both the reference and the
hypothesis are flattened to ONE speaker per frame. That is not a shortcut
taken for convenience — it is forced by the system under test. The embedding
backend cannot emit two speakers for the same instant, so a region where two
people talk at once can only ever be scored as "one of them, plus a confusion
against the other".

The consequence is worth stating plainly, because it decides how you read the
output: **this metric cannot show you the cost of the overlap limitation.** A
backend that handles overlap (Sortformer) and one that cannot (embedding) will
look closer here than they really are on overlapping audio. If overlap is what
you care about, the number to look at is not DER — it is listening to the
overlapped regions and checking which speaker got dropped.

Where the reference marks two speakers at the same time, the LAST annotation
wins on that frame. Which one wins is arbitrary; what matters is that the same
rule applies to every backend being compared, so the comparison stays fair
even though the absolute number is soft.

THE COLLAR
----------
Frames within `collar` seconds of a reference boundary are excluded from
scoring. Human annotation of a turn boundary is not accurate to the
millisecond, and without a collar you are largely measuring the annotator's
reflexes. 0.25 s each side is the usual convention (NIST). Set it to 0.0 to
see the unforgiving number.

MAPPING
-------
Speaker *names* are arbitrary. The system's "Speaker 1" and the reference's
"Speaker A" refer to whoever they refer to, and a run that got every turn
right but swapped the two names is a perfect run with a bad legend. So labels
are matched with the Hungarian algorithm — the single one-to-one mapping that
maximises agreement — before anything is counted. `scipy.optimize` is already
a dependency (the diarizer uses it for exactly this).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np

# 10 ms. Fine enough that the grid contributes nothing next to annotation
# error, coarse enough that an hour of audio is 360k frames rather than 16M.
FRAME_SEC = 0.01

DEFAULT_COLLAR_SEC = 0.25

# Frame value meaning "nobody is speaking here".
SILENCE = -1


@dataclass
class Turn:
    """One labelled span. `speaker` is whatever string the source used."""

    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Score:
    """The result of one comparison. All times in seconds."""

    reference_speech: float
    scored_speech: float  # reference speech left after the collar
    miss: float
    false_alarm: float
    confusion: float
    correct: float
    reference_speakers: int
    hypothesis_speakers: int
    reference_turns: int
    hypothesis_turns: int
    mapping: dict = field(default_factory=dict)
    collar: float = DEFAULT_COLLAR_SEC

    @property
    def der(self) -> float:
        """Diarization error rate over the scored region. 0.0 is perfect."""
        if self.scored_speech <= 0:
            return 0.0
        return (self.miss + self.false_alarm + self.confusion) / self.scored_speech

    @property
    def accuracy(self) -> float:
        """Share of scored reference speech given the right speaker."""
        if self.scored_speech <= 0:
            return 0.0
        return self.correct / self.scored_speech

    @property
    def speaker_count_correct(self) -> bool:
        return self.reference_speakers == self.hypothesis_speakers

    def report(self) -> str:
        lines = [
            f"  reference speech      {self.reference_speech:8.2f}s",
            f"  scored (collar {self.collar:.2f}s) {self.scored_speech:8.2f}s",
            "",
            f"  correct               {self.correct:8.2f}s  {self.accuracy:6.1%}",
            f"  confusion             {self.confusion:8.2f}s  "
            f"{self._share(self.confusion):6.1%}",
            f"  missed                {self.miss:8.2f}s  "
            f"{self._share(self.miss):6.1%}",
            f"  false alarm           {self.false_alarm:8.2f}s  "
            f"{self._share(self.false_alarm):6.1%}",
            "",
            f"  DER                   {self.der:8.1%}",
            "",
            f"  speakers   reference {self.reference_speakers}   "
            f"system {self.hypothesis_speakers}"
            + ("" if self.speaker_count_correct else "   <-- MISMATCH"),
            f"  turns      reference {self.reference_turns}   "
            f"system {self.hypothesis_turns}",
        ]
        if self.mapping:
            pairs = ", ".join(f"{h} -> {r}" for h, r in sorted(self.mapping.items()))
            lines.append(f"  mapping    {pairs}")
        return "\n".join(lines)

    def _share(self, value: float) -> float:
        return value / self.scored_speech if self.scored_speech > 0 else 0.0


# ---------------------------------------------------------------- parsing


def parse_rttm(text: str) -> list[Turn]:
    """Read NIST RTTM. Only SPEAKER lines matter.

    SPEAKER file 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>
    """
    turns: list[Turn] = []
    for raw in text.splitlines():
        parts = raw.split()
        if len(parts) < 8 or parts[0].upper() != "SPEAKER":
            continue
        try:
            start = float(parts[3])
            duration = float(parts[4])
        except ValueError:
            continue
        turns.append(Turn(start, start + duration, parts[7]))
    return turns


def parse_simple(text: str) -> list[Turn]:
    """Read the format you can actually produce by hand in Audacity.

        start end speaker

    Whitespace, comma or tab separated. `#` comments and blank lines ignored.
    Audacity's "Export Labels" writes exactly this (tab separated), which is
    why it is supported: annotating a ten-minute Sinhala recording should not
    require learning RTTM.
    """
    turns: list[Turn] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [p for p in line.replace(",", " ").replace("\t", " ").split() if p]
        if len(parts) < 3:
            continue
        try:
            start, end = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        turns.append(Turn(start, end, " ".join(parts[2:])))
    return turns


def parse_annotation(text: str) -> list[Turn]:
    """Accept either format; RTTM wins if any SPEAKER line is present."""
    rttm = parse_rttm(text)
    return rttm if rttm else parse_simple(text)


def timeline_to_turns(
    timeline: Iterable[tuple[float, float, int]], prefix: str = "S"
) -> list[Turn]:
    """Convert a diarizer `timeline()` into Turns.

    The engines emit 0-based integer ids; the UI shows them 1-based
    ("Speaker 1"), so the same convention is used here to keep the printed
    mapping readable against what you saw on screen.
    """
    return [Turn(a, b, f"{prefix}{int(s) + 1}") for a, b, s in timeline]


# ------------------------------------------------------------- frame grid


def _grid(turns: Sequence[Turn], n_frames: int, labels: dict[str, int]) -> np.ndarray:
    """Flatten turns onto a frame grid. Later turns overwrite earlier ones."""
    frames = np.full(n_frames, SILENCE, dtype=np.int32)
    for turn in turns:
        i0 = max(0, int(round(turn.start / FRAME_SEC)))
        i1 = min(n_frames, int(round(turn.end / FRAME_SEC)))
        if i1 > i0:
            frames[i0:i1] = labels[turn.speaker]
    return frames


def _collar_mask(turns: Sequence[Turn], n_frames: int, collar: float) -> np.ndarray:
    """True where a frame may be scored (i.e. away from reference boundaries)."""
    mask = np.ones(n_frames, dtype=bool)
    if collar <= 0:
        return mask
    half = int(round(collar / FRAME_SEC))
    for turn in turns:
        for boundary in (turn.start, turn.end):
            centre = int(round(boundary / FRAME_SEC))
            mask[max(0, centre - half) : min(n_frames, centre + half + 1)] = False
    return mask


def _label_index(turns: Sequence[Turn]) -> dict[str, int]:
    return {name: i for i, name in enumerate(dict.fromkeys(t.speaker for t in turns))}


def _best_mapping(
    ref: np.ndarray,
    hyp: np.ndarray,
    n_ref: int,
    n_hyp: int,
) -> dict[int, int]:
    """Hungarian match of hypothesis ids to reference ids.

    Maximises agreement, so a run that is right about every turn but names the
    speakers the other way round scores as correct — which it is.
    """
    if n_ref == 0 or n_hyp == 0:
        return {}

    counts = np.zeros((n_hyp, n_ref), dtype=np.int64)
    both = (ref != SILENCE) & (hyp != SILENCE)
    if both.any():
        np.add.at(counts, (hyp[both], ref[both]), 1)

    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(-counts)
        return {int(r): int(c) for r, c in zip(rows, cols) if counts[r, c] > 0}
    except ImportError:
        # Greedy fallback. scipy is in requirements.txt, so this should not
        # run — but a scoring tool refusing to start is worse than a slightly
        # pessimistic mapping.
        mapping: dict[int, int] = {}
        used: set[int] = set()
        order = np.argsort(-counts, axis=None)
        for flat in order:
            h, r = divmod(int(flat), n_ref)
            if counts[h, r] <= 0:
                break
            if h in mapping or r in used:
                continue
            mapping[h] = r
            used.add(r)
        return mapping


# ---------------------------------------------------------------- scoring


def score_timelines(
    reference: Sequence[Turn],
    hypothesis: Sequence[Turn],
    duration: Optional[float] = None,
    collar: float = DEFAULT_COLLAR_SEC,
) -> Score:
    """Compare a system timeline against an annotated one."""
    ends = [t.end for t in reference] + [t.end for t in hypothesis]
    total = duration if duration is not None else (max(ends) if ends else 0.0)
    n_frames = max(1, int(round(total / FRAME_SEC)))

    ref_labels = _label_index(reference)
    hyp_labels = _label_index(hypothesis)

    ref = _grid(reference, n_frames, ref_labels)
    hyp = _grid(hypothesis, n_frames, hyp_labels)

    mapping = _best_mapping(ref, hyp, len(ref_labels), len(hyp_labels))
    # Unmapped hypothesis speakers become a value no reference id can equal,
    # so every frame they cover counts as confusion rather than silently
    # matching. This is what makes an over-clustered run (17 speakers for 2)
    # score badly instead of accidentally well.
    remap = np.full(max(1, len(hyp_labels)), -2, dtype=np.int32)
    for h, r in mapping.items():
        remap[h] = r
    hyp_mapped = np.where(hyp == SILENCE, SILENCE, remap[np.clip(hyp, 0, None)])

    scorable = _collar_mask(reference, n_frames, collar)
    ref_s = ref[scorable]
    hyp_s = hyp_mapped[scorable]

    ref_speech = ref_s != SILENCE
    hyp_speech = hyp_s != SILENCE

    frame = FRAME_SEC
    correct = (
        float(np.count_nonzero(ref_speech & hyp_speech & (ref_s == hyp_s))) * frame
    )
    confusion = (
        float(np.count_nonzero(ref_speech & hyp_speech & (ref_s != hyp_s))) * frame
    )
    miss = float(np.count_nonzero(ref_speech & ~hyp_speech)) * frame
    false_alarm = float(np.count_nonzero(~ref_speech & hyp_speech)) * frame

    inverse = {h: list(hyp_labels)[h] for h in range(len(hyp_labels))}
    ref_names = list(ref_labels)
    readable = {inverse[h]: ref_names[r] for h, r in mapping.items()}

    return Score(
        reference_speech=sum(t.duration for t in reference),
        scored_speech=float(np.count_nonzero(ref_speech)) * frame,
        miss=miss,
        false_alarm=false_alarm,
        confusion=confusion,
        correct=correct,
        reference_speakers=len(ref_labels),
        hypothesis_speakers=len(hyp_labels),
        reference_turns=len(reference),
        hypothesis_turns=len(hypothesis),
        mapping=readable,
        collar=collar,
    )


__all__ = [
    "DEFAULT_COLLAR_SEC",
    "FRAME_SEC",
    "Score",
    "Turn",
    "parse_annotation",
    "parse_rttm",
    "parse_simple",
    "score_timelines",
    "timeline_to_turns",
]
