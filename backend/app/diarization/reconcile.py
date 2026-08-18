"""
reconcile.py — compare two speaker timelines deterministically.

WHY
---
Review point 10, stated as plainly as it can be: do not ask a second system to
decide the final speaker labels blindly. When Sortformer says A B A and another
system says A A B, "the newer model is probably right" is not a reconciliation
strategy. You need a number.

So this module does what the review specified:

    timeline A + timeline B
            |
      time overlap
            |
    speaker correspondence matrix
            |
      Hungarian matching
            |
      agreement score  +  where they disagree

It is arithmetic on two lists of tuples. No model, no audio, no I/O — which
means it is unit tested, and it works on ANY two timelines: the rolling
Sortformer result against the offline one, the offline one against pyannote,
either against a Gemini diarized transcript, or any of them against a hand
annotation.

THE RELATIONSHIP TO scoring.py
------------------------------
`scoring.py` scores a hypothesis against GROUND TRUTH and gives you a DER. It
is the right tool when you have annotated the audio by hand.

This is for when you have not. Two systems agreeing tells you nothing about
whether either is correct — they can be wrong together — but two systems
DISAGREEING localises where at least one of them is wrong, which is exactly
what you want to hand the annotator so ten minutes of their time lands on the
regions that matter.

Read the agreement score as a confidence signal, never as an accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

Segment = tuple[float, float, int]


def _overlap(a: Segment, b: Segment) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def correspondence_matrix(
    left: Sequence[Segment], right: Sequence[Segment]
) -> tuple[np.ndarray, list[int], list[int]]:
    """Seconds every (left speaker, right speaker) pair spend labelling the
    same audio. This is the whole evidence base for the matching below."""
    left_ids = sorted({s for _, _, s in left})
    right_ids = sorted({s for _, _, s in right})
    matrix = np.zeros((len(left_ids), len(right_ids)))
    li = {s: i for i, s in enumerate(left_ids)}
    ri = {s: i for i, s in enumerate(right_ids)}
    for seg_l in left:
        for seg_r in right:
            ov = _overlap(seg_l, seg_r)
            if ov > 0:
                matrix[li[seg_l[2]], ri[seg_r[2]]] += ov
    return matrix, left_ids, right_ids


@dataclass
class Disagreement:
    start: float
    end: float
    left_speaker: int
    right_speaker: int

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Reconciliation:
    """The verdict, and the working that produced it."""

    mapping: dict[int, int] = field(default_factory=dict)  # right id -> left id
    agreed_sec: float = 0.0
    disagreed_sec: float = 0.0
    left_only_sec: float = 0.0  # left labels speech, right labels nothing
    right_only_sec: float = 0.0
    left_speakers: int = 0
    right_speakers: int = 0
    disagreements: list[Disagreement] = field(default_factory=list)

    @property
    def compared_sec(self) -> float:
        return self.agreed_sec + self.disagreed_sec

    @property
    def agreement(self) -> float:
        """Share of co-labelled time the two systems assign to the same
        speaker, after the best possible renaming. 1.0 means they differ only
        in what they call people."""
        return self.agreed_sec / self.compared_sec if self.compared_sec else 0.0

    @property
    def speaker_count_agrees(self) -> bool:
        return self.left_speakers == self.right_speakers

    def worst(self, n: int = 10) -> list[Disagreement]:
        """The longest disagreements — where to point the annotator first."""
        return sorted(self.disagreements, key=lambda d: -d.duration)[:n]

    def report(self, left_name: str = "left", right_name: str = "right") -> str:
        lines = [
            f"{right_name}  vs  {left_name}",
            f"  speakers          {self.left_speakers} ({left_name})  "
            f"vs  {self.right_speakers} ({right_name})"
            f"{'' if self.speaker_count_agrees else '   <-- DISAGREE'}",
            f"  compared          {self.compared_sec:.1f}s of co-labelled speech",
            f"  agreement         {self.agreement:.1%}",
            f"  disagreed         {self.disagreed_sec:.1f}s",
            f"  only {left_name:<12} {self.left_only_sec:.1f}s",
            f"  only {right_name:<12} {self.right_only_sec:.1f}s",
        ]
        if self.mapping:
            pairs = ", ".join(f"{r}->{l}" for r, l in sorted(self.mapping.items()))
            lines.append(f"  speaker mapping   {pairs}")
        worst = self.worst(5)
        if worst:
            lines.append("  longest disagreements:")
            for d in worst:
                lines.append(
                    f"    {d.start:7.2f} - {d.end:7.2f}s   "
                    f"{left_name} said {d.left_speaker}, "
                    f"{right_name} said {d.right_speaker}"
                )
        return "\n".join(lines)


def reconcile(
    left: Sequence[Segment],
    right: Sequence[Segment],
    min_disagreement_sec: float = 0.20,
) -> Reconciliation:
    """Align `right`'s speaker numbering onto `left`'s and measure the fit.

    `left` is the reference namespace — its ids are kept. Nothing is decided
    here and nothing is overridden: the caller gets a mapping and a score and
    chooses what to do with them.

    Regions shorter than `min_disagreement_sec` are still counted in the
    totals but not listed individually, because boundary jitter between two
    systems produces a great many sub-100 ms slivers that are noise, not
    evidence, and drowning the report in them hides the real ones.
    """
    result = Reconciliation(
        left_speakers=len({s for _, _, s in left}),
        right_speakers=len({s for _, _, s in right}),
    )
    if not left or not right:
        result.left_only_sec = sum(b - a for a, b, _ in left)
        result.right_only_sec = sum(b - a for a, b, _ in right)
        return result

    matrix, left_ids, right_ids = correspondence_matrix(left, right)

    # Hungarian on the negated matrix: the one-to-one renaming that maximises
    # total agreement. One-to-one matters — without it two distinct speakers on
    # the right could both collapse onto one on the left and the score would
    # flatter a system that had merged them.
    rows, cols = linear_sum_assignment(-matrix)
    for r, c in zip(rows, cols):
        if matrix[r, c] > 0:
            result.mapping[right_ids[c]] = left_ids[r]

    remapped = [(a, b, result.mapping.get(s, -1000 - s)) for a, b, s in right]

    # Walk both timelines on a shared set of boundaries so every instant is
    # attributed exactly once.
    edges = sorted(
        {t for a, b, _ in left for t in (a, b)}
        | {t for a, b, _ in remapped for t in (a, b)}
    )
    for start, end in zip(edges, edges[1:]):
        if end <= start:
            continue
        mid = (start + end) / 2.0
        l_spk = _speaker_at(left, mid)
        r_spk = _speaker_at(remapped, mid)
        span = end - start

        if l_spk is None and r_spk is None:
            continue
        if r_spk is None:
            result.left_only_sec += span
        elif l_spk is None:
            result.right_only_sec += span
        elif l_spk == r_spk:
            result.agreed_sec += span
        else:
            result.disagreed_sec += span
            original = next(
                (s for s, mapped in result.mapping.items() if mapped == r_spk), r_spk
            )
            if span >= min_disagreement_sec:
                result.disagreements.append(Disagreement(start, end, l_spk, original))

    result.disagreements = _merge_adjacent(result.disagreements)
    return result


def _speaker_at(timeline: Sequence[Segment], t: float):
    for a, b, s in timeline:
        if a <= t < b:
            return s
    return None


def _merge_adjacent(items: list[Disagreement]) -> list[Disagreement]:
    out: list[Disagreement] = []
    for d in sorted(items, key=lambda x: x.start):
        if (
            out
            and out[-1].left_speaker == d.left_speaker
            and out[-1].right_speaker == d.right_speaker
            and abs(out[-1].end - d.start) < 1e-6
        ):
            out[-1].end = d.end
        else:
            out.append(d)
    return out


def relabel(timeline: Sequence[Segment], mapping: dict[int, int]) -> list[Segment]:
    """Rewrite a timeline into another's speaker namespace. Unmapped speakers
    keep their own id, so a genuinely extra speaker survives instead of being
    quietly folded into someone else."""
    return [(a, b, mapping.get(s, s)) for a, b, s in timeline]
