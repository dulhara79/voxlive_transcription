"""
stitching.py — cross-window speaker identity for the Sortformer backend.

WHY THIS MODULE EXISTS
----------------------
The review of `fature/optimize_diarization` made one structural criticism of
`SortformerDiarizer._stitch()`: it was doing too much in one function. In a
single body it received the local result, matched local slots to session
speakers, minted new speakers, enforced the speaker cap, applied the
nearest-prior fallback, and merged the timeline.

The cost of that was not style. It was that a wrong final timeline could not be
attributed. When the output is wrong you need to answer:

    Was Sortformer wrong?          (model / input)
    Was our stitching wrong?       (this file)

and with one function those two error sources were intertwined.

So the pipeline is now five named stages, exactly as the review specified:

    SortformerInference          (stays in sortformer.py — the model call)
            |
    LocalSpeakerTimeline         one window's slots, in session time
            |
    SpeakerAlignment             local slot -> session id, on overlap evidence
            |
    SessionIdentityManager       minting, the cap, the fallback
            |
    GlobalTimeline               splice, collapse, change detection

Each stage is a plain object with plain inputs and outputs, so each can be
tested on its own without loading a 493 MB checkpoint, and `StitchTrace`
records what every stage decided so `diag_diarization.py` can print the chain.

BEHAVIOUR IS UNCHANGED ON PURPOSE
---------------------------------
This is a refactor, not a fix. The algorithm here is the same one that was in
`_stitch()`, decision for decision, and `tests/test_stitching.py` holds a copy
of the original function and asserts the two agree on randomised input. That
matters: the whole point of the split is to let you measure the stitching
layer, and a refactor that also changed its behaviour would invalidate every
measurement taken before it.

Whether the algorithm is RIGHT is the separate question the diagnostic tool
exists to answer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

log = logging.getLogger("voxlive.sortformer.stitch")

# A speaker run: (start_seconds, end_seconds, speaker_id).
Segment = tuple[float, float, int]

# Two runs of the same speaker closer than this are one run. The rolling window
# re-runs the model every interval, and a turn that straddles the seam comes
# back as two runs with a hairline gap; this closes it.
COLLAPSE_GAP_SEC = 0.12


# --------------------------------------------------------------------------
# Stage 1: LocalSpeakerTimeline
# --------------------------------------------------------------------------


@dataclass
class LocalSpeakerTimeline:
    """One window's Sortformer output, in session seconds.

    Sortformer numbers speakers by arrival order WITHIN THE INPUT IT WAS GIVEN.
    These ids are local to this window and carry no session meaning — slot 0
    here may be slot 1 in the window before it. Nothing downstream may treat
    them as identities.

    `raw` keeps the model's output before filtering and before the offset was
    applied, because the first question the diagnostic asks is whether the
    model was already wrong, and that question is unanswerable once the
    filtered, shifted version is all that survives.
    """

    segments: list[Segment]  # filtered, offset applied — session seconds
    offset: float  # session second at window start
    raw: list[Segment] = field(default_factory=list)  # model output, verbatim
    dropped_short: int = 0
    dropped_over_cap: int = 0

    @classmethod
    def from_inference(
        cls,
        raw_segments: Sequence[Segment],
        offset: float,
        min_turn_sec: float,
        max_speakers: int,
    ) -> "LocalSpeakerTimeline":
        kept: list[Segment] = []
        short = 0
        over = 0
        for a, b, s in raw_segments:
            if b - a < min_turn_sec:
                short += 1
                continue
            if s >= max_speakers:
                over += 1
                continue
            kept.append((a + offset, b + offset, s))
        kept.sort()
        return cls(
            segments=kept,
            offset=offset,
            raw=[(float(a), float(b), int(s)) for a, b, s in raw_segments],
            dropped_short=short,
            dropped_over_cap=over,
        )

    def slots(self) -> list[int]:
        return sorted({s for _, _, s in self.segments})

    def speech_by_slot(self) -> dict[int, float]:
        totals: dict[int, float] = {}
        for a, b, s in self.segments:
            totals[s] = totals.get(s, 0.0) + (b - a)
        return totals

    def first_appearance(self, slot: int) -> float:
        return min((a for a, _, s in self.segments if s == slot), default=0.0)

    def __bool__(self) -> bool:
        return bool(self.segments)


# --------------------------------------------------------------------------
# Stage 2: SpeakerAlignment
# --------------------------------------------------------------------------


@dataclass
class AlignmentResult:
    """What the acoustic-overlap evidence alone supports.

    `mapping` holds ONLY slots with positive evidence. Slots with none are
    listed in `unmatched` and left for the identity manager to decide, because
    "no evidence" is a policy question, not a matching one, and keeping the two
    apart is what makes each testable.
    """

    mapping: dict[int, int] = field(default_factory=dict)
    shared_sec: dict[int, float] = field(default_factory=dict)
    unmatched: list[int] = field(default_factory=list)
    session_ids: list[int] = field(default_factory=list)


class SpeakerAlignment:
    """Match this window's local slots to session speakers.

    The evidence is the region the window and the existing session timeline
    both cover: for every (local slot, session speaker) pair, how many seconds
    of speech do they share? The Hungarian algorithm then picks the single
    one-to-one assignment maximising that total, so one session speaker cannot
    absorb two local slots.

    This stage is pure. It reads nothing, mutates nothing, and mints nothing.
    """

    def match(
        self,
        local: LocalSpeakerTimeline,
        session_timeline: Sequence[Segment],
    ) -> AlignmentResult:
        # Only the part of the session the window also covers is evidence.
        overlap = [(a, b, s) for a, b, s in session_timeline if b > local.offset]
        local_ids = local.slots()
        sess_ids = sorted({s for _, _, s in overlap})

        result = AlignmentResult(session_ids=sess_ids)
        if not (overlap and local_ids and sess_ids):
            result.unmatched = list(local_ids)
            return result

        cost = np.zeros((len(local_ids), len(sess_ids)))
        for i, li in enumerate(local_ids):
            for j, sj in enumerate(sess_ids):
                shared = 0.0
                for a1, b1, s1 in local.segments:
                    if s1 != li:
                        continue
                    for a2, b2, s2 in overlap:
                        if s2 != sj:
                            continue
                        ov = min(b1, b2) - max(a1, a2)
                        if ov > 0:
                            shared += ov
                cost[i, j] = -shared

        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            if cost[r, c] < 0:  # strictly positive shared speech
                result.mapping[local_ids[r]] = sess_ids[c]
                result.shared_sec[local_ids[r]] = -float(cost[r, c])

        result.unmatched = [li for li in local_ids if li not in result.mapping]
        return result


# --------------------------------------------------------------------------
# Stage 3: SessionIdentityManager
# --------------------------------------------------------------------------


@dataclass
class IdentityDecision:
    mapping: dict[int, int] = field(default_factory=dict)
    minted: list[int] = field(default_factory=list)  # session ids created
    folded: dict[int, int] = field(default_factory=dict)  # slot -> id, at the cap


class SessionIdentityManager:
    """Owns the session's speaker ids: who exists, and what happens to a slot
    the alignment could not place.

    A local slot with no overlap evidence is USUALLY a genuinely new
    participant. But it can also be someone who simply said nothing during the
    overlap region — Sortformer has no memory across two separate `diarize()`
    calls, so there is no acoustic way to tell the two apart from here. Minting
    freely would let one person collect several ids over a long meeting, so the
    count is capped, and past the cap an unplaced slot is folded into whoever
    spoke most recently before it.

    That fallback is the single most suspicious line in this backend — it is
    what can turn a true A B C D into A B A B — so it is isolated here, counted
    in `folded`, and surfaced by the diagnostic rather than buried in a debug
    log.
    """

    def __init__(self, max_speakers: int):
        self.max_speakers = int(max_speakers)
        self.max_sid = -1

    def seed(self, timeline: Sequence[Segment]) -> None:
        """Adopt an existing timeline's numbering (first window, or reset)."""
        self.max_sid = max((s for _, _, s in timeline), default=-1)

    def reset(self) -> None:
        self.max_sid = -1

    def resolve(
        self,
        alignment: AlignmentResult,
        local: LocalSpeakerTimeline,
        session_timeline: Sequence[Segment],
    ) -> IdentityDecision:
        decision = IdentityDecision(mapping=dict(alignment.mapping))

        nxt = self.max_sid + 1
        for li in local.slots():
            if li in decision.mapping:
                continue
            if nxt >= self.max_speakers and session_timeline:
                target = self._nearest_prior_speaker(local, li, session_timeline)
                decision.mapping[li] = target
                decision.folded[li] = target
                log.debug(
                    "slot %d unmatched at the speaker cap; folded into %d", li, target
                )
                continue
            decision.mapping[li] = nxt
            decision.minted.append(nxt)
            nxt += 1

        self.max_sid = max(self.max_sid, nxt - 1)
        return decision

    @staticmethod
    def _nearest_prior_speaker(
        local: LocalSpeakerTimeline,
        slot: int,
        session_timeline: Sequence[Segment],
    ) -> int:
        """Last resort at the cap: whoever was talking most recently before
        this slot's first appearance. Conversation is sticky, so turn-taking
        adjacency is the only signal left once acoustic evidence has run out.
        It is a guess, and it is labelled as one."""
        first = local.first_appearance(slot)
        best, best_gap = 0, float("inf")
        for _, b, s in session_timeline:
            if b <= first and (first - b) < best_gap:
                best, best_gap = s, first - b
        return best


# --------------------------------------------------------------------------
# Stage 4: GlobalTimeline
# --------------------------------------------------------------------------


class GlobalTimeline:
    """The session timeline, and the only thing allowed to mutate it.

    Splicing keeps everything that ended before the window began and replaces
    the rest with the window's renamed runs, so the newest inference always
    wins where the two disagree.
    """

    def __init__(self) -> None:
        self._runs: list[Segment] = []

    def __len__(self) -> int:
        return len(self._runs)

    def __bool__(self) -> bool:
        return bool(self._runs)

    def runs(self) -> list[Segment]:
        return list(self._runs)

    def clear(self) -> None:
        self._runs = []

    def adopt(self, runs: Sequence[Segment]) -> bool:
        """Take a timeline verbatim, without collapsing.

        Used for the very first window, where the local numbering IS the
        session numbering. The original `_stitch()` did exactly this
        (`self._timeline = window`) and deliberately did not collapse, because
        a single Sortformer result has no seam in it to close. Collapsing here
        would merge adjacent same-speaker runs the model chose to keep apart,
        which is a behaviour change, not a tidy-up.
        """
        new = list(runs)
        changed = new != self._runs
        self._runs = new
        return changed

    def replace(self, runs: Sequence[Segment]) -> bool:
        """Overwrite wholesale, sorted and collapsed. Used by the offline final
        pass, which produces one timeline for the whole session and owes
        nothing to the rolling one."""
        new = self._collapse(sorted(runs))
        changed = new != self._runs
        self._runs = new
        return changed

    def splice(self, renamed: Sequence[Segment], offset: float) -> bool:
        merged = [(a, b, s) for a, b, s in self._runs if b <= offset]
        merged += list(renamed)
        merged.sort()
        new = self._collapse(merged)
        changed = new != self._runs
        self._runs = new
        return changed

    def end(self) -> Optional[float]:
        return self._runs[-1][1] if self._runs else None

    def speaker_count(self) -> int:
        return len({s for _, _, s in self._runs})

    @staticmethod
    def _collapse(runs: Sequence[Segment]) -> list[Segment]:
        out: list[Segment] = []
        for a, b, s in runs:
            if out and out[-1][2] == s and a - out[-1][1] < COLLAPSE_GAP_SEC:
                out[-1] = (out[-1][0], max(out[-1][1], b), s)
            else:
                out.append((a, b, s))
        return out


# --------------------------------------------------------------------------
# Orchestration + trace
# --------------------------------------------------------------------------


@dataclass
class StitchTrace:
    """One pass, end to end, in enough detail to assign blame.

    This is the record the review asked for in point 17: raw model output on
    one side, the stitched timeline on the other, and every renaming decision
    in between. Written to `raw_sortformer.json` / `stitched_timeline.json` by
    the diagnostic recorder.
    """

    index: int
    offset: float
    raw: list[Segment]
    window: list[Segment]
    alignment: dict[int, int]
    shared_sec: dict[int, float]
    minted: list[int]
    folded: dict[int, int]
    timeline_after: list[Segment]
    changed: bool

    def to_dict(self) -> dict:
        return {
            "pass": self.index,
            "offset_sec": round(self.offset, 3),
            "raw_segments": [[round(a, 3), round(b, 3), s] for a, b, s in self.raw],
            "window_segments": [
                [round(a, 3), round(b, 3), s] for a, b, s in self.window
            ],
            "alignment": {str(k): v for k, v in self.alignment.items()},
            "shared_sec": {str(k): round(v, 3) for k, v in self.shared_sec.items()},
            "minted": self.minted,
            "folded_at_cap": {str(k): v for k, v in self.folded.items()},
            "timeline_after": [
                [round(a, 3), round(b, 3), s] for a, b, s in self.timeline_after
            ],
            "changed": self.changed,
        }


class SortformerStitcher:
    """Runs stages 2-4 for one window and records what happened.

    `sortformer.py` owns stage 1 (the model call) and hands the result here.
    """

    def __init__(self, max_speakers: int):
        self.alignment = SpeakerAlignment()
        self.identity = SessionIdentityManager(max_speakers)
        self.timeline = GlobalTimeline()
        self.traces: list[StitchTrace] = []
        self.keep_traces = False
        self._passes = 0

    def reset(self) -> None:
        self.timeline.clear()
        self.identity.reset()
        self.traces.clear()
        self._passes = 0

    def ingest(self, local: LocalSpeakerTimeline) -> bool:
        """Fold one window into the session timeline. Returns True if the
        timeline changed."""
        self._passes += 1

        # First window: the local numbering IS the session numbering. There is
        # nothing to align against, so there is no alignment stage to run.
        if not self.timeline:
            changed = self.timeline.adopt(local.segments)
            self.identity.seed(local.segments)
            self._trace(local, {}, {}, [], {}, changed)
            return changed

        prior = self.timeline.runs()
        alignment = self.alignment.match(local, prior)
        decision = self.identity.resolve(alignment, local, prior)
        renamed = [(a, b, decision.mapping[s]) for a, b, s in local.segments]
        changed = self.timeline.splice(renamed, local.offset)

        self._trace(
            local,
            decision.mapping,
            alignment.shared_sec,
            decision.minted,
            decision.folded,
            changed,
        )
        return changed

    def adopt_offline(self, runs: Sequence[Segment]) -> bool:
        """Install a whole-session result, discarding the rolling numbering.

        The offline pass saw the entire recording in one inference, so its
        speaker ids are internally consistent by construction and there is
        nothing to stitch. Trying to reconcile them with the rolling timeline
        would reintroduce the very error source the offline pass exists to
        remove.
        """
        changed = self.timeline.replace(runs)
        self.identity.seed(self.timeline.runs())
        return changed

    def _trace(self, local, mapping, shared, minted, folded, changed) -> None:
        if not self.keep_traces:
            return
        self.traces.append(
            StitchTrace(
                index=self._passes,
                offset=local.offset,
                raw=list(local.raw),
                window=list(local.segments),
                alignment=dict(mapping),
                shared_sec=dict(shared),
                minted=list(minted),
                folded=dict(folded),
                timeline_after=self.timeline.runs(),
                changed=changed,
            )
        )

    def fold_count(self) -> int:
        """How many times the cap fallback fired. A non-zero number here is the
        first thing to look at when speakers merge."""
        return sum(len(t.folded) for t in self.traces)
