"""
transcript.py — chunk-sourced transcript with STABLE paragraph ids (v10).

THE BUG THIS FIXES
------------------
v7's TranscriptStore numbered paragraphs by POSITION:

    paragraph_id = len(paras) + 1

and `add()` re-sorted all chunks by audio start time on every insertion. ASR
runs with ASR_CONCURRENCY=6, so chunks complete OUT OF ORDER: a chunk from
0:12 can land after a chunk from 0:15. When it does, it is inserted earlier in
the list and every paragraph after it is renumbered.

The client upserts by paragraph_id. So paragraph 4's text arrives under id 5,
id 4 keeps its now-stale content, and the transcript on screen duplicates
lines, attaches text to the wrong speaker, and generally scrambles — while the
server-side data was correct the whole time. Only a `refresh` repaired it, and
refreshes were only sent when a re-cluster happened to change a label.

A lot of what looks like "the speaker detection is wrong" is this. It is a
message-protocol bug, not a diarization bug, and it survives any amount of
diarization tuning.

THE FIX
Paragraph identity is derived from data, not position: a paragraph is named
after the chunk_id of its FIRST chunk. Chunk ids are handed out monotonically
when a chunk is created (before ASR), so they never move. Late arrivals now
land in the right paragraph, and a paragraph that genuinely changes shape is
detected by diffing and pushed as a `refresh`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("voxlive.transcript")


@dataclass
class Chunk:
    chunk_id: int
    seg_id: int
    start: float
    end: float
    text: str = ""
    language: str = ""
    speaker: Optional[int] = None  # 0-based; None = not diarized yet
    # Which RECORDING inside this session the chunk belongs to. Bumped by the
    # explicit new-recording control, never by silence. Speaker numbers are
    # only comparable WITHIN one recording: "Speaker 1" in recording 2 is a
    # different human from "Speaker 1" in recording 1, because the identity
    # registry was reset between them.
    recording: int = 1


@dataclass
class TranscriptStore:
    chunks: list[Chunk] = field(default_factory=list)
    _next_id: int = 1
    _emitted: dict[int, dict] = field(default_factory=dict)

    # ----------------------------------------------------------------- chunks

    def new_chunk(
        self, seg_id: int, start: float, end: float, recording: int = 1
    ) -> Chunk:
        """Reserve a chunk id BEFORE transcription.

        Reserving early is what makes the id stable: it reflects creation
        order, which is the audio clock, not ASR completion order, which is
        whatever Gemini's queue felt like.
        """
        c = Chunk(
            chunk_id=self._next_id,
            seg_id=seg_id,
            start=start,
            end=end,
            recording=recording,
        )
        self._next_id += 1
        self.chunks.append(c)
        return c

    def drop(self, chunk: Chunk) -> None:
        """Discard a chunk that produced no usable text."""
        try:
            self.chunks.remove(chunk)
        except ValueError:
            pass

    # ------------------------------------------------------------- relabeling

    def relabel(self, lookup, recording: Optional[int] = None) -> bool:
        """Re-derive every chunk's speaker from the diarization timeline.

        `lookup(start, end) -> speaker_id | None`. Returns True if anything
        moved. Called after every diarization pass, so the transcript converges
        on the timeline rather than being frozen at whatever was known when the
        text happened to arrive.

        `recording` SCOPES the update. The diarizer only holds identities for
        the recording currently in progress — the new-recording control clears
        the rest — so relabelling everything would drag earlier recordings'
        chunks against a timeline that no longer describes them. Passing the
        current recording index leaves finished recordings alone, which is the
        whole point of giving each one its own speaker universe.
        """
        changed = False
        last = None
        rows = [c for c in self.chunks if recording is None or c.recording == recording]
        for c in sorted(rows, key=lambda x: (x.start, x.chunk_id)):
            s = lookup(c.start, c.end)
            if s is None:
                # No timeline coverage (silence-only or not yet diarized).
                # Continuing the previous speaker beats inventing one.
                s = last
            if s is not None and c.speaker != s:
                c.speaker = s
                changed = True
            if s is not None:
                last = s
        return changed

    def unsplit_chunks(self, timeline) -> tuple[int, float]:
        """Chunks the timeline says hold a speaker change. `(count, seconds)`.

        A DIAGNOSTIC, not a repair. Chunk boundaries are decided once, before
        ASR, from whatever the timeline knew then; `relabel()` can move a chunk
        to a different speaker but cannot divide one. So if the final timeline
        shows a turn change INSIDE a chunk, that chunk's text is a blend of two
        people and will be shown under a single name no matter how good the
        clustering gets.

        The text cannot be split retroactively — Gemini returns a string with
        no word timings, so there is nothing to cut it on. What this number is
        for is telling you how much of the diarization error you are looking at
        is clustering (fixable by tuning) versus segmentation (fixable only by
        producing shorter chunks in the first place, via SILENCE_MS,
        SOFT_MAX_SEGMENT_MS and MAX_SEGMENT_MS).
        """
        runs = [r for r in timeline if r[2] is not None]
        if not runs:
            return 0, 0.0
        count = 0
        seconds = 0.0
        for c in self.chunks:
            if not c.text:
                continue
            inside = {s for a, b, s in runs if b > c.start + 0.35 and a < c.end - 0.35}
            if len(inside) > 1:
                count += 1
                seconds += c.end - c.start
        return count, round(seconds, 1)

    # ------------------------------------------------------------- paragraphs

    def paragraphs(self) -> list[dict]:
        """Group consecutive same-speaker chunks. Ids come from the head chunk."""
        out: list[dict] = []
        for c in sorted(self.chunks, key=lambda x: (x.start, x.chunk_id)):
            if not c.text:
                continue
            spk = c.speaker if c.speaker is not None else 0
            # A recording boundary always breaks the paragraph, even when the
            # speaker NUMBER matches: the numbers are not comparable across
            # recordings, so merging them would silently glue two different
            # people into one turn.
            if out and out[-1]["_spk"] == spk and out[-1]["recording"] == c.recording:
                p = out[-1]
                p["text"] = f"{p['text']} {c.text}".strip()
                p["end"] = round(c.end, 2)
                p["segment_id"] = c.seg_id
                if c.language and c.language not in p["_langs"]:
                    p["_langs"].append(c.language)
            else:
                out.append(
                    {
                        "type": "transcript",
                        "paragraph_id": c.chunk_id,  # STABLE
                        "segment_id": c.seg_id,
                        "speaker": f"Speaker {spk + 1}",
                        "language": c.language,
                        "text": c.text,
                        "start": round(c.start, 2),
                        "end": round(c.end, 2),
                        "final": True,
                        "recording": c.recording,
                        "_spk": spk,
                        "_langs": [c.language] if c.language else [],
                    }
                )

        for p in out:
            langs = p.pop("_langs")
            p.pop("_spk")
            # A paragraph can legitimately be code-switched: Sinhala, then an
            # English clause, then back. Report the mix rather than pretending
            # the last chunk's language was the whole paragraph's.
            #
            # `langs` can be EMPTY: it is only appended to when a chunk carries
            # a non-empty `language`, so a paragraph built entirely from chunks
            # with a blank language left this list empty and `langs[0]` raised
            # IndexError — taking down the whole transcript render, not just
            # that paragraph. The normal ASR path always sets a language, which
            # is why this has not fired in production; anything that sets text
            # without one (a test, a future provider, a postprocess stage)
            # would hit it.
            p["language"] = langs[0] if len(langs) == 1 else "+".join(langs)
        return out

    def diff(self) -> tuple[list[dict], str]:
        """Return (paragraphs, mode) where mode is 'append' or 'refresh'.

        'append' means exactly one paragraph changed and it is the last one, so
        a single `transcript` message is enough. Anything else — a late chunk
        landing mid-transcript, two paragraphs merging after a relabel, a
        paragraph splitting in two — needs a wholesale `refresh`, because the
        client cannot reconstruct those from an upsert.
        """
        paras = self.paragraphs()
        cur = {p["paragraph_id"]: p for p in paras}
        prev = self._emitted

        changed = [pid for pid in set(cur) | set(prev) if cur.get(pid) != prev.get(pid)]
        self._emitted = {pid: dict(p) for pid, p in cur.items()}

        if not changed:
            return paras, "none"
        if len(changed) == 1 and paras and changed[0] == paras[-1]["paragraph_id"]:
            return paras, "append"
        return paras, "refresh"

    def reset(self) -> None:
        self.chunks.clear()
        self._emitted.clear()
        self._next_id = 1
