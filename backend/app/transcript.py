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


@dataclass
class TranscriptStore:
    chunks: list[Chunk] = field(default_factory=list)
    _next_id: int = 1
    _emitted: dict[int, dict] = field(default_factory=dict)

    # ----------------------------------------------------------------- chunks

    def new_chunk(self, seg_id: int, start: float, end: float) -> Chunk:
        """Reserve a chunk id BEFORE transcription.

        Reserving early is what makes the id stable: it reflects creation
        order, which is the audio clock, not ASR completion order, which is
        whatever Gemini's queue felt like.
        """
        c = Chunk(chunk_id=self._next_id, seg_id=seg_id, start=start, end=end)
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

    def relabel(self, lookup) -> bool:
        """Re-derive every chunk's speaker from the diarization timeline.

        `lookup(start, end) -> speaker_id | None`. Returns True if anything
        moved. Called after every diarization pass, so the transcript converges
        on the timeline rather than being frozen at whatever was known when the
        text happened to arrive.
        """
        changed = False
        last = None
        for c in sorted(self.chunks, key=lambda x: (x.start, x.chunk_id)):
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

    # ------------------------------------------------------------- paragraphs

    def paragraphs(self) -> list[dict]:
        """Group consecutive same-speaker chunks. Ids come from the head chunk."""
        out: list[dict] = []
        for c in sorted(self.chunks, key=lambda x: (x.start, x.chunk_id)):
            if not c.text:
                continue
            spk = c.speaker if c.speaker is not None else 0
            if out and out[-1]["_spk"] == spk:
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
            p["language"] = langs[0] if len(langs) <= 1 else "+".join(langs)
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
