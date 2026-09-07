"""
WebSocket message contract (server -> client).

status:      {type:"status", state:"ready|transcribing|stopped"}
transcript:  {type:"transcript", paragraph_id, segment_id, speaker, language,
              text, start, end, final, recording, language_spans}
refresh:     {type:"refresh", paragraphs:[<transcript messages>]}
speakers:    {type:"speakers", count:int}
error:       {type:"error", segment_id, message}

LANGUAGE_SPANS (new — Phase 4, code-switching)
  `language` remains a SUMMARY: a single code ("si"), or "+"-joined ("si+en+ta")
  when a turn mixes languages. Keep using it for colouring and filtering.

  `language_spans` is where the mixing actually lives — an ordered, contiguous,
  non-overlapping list covering the turn:

      [{"language": "si", "start_char": 0,  "end_char": 4,
        "start": null, "end": null, "timing": "none"},
       {"language": "en", "start_char": 5,  "end_char": 40, ...},
       {"language": "ta", "start_char": 60, "end_char": 76, ...}]

  start_char/end_char index `text` as a Python-style slice, so
  text[start_char:end_char] is exactly the span. They are derived from Unicode
  script ranges (Sinhala U+0D80-0DFF, Tamil U+0B80-0BFF, Latin), which are
  disjoint for these three languages — so the offsets are EXACT, not predicted.

  start/end are absolute session SECONDS and are `null` unless the ASR provider
  supplied word-level timings. `timing` says which: "none" (no timings
  available) or "api" (from the provider's own word annotations). They are
  never interpolated from character position — Sinhala and Tamil agglutinate
  and English does not, so a constant chars-per-second assumption is biased
  precisely at the switch boundaries where these times would be used.

  A client must treat `null` timings as absent, not as zero. An empty list is
  also valid and means the text carries no script at all (a bare number).

  Spans do not survive a `refresh` unchanged: they are recomputed from the
  chunks each time, so always take them from the message rather than caching.

RECORDING (new)
  `recording` is a 1-based index of which recording inside this session the
  paragraph belongs to. It only changes when the user presses NEW RECORDING,
  never on silence.

  It exists because speaker numbers are ONLY comparable within one recording.
  The control resets the diarizer's identity registry, so "Speaker 1" in
  recording 2 is a different human from "Speaker 1" in recording 1. A client
  that ignores this field will show two different people under one name with
  nothing to separate them, so render a divider when the value changes.

PARAGRAPH IDENTITY (v10 — changed, read this)
  paragraph_id is the chunk_id of the paragraph's FIRST chunk. It is assigned
  in audio order before transcription and never renumbered.

  v7 numbered paragraphs by position in the list, which meant an out-of-order
  ASR completion silently shifted every id after it and the client's upsert
  landed in the wrong slot. If you are porting a client, the important
  consequence is that ids are now STABLE but NOT CONTIGUOUS — do not assume
  1,2,3. Sort by `start`, not by paragraph_id.

REFRESH
  Speaker labels are attached to text by timestamp, and the diarizer re-derives
  the whole session's timeline every pass. Whenever that changes the shape of
  the transcript — a speaker corrected, two paragraphs merging because they
  turned out to be the same person, a late chunk landing mid-transcript — the
  server sends the complete corrected list. The client must REPLACE its
  paragraph map with refresh.paragraphs, not merge into it.

  A plain `transcript` message is only sent when exactly one paragraph changed
  and it is the last one. Everything else is a refresh.
"""


def status_msg(state: str) -> dict:
    return {"type": "status", "state": state}


def refresh_msg(paragraphs: list) -> dict:
    return {"type": "refresh", "paragraphs": paragraphs}


def speakers_msg(count: int) -> dict:
    return {"type": "speakers", "count": count}


def error_msg(message: str, segment_id: int | None = None) -> dict:
    return {"type": "error", "segment_id": segment_id, "message": message}


def transcript_msg(
    segment_id: int,
    speaker: str,
    language: str,
    text: str,
    start: float,
    end: float,
    paragraph_id: int = 0,
    final: bool = True,
) -> dict:
    return {
        "type": "transcript",
        "paragraph_id": paragraph_id,
        "segment_id": segment_id,
        "speaker": speaker,
        "language": language,
        "text": text,
        "start": start,
        "end": end,
        "final": final,
    }
