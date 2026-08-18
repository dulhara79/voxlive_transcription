"""
WebSocket message contract (server -> client).

status:      {type:"status", state:"ready|transcribing|stopped"}
transcript:  {type:"transcript", paragraph_id, segment_id, speaker, language,
              text, start, end, final, recording}
refresh:     {type:"refresh", paragraphs:[<transcript messages>]}
speakers:    {type:"speakers", count:int}
error:       {type:"error", segment_id, message}

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
