"""
WebSocket message contract (server -> client).

status:      {type:"status", state:"ready|transcribing|stopped"}
transcript:  {type:"transcript", paragraph_id, segment_id, speaker, language,
              text, start, end, final}
error:       {type:"error", segment_id, message}

PARAGRAPH GROUPING
  Consecutive segments from the SAME speaker share one paragraph_id, and each
  message carries the FULL accumulated paragraph text so far (not just the new
  segment). The frontend must therefore UPSERT by paragraph_id:

      // React sketch — paragraphs is an ordered map keyed by paragraph_id
      onMessage(msg) {
        if (msg.type !== "transcript") return;
        setParagraphs(prev => ({ ...prev, [msg.paragraph_id]: msg }));
      }
      // render: Object.values(paragraphs)
      //           .sort((a, b) => a.paragraph_id - b.paragraph_id)
      //           .map(p => <p key={p.paragraph_id}>
      //                       <b>{p.speaker}:</b> {p.text}
      //                     </p>)

  Result: "Speaker 1: <everything they said in that turn as one paragraph>",
  then a new paragraph when the speaker changes.
"""


def status_msg(state: str) -> dict:
    return {"type": "status", "state": state}


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
