/**
 * TranscriptView.jsx — paragraph upsert + refresh (v10).
 *
 * WHAT CHANGED, AND WHY IT MATTERS
 * --------------------------------
 * paragraph_id used to be the paragraph's POSITION in the list, so it was
 * renumbered whenever an ASR result arrived out of order. It is now the
 * chunk_id of the paragraph's first chunk: stable for the life of the
 * session, but NO LONGER CONTIGUOUS OR SORTED.
 *
 * So this file sorts by `start` (the audio clock) instead of by paragraph_id.
 * Sorting by id was correct under the old scheme and is subtly wrong under
 * the new one — it would put a late-arriving early turn at the bottom.
 *
 * TWO MESSAGE BEHAVIOURS:
 *
 * 1. type:"transcript" — the backend re-sends the FULL accumulated paragraph
 *    (same paragraph_id, growing text) while one speaker keeps talking.
 *    UPSERT by paragraph_id, never append.
 *
 * 2. type:"refresh" — the whole corrected paragraph list. Sent whenever the
 *    transcript changes shape: a speaker corrected by a later diarization
 *    pass, two paragraphs merging because they turned out to be one person,
 *    or a late chunk landing mid-transcript. REPLACE the map wholesale.
 *
 * Usage:
 *   const { paragraphs, speakers, handleMessage, reset } = useTranscript();
 *   ws.onmessage = (e) => handleMessage(JSON.parse(e.data));
 *   <TranscriptView paragraphs={paragraphs} />
 */

import { Fragment, useCallback, useMemo, useState } from "react";

const LANG_LABEL = { si: "සිංහල", en: "English", ta: "தமிழ்" };

// Per-language tint for code-switched turns (Phase 4). Deliberately faint:
// the transcript is for READING, and a turn that switches five times must not
// look like a ransom note. Colour is a secondary cue only — the `title`
// attribute names the language, so this never carries meaning by colour alone.
const LANG_TINT = {
  si: "rgba(37,99,235,0.10)",
  ta: "rgba(217,119,6,0.14)",
  en: "transparent",
};

// Distinct, colour-blind-safe accents so two speakers never read as one.
const SPEAKER_COLORS = [
  { bg: "rgba(37,99,235,0.10)", fg: "#1d4ed8", bar: "#2563eb" },
  { bg: "rgba(217,119,6,0.12)", fg: "#b45309", bar: "#d97706" },
  { bg: "rgba(22,163,74,0.10)", fg: "#15803d", bar: "#16a34a" },
  { bg: "rgba(147,51,234,0.10)", fg: "#7e22ce", bar: "#9333ea" },
  { bg: "rgba(220,38,38,0.10)", fg: "#b91c1c", bar: "#dc2626" },
  { bg: "rgba(13,148,136,0.10)", fg: "#0f766e", bar: "#0d9488" },
];

function speakerStyle(label) {
  const n = parseInt(String(label).replace(/\D+/g, ""), 10);
  return SPEAKER_COLORS[
    (Number.isFinite(n) ? n - 1 : 0) % SPEAKER_COLORS.length
  ];
}

/**
 * Render one paragraph's text split by `language_spans`.
 *
 * Spans carry start_char/end_char indexing `text` as a JS slice, so this is a
 * plain substring walk — no re-detection on the client, and no chance of the
 * client and server disagreeing about where a switch happened.
 *
 * The gaps BETWEEN spans are whitespace (the server guarantees contiguity),
 * and they are emitted verbatim so the sentence still reads normally. A
 * paragraph with zero or one span renders as plain text: a monolingual turn
 * should look exactly as it did before Phase 4.
 */
function CodeSwitchedText({ text, spans }) {
  if (!spans || spans.length < 2) {
    return <p className="leading-relaxed text-neutral-900">{text}</p>;
  }

  const parts = [];
  let cursor = 0;
  spans.forEach((s, i) => {
    // Whitespace between the previous span and this one.
    if (s.start_char > cursor) {
      parts.push(<span key={`g${i}`}>{text.slice(cursor, s.start_char)}</span>);
    }
    parts.push(
      <span
        key={`s${i}`}
        title={LANG_LABEL[s.language] ?? s.language}
        style={{
          background: LANG_TINT[s.language] ?? "transparent",
          borderRadius: "2px",
        }}
      >
        {text.slice(s.start_char, s.end_char)}
      </span>,
    );
    cursor = s.end_char;
  });
  // Anything after the final span (trailing whitespace, or text the server
  // could not attribute). Never dropped: the rendered string must always equal
  // `text` exactly, whatever the spans say.
  if (cursor < text.length) {
    parts.push(<span key="tail">{text.slice(cursor)}</span>);
  }
  return <p className="leading-relaxed text-neutral-900">{parts}</p>;
}

function langLabel(code) {
  // A code-switched paragraph reports e.g. "si+en".
  return String(code || "")
    .split("+")
    .map((c) => LANG_LABEL[c] ?? c)
    .join(" + ");
}

function fmtTime(sec) {
  const s = Math.max(0, Math.round(sec));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

export function useTranscript() {
  const [map, setMap] = useState({});
  const [speakers, setSpeakers] = useState(0);

  const handleMessage = useCallback((msg) => {
    if (msg?.type === "refresh") {
      const next = {};
      for (const p of msg.paragraphs || []) next[p.paragraph_id] = p;
      setMap(next); // REPLACE — ids are renumbered when paragraphs merge
      return;
    }
    if (msg?.type === "speakers") {
      setSpeakers(msg.count || 0);
      return;
    }
    if (msg?.type !== "transcript") return;
    setMap((prev) => ({ ...prev, [msg.paragraph_id]: msg }));
  }, []);

  const reset = useCallback(() => {
    setMap({});
    setSpeakers(0);
  }, []);

  // Order by the AUDIO CLOCK. paragraph_id is stable but not ordered.
  const paragraphs = useMemo(
    () =>
      Object.values(map).sort(
        (a, b) => a.start - b.start || a.paragraph_id - b.paragraph_id,
      ),
    [map],
  );

  return { paragraphs, speakers, handleMessage, reset };
}

export default function TranscriptView({ paragraphs }) {
  if (!paragraphs || paragraphs.length === 0) {
    return (
      <p className="text-sm italic text-neutral-400">
        Transcript will appear here…
      </p>
    );
  }

  return (
    <div className="space-y-3">
      {paragraphs.map((p, i) => {
        const c = speakerStyle(p.speaker);
        const prev = i > 0 ? paragraphs[i - 1] : null;
        // A RECORDING boundary. Speaker numbers reset when the user starts a
        // new recording, so "Speaker 1" below the divider is a different human
        // from "Speaker 1" above it. Without this line the transcript would
        // show two people under one name with nothing to separate them.
        const rec = p.recording ?? 1;
        const isNewRecording = prev != null && (prev.recording ?? 1) !== rec;
        // Only re-announce the speaker when it actually changes: a wall of
        // repeated name badges makes a two-person conversation unreadable.
        const isNewSpeaker =
          i === 0 || isNewRecording || prev.speaker !== p.speaker;
        return (
          <Fragment key={p.paragraph_id}>
            {isNewRecording && (
              <div className="flex items-center gap-3 pt-2" role="separator">
                <span className="h-px flex-1 bg-neutral-200" />
                <span className="text-[11px] font-medium uppercase tracking-wide text-neutral-400">
                  Recording {rec} · speakers renumbered
                </span>
                <span className="h-px flex-1 bg-neutral-200" />
              </div>
            )}
            <div
              className="rounded-lg border border-neutral-200 bg-white p-4 shadow-sm"
              style={{ borderLeft: `3px solid ${c.bar}` }}
            >
              <div className="mb-2 flex items-center gap-2">
                <span
                  className="rounded px-1.5 py-0.5 text-[11px] font-semibold"
                  style={{ background: c.bg, color: c.fg }}
                >
                  {p.speaker}
                </span>
                {isNewSpeaker && (
                  <span className="text-[11px] text-neutral-400">
                    {langLabel(p.language)}
                  </span>
                )}
                <span className="ml-auto text-[11px] tabular-nums text-neutral-400">
                  {fmtTime(p.start)} – {fmtTime(p.end)}
                </span>
              </div>
              <CodeSwitchedText text={p.text} spans={p.language_spans} />
            </div>
          </Fragment>
        );
      })}
    </div>
  );
}
