/**
 * TranscriptView.jsx — drop-in fix for "same paragraph repeated in separate
 * growing lines" bug.
 *
 * THE BUG IS IN THE FRONTEND, NOT THE BACKEND. The backend intentionally
 * re-sends the FULL accumulated paragraph every time the same speaker keeps
 * talking (that's what makes it live). Each of those messages carries the
 * SAME paragraph_id. If the UI appends every message, you get:
 *     Speaker 1  0:12–0:20  <partial>
 *     Speaker 1  0:12–0:28  <longer partial>
 *     Speaker 1  0:12–0:48  <full>
 *
 * The fix: key paragraphs by paragraph_id and REPLACE (upsert). Then one
 * paragraph per speaker turn grows in place and ends as a single block:
 *     Speaker 1  0:12–0:48  <full paragraph>
 *
 * Usage:
 *   const { paragraphs, handleMessage, reset } = useTranscript();
 *   ws.onmessage = (e) => handleMessage(JSON.parse(e.data));
 *   ...
 *   <TranscriptView paragraphs={paragraphs} />
 */

import { useCallback, useState } from "react";

const LANG_LABEL = { si: "සිංහල", en: "English", ta: "தமிழ்" };

function fmtTime(sec) {
  const s = Math.max(0, Math.round(sec));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/**
 * Hook that manages paragraphs (upserts by paragraph_id).
 * Returns: { paragraphs, handleMessage, reset }
 *   - paragraphs: ordered array of paragraph objects
 *   - handleMessage: call this with each WebSocket message
 *   - reset: clears all paragraphs (for new session)
 */
export function useTranscript() {
  const [paragraphsMap, setParagraphsMap] = useState({}); // paragraph_id -> msg

  const handleMessage = useCallback((msg) => {
    if (msg?.type !== "transcript") return;
    // UPSERT: replace the paragraph with the latest full version.
    // The backend sends the SAME paragraph_id with growing text as the
    // speaker continues, so this replaces the old entry with the new one.
    setParagraphsMap((prev) => ({
      ...prev,
      [msg.paragraph_id]: msg,
    }));
  }, []);

  const reset = useCallback(() => setParagraphsMap({}), []);

  // Sort by paragraph_id to maintain speaking order
  const ordered = Object.values(paragraphsMap).sort(
    (a, b) => a.paragraph_id - b.paragraph_id,
  );

  return { paragraphs: ordered, handleMessage, reset };
}

/**
 * Component that renders the transcript.
 * Props: { paragraphs: array of paragraph objects from useTranscript }
 */
export default function TranscriptView({ paragraphs }) {
  if (!paragraphs || paragraphs.length === 0) {
    return (
      <p className="text-sm text-gray-400 italic">
        Transcript will appear here…
      </p>
    );
  }

  return (
    <div className="space-y-4">
      {paragraphs.map((p) => (
        <div
          key={p.paragraph_id}
          className="rounded-lg border border-neutral-200 bg-white p-4 shadow-sm"
        >
          <div className="mb-2 flex items-center gap-2">
            <span className="text-sm font-medium text-neutral-700">
              {p.speaker}
            </span>
            <span
              className="rounded px-1.5 py-0.5 text-[11px] font-medium"
              style={{
                background: "rgba(96,165,250,0.16)",
                color: "#1d4ed8", // default; override per language below
              }}
            >
              {LANG_LABEL[p.language] ?? p.language}
            </span>
            <span className="ml-auto text-[11px] tabular-nums text-neutral-400">
              {fmtTime(p.start)} – {fmtTime(p.end)}
            </span>
          </div>
          <p className="leading-relaxed text-neutral-900">{p.text}</p>
        </div>
      ))}
    </div>
  );
}
