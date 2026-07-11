import { useCallback, useMemo, useRef, useState, useEffect } from "react";
import { useAudioStream } from "./hooks/useAudioStream";
import { useTranscript } from "./components/TranscriptView.jsx";
import TranscriptView from "./components/TranscriptView.jsx";

const WS_URL =
  (import.meta.env.VITE_WS_URL || "ws://localhost:8000") + "/ws/transcribe";

// The three supported languages each get a label + accent colour.
const LANGS = {
  si: { label: "සිංහල", accent: "#b45309", chip: "rgba(245,165,36,0.14)" },
  ta: { label: "தமிழ்", accent: "#15803d", chip: "rgba(34,197,94,0.14)" },
  en: { label: "English", accent: "#1d4ed8", chip: "rgba(96,165,250,0.16)" },
};
const fallbackLang = {
  label: "?",
  accent: "#525252",
  chip: "rgba(163,163,163,0.15)",
};

const STATUS_TEXT = {
  idle: "Idle",
  connecting: "Connecting…",
  ready: "Listening",
  transcribing: "Transcribing…",
  stopped: "Stopped",
  error: "Connection failed",
};

const SOURCES = [
  { id: "mic", label: "Microphone" },
  { id: "tab", label: "Tab audio" },
];

export default function App() {
  const [source, setSource] = useState("mic");
  const [errors, setErrors] = useState([]);
  const endRef = useRef(null);

  // Use the paragraph-based hook instead of segment-based state
  const { paragraphs, handleMessage, reset } = useTranscript();

  const onMessage = useCallback(
    (data) => {
      if (data.type === "transcript" || data.type === "status") {
        handleMessage(data);
      } else if (data.type === "error") {
        // Errors are separate (not paragraphs)
        setErrors((prev) => [...prev, { ...data, _error: true }]);
      }
    },
    [handleMessage],
  );

  const { start, stop, recording, status, level } = useAudioStream(
    WS_URL,
    onMessage,
  );

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [paragraphs, errors]);

  const wordCount = useMemo(
    () =>
      paragraphs.reduce(
        (n, p) =>
          n + (p.text ? p.text.trim().split(/\s+/).filter(Boolean).length : 0),
        0,
      ),
    [paragraphs],
  );

  // FIX: this used to display paragraphs.length as "N speakers" — but
  // paragraphs are TURNS, not people. A 12-minute two-person interview has
  // dozens of paragraphs. Count UNIQUE speaker labels instead.
  const speakerCount = useMemo(
    () => new Set(paragraphs.map((p) => p.speaker)).size,
    [paragraphs],
  );

  // ---- transcript download (.txt) ----
  // The leading \uFEFF is a UTF-8 BOM: without it, Windows Notepad can
  // misdetect the encoding and render Sinhala/Tamil as mojibake.
  const downloadTxt = useCallback(() => {
    const header =
      `VoxLive transcript — ${new Date().toLocaleString()}\n` +
      `${speakerCount} speaker${speakerCount !== 1 ? "s" : ""} · ` +
      `${paragraphs.length} turn${paragraphs.length !== 1 ? "s" : ""} · ` +
      `${wordCount} words\n` +
      `${"─".repeat(46)}\n\n`;
    const body = paragraphs
      .map(
        (p) =>
          `[${fmtTime(p.start)} – ${fmtTime(p.end)}] ${p.speaker} (${p.language}):\n${p.text}\n`,
      )
      .join("\n");
    const blob = new Blob(["\uFEFF" + header + body], {
      type: "text/plain;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `voxlive-transcript-${tsForFilename()}.txt`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }, [paragraphs, wordCount, speakerCount]);

  const handleNewSession = useCallback(() => {
    reset();
    setErrors([]);
  }, [reset]);

  return (
    <div className="flex h-full flex-col bg-neutral-50 text-neutral-900">
      <header className="flex items-center justify-between border-b border-neutral-200 bg-white px-6 py-4">
        <div className="flex items-baseline gap-3">
          <h1 className="text-lg font-semibold tracking-tight">VoxLive</h1>
          <span className="text-xs text-neutral-500">
            Sinhala · English · Tamil
          </span>
        </div>
        <div className="flex items-center gap-4">
          {recording && <LevelMeter level={level} />}
          <StatusPill status={status} recording={recording} />
        </div>
      </header>

      <main className="flex-1 overflow-y-auto px-6 py-6">
        <div className="mx-auto max-w-3xl">
          {recording && level < 0.004 && (
            <div className="fixed top-20 right-6 z-50 max-w-sm rounded-lg border border-amber-300 bg-amber-50 px-4 py-3 shadow-lg text-xs text-amber-800">
              <strong>Low microphone volume</strong>
              <div className="mt-1">
                Mic: check the input device and volume, and Windows Sound →
                Communications → "Do nothing". Tab audio: make sure the shared
                tab is actually playing.
              </div>
            </div>
          )}
          {paragraphs.length === 0 && errors.length === 0 ? (
            <EmptyState recording={recording} source={source} />
          ) : (
            <div className="space-y-4">
              {/* Errors first */}
              {errors.map((e, i) => (
                <ErrorRow key={`err-${i}`} seg={e} />
              ))}
              {/* Then transcript paragraphs */}
              <TranscriptView paragraphs={paragraphs} />
            </div>
          )}
          <div ref={endRef} />
        </div>
      </main>

      <footer className="border-t border-neutral-200 bg-white px-6 py-4">
        <div className="mx-auto flex max-w-3xl items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <SourceToggle
              source={source}
              setSource={setSource}
              disabled={recording}
            />
            <span className="hidden text-xs text-neutral-400 sm:block">
              {speakerCount} speaker{speakerCount !== 1 ? "s" : ""} ·{" "}
              {paragraphs.length} turn{paragraphs.length !== 1 ? "s" : ""} ·{" "}
              {wordCount} words
            </span>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={downloadTxt}
              disabled={paragraphs.length === 0}
              className="rounded-full border border-neutral-300 bg-white px-5 py-2.5 text-sm font-medium text-neutral-700 transition-colors hover:bg-neutral-100 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Download .txt
            </button>
            <button
              onClick={() => {
                if (recording) {
                  stop();
                } else {
                  handleNewSession();
                  start(source);
                }
              }}
              className={
                "rounded-full px-6 py-2.5 text-sm font-medium transition-colors " +
                (recording
                  ? "bg-red-600 text-white hover:bg-red-700"
                  : "bg-neutral-900 text-white hover:bg-neutral-800")
              }
            >
              {recording ? "Stop" : "Start recording"}
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
}

function SourceToggle({ source, setSource, disabled }) {
  return (
    <div
      className={
        "flex rounded-full border border-neutral-300 bg-neutral-100 p-0.5 " +
        (disabled ? "opacity-50" : "")
      }
      title="Tab audio captures a browser tab digitally — use it to transcribe played clips (YouTube, recordings) without speaker/mic quality loss"
    >
      {SOURCES.map((s) => (
        <button
          key={s.id}
          onClick={() => !disabled && setSource(s.id)}
          disabled={disabled}
          className={
            "rounded-full px-3.5 py-1.5 text-xs font-medium transition-colors " +
            (source === s.id
              ? "bg-white text-neutral-900 shadow-sm"
              : "text-neutral-500 hover:text-neutral-700")
          }
        >
          {s.label}
        </button>
      ))}
    </div>
  );
}

function LevelMeter({ level }) {
  // Map RMS (speech is usually ~0.01–0.2) onto a readable 0–100% bar.
  const pct = Math.min(100, Math.round(Math.sqrt(level) * 260));
  return (
    <div
      className="flex items-center gap-2"
      title="Input level — if this doesn't move, no audio is reaching VoxLive"
    >
      <span className="text-[10px] uppercase tracking-wide text-neutral-400">
        Input
      </span>
      <div className="h-1.5 w-24 overflow-hidden rounded-full bg-neutral-200">
        <div
          className={
            "h-full rounded-full transition-[width] duration-100 " +
            (pct < 8 ? "bg-neutral-400" : "bg-emerald-500")
          }
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

function StatusPill({ status, recording }) {
  const live = recording && (status === "ready" || status === "transcribing");
  return (
    <div className="flex items-center gap-2 text-xs text-neutral-500">
      <span
        className={
          "inline-block h-2 w-2 rounded-full " +
          (status === "error"
            ? "bg-amber-500"
            : live
              ? "bg-red-500 animate-pulse"
              : "bg-neutral-300")
        }
      />
      {STATUS_TEXT[status] || status}
    </div>
  );
}

function ErrorRow({ seg }) {
  return (
    <li className="rounded-lg border border-red-200 bg-red-50 p-3">
      <p className="text-xs text-red-700">
        ⚠{" "}
        {seg.segment_id != null
          ? `Segment ${seg.segment_id} failed to transcribe — `
          : ""}
        {seg.message}
      </p>
    </li>
  );
}

function EmptyState({ recording, source }) {
  return (
    <div className="mt-24 text-center text-neutral-400">
      <p className="text-sm">
        {recording
          ? "Listening… text appears after each pause."
          : source === "tab"
            ? "Press Start recording, pick the browser tab that will play the audio, and tick \u201CAlso share tab audio\u201D."
            : "Press Start recording. Transcripts finalize on natural pauses."}
      </p>
    </div>
  );
}

function fmtTime(sec) {
  const s = Math.max(0, Math.floor(sec));
  const m = Math.floor(s / 60);
  return `${m}:${String(s % 60).padStart(2, "0")}`;
}

function tsForFilename() {
  const d = new Date();
  const p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}_${p(
    d.getHours(),
  )}-${p(d.getMinutes())}`;
}
