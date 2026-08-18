import { useCallback, useMemo, useRef, useState, useEffect } from "react";
import { useAudioStream } from "./hooks/useAudioStream";
import { useTranscript } from "./components/TranscriptView.jsx";
import TranscriptView from "./components/TranscriptView.jsx";
import { useAuth } from "./auth/AuthContext.jsx";

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

// SPEAKER COUNT — semantics changed (supervisor review, Fix #1).
//
// It used to mean a CEILING: picking 2 told the backend "at most two", which
// left it free to decide, several minutes into a two-person interview, that
// the voices were similar enough to be one person — and then relabel the whole
// transcript Speaker 1. For a recording where the user KNOWS there are two
// people, that is the wrong contract.
//
//   Auto (0)  estimate the number of speakers
//   2/3/4...  assume EXACTLY that many. The backend will not collapse them.
//
// Auto stays the default because the exact-count mode has a real cost, and it
// is the mirror image of the bug it fixes: if you say 2 and only one person
// speaks, that person WILL be split into two speakers. The selector says so.
const SPEAKER_CHOICES = [0, 2, 3, 4, 5, 6];

export default function App() {
  const { user, signOut } = useAuth();
  const [source, setSource] = useState("mic");
  const [expectedSpeakers, setExpectedSpeakers] = useState(0);
  const [errors, setErrors] = useState([]);
  const endRef = useRef(null);

  // Paragraph-based transcript state (upsert by paragraph_id)
  const {
    paragraphs,
    speakers: detectedSpeakers,
    handleMessage,
    reset,
  } = useTranscript();

  const onMessage = useCallback(
    (data) => {
      // This allow-list has to include every message useTranscript() can
      // handle. "refresh" was missing once and the backend's retroactive
      // speaker correction was computed, sent, and silently discarded —
      // nothing on screen ever improved. "speakers" is new in v10.
      if (
        data.type === "transcript" ||
        data.type === "status" ||
        data.type === "refresh" ||
        data.type === "speakers"
      ) {
        handleMessage(data);
      } else if (data.type === "error") {
        setErrors((prev) => [...prev, { ...data, _error: true }]);
      } else if (data.type === "rejected") {
        // Admission control refused the session: the platform is at capacity,
        // or this organization has used every concurrent session on its plan.
        // Without this branch the frame is silently dropped and the user sees
        // nothing at all, which is worse than any error message.
        const retry = data.retry_after_sec
          ? ` Try again in about ${data.retry_after_sec}s.`
          : "";
        const usage =
          data.limit > 0 ? ` (${data.current}/${data.limit} in use)` : "";
        setErrors((prev) => [
          ...prev,
          {
            ...data,
            _error: true,
            message: (data.message || "Session refused.") + usage + retry,
          },
        ]);
      }
    },
    [handleMessage],
  );

  const { start, stop, newRecording, recording, status, level } =
    useAudioStream(WS_URL, onMessage);

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

  // Prefer the server's count: it comes from the clustering itself. Counting
  // distinct labels on the client over-reports during the window between a
  // wrong label being rendered and the refresh that corrects it.
  const speakerCount = useMemo(
    () => detectedSpeakers || new Set(paragraphs.map((p) => p.speaker)).size,
    [detectedSpeakers, paragraphs],
  );

  // ---- unsaved-work tracking -------------------------------------------
  // A transcript exists only in this tab until it is downloaded, so a reload,
  // a closed tab, or starting a second recording destroys it permanently. The
  // signature is the transcript's content, not a boolean flag: downloading and
  // then speaking again correctly counts as unsaved once more.
  const signature = useMemo(
    () => `${paragraphs.length}:${wordCount}`,
    [paragraphs.length, wordCount],
  );
  const [savedSignature, setSavedSignature] = useState("");
  const [confirming, setConfirming] = useState(null); // null | "new"

  const hasUnsaved =
    recording || (paragraphs.length > 0 && signature !== savedSignature);

  useEffect(() => {
    if (!hasUnsaved) return;
    const warn = (e) => {
      // Browsers ignore custom text and show their own wording, but they only
      // prompt at all if the handler both preventDefault()s and sets
      // returnValue. Chrome additionally requires a prior interaction with the
      // page, which pressing Start already satisfies.
      e.preventDefault();
      e.returnValue = "";
      return "";
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [hasUnsaved]);

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
    setSavedSignature(signature);
  }, [paragraphs, wordCount, speakerCount, signature]);

  const beginSession = useCallback(() => {
    reset();
    setErrors([]);
    setSavedSignature("");
    setConfirming(null);
    // expectedSpeakers > 0 means the user asserted a count, so the backend runs
    // in fixed mode and holds exactly that many identities. Auto (0) estimates.
    start(source, expectedSpeakers, expectedSpeakers > 0 ? "fixed" : "auto");
  }, [reset, start, source, expectedSpeakers]);

  // Explicit NEW RECORDING (supervisor review §10/§11). Deliberately a button
  // and not a silence timer: only the user knows whether a gap was a pause or
  // the end of a clip, and guessing wrong invents speakers mid-interview.
  const onNewRecording = useCallback(() => {
    if (!newRecording()) {
      setErrors((e) => [...e, "Not connected — cannot start a new recording."]);
    }
  }, [newRecording]);

  const onPrimaryClick = useCallback(() => {
    if (recording) {
      stop();
      return;
    }
    // Starting over wipes the previous transcript. Ask first, and offer the
    // download rather than just blocking the action.
    if (paragraphs.length > 0 && signature !== savedSignature) {
      setConfirming("new");
      return;
    }
    beginSession();
  }, [
    recording,
    stop,
    paragraphs.length,
    signature,
    savedSignature,
    beginSession,
  ]);

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
          <AccountMenu user={user} onSignOut={signOut} recording={recording} />
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
              {errors.map((e, i) => (
                <ErrorRow key={`err-${i}`} seg={e} />
              ))}
              <TranscriptView paragraphs={paragraphs} />
            </div>
          )}
          <div ref={endRef} />
        </div>
      </main>

      <footer className="border-t border-neutral-200 bg-white px-6 py-4">
        <div className="mx-auto flex max-w-3xl items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <SourceToggle
              source={source}
              setSource={setSource}
              disabled={recording}
            />
            <SpeakerSelect
              speakers={expectedSpeakers}
              setSpeakers={setExpectedSpeakers}
              disabled={recording}
            />
            {recording && (
              <button
                onClick={onNewRecording}
                title={
                  "Start a new recording without stopping the session.\n\n" +
                  "Speaker identities start fresh from here, so the next clip's " +
                  "Speaker 1 is a different person from this one's. The " +
                  "transcript above is kept.\n\n" +
                  "Use this between independent clips. A pause in a " +
                  "conversation is NOT a new recording — don't press it when " +
                  "someone is just thinking."
                }
                className="rounded-full border border-neutral-300 bg-white px-3 py-1.5 text-xs font-medium text-neutral-700 transition-colors hover:bg-neutral-100"
              >
                New recording
              </button>
            )}
            <span className="hidden text-xs text-neutral-400 sm:block">
              {speakerCount} speaker{speakerCount !== 1 ? "s" : ""} ·{" "}
              {paragraphs.length} turn{paragraphs.length !== 1 ? "s" : ""} ·{" "}
              {wordCount} words
            </span>
          </div>
          <div className="flex items-center gap-3">
            {paragraphs.length > 0 && signature !== savedSignature && (
              <span
                className="flex items-center gap-1.5 text-xs text-amber-700"
                title="This transcript only exists in this tab until you download it"
              >
                <span className="inline-block h-1.5 w-1.5 rounded-full bg-amber-500" />
                Not downloaded
              </span>
            )}
            <button
              onClick={downloadTxt}
              disabled={paragraphs.length === 0}
              className="rounded-full border border-neutral-300 bg-white px-5 py-2.5 text-sm font-medium text-neutral-700 transition-colors hover:bg-neutral-100 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Download .txt
            </button>
            <button
              onClick={onPrimaryClick}
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

      {confirming === "new" && (
        <ConfirmDiscard
          turns={paragraphs.length}
          words={wordCount}
          onDownload={() => {
            downloadTxt();
            beginSession();
          }}
          onDiscard={beginSession}
          onCancel={() => setConfirming(null)}
        />
      )}
    </div>
  );
}

function ConfirmDiscard({ turns, words, onDownload, onDiscard, onCancel }) {
  // Escape cancels, and focus lands on the safe action — the destructive one
  // should never be a stray Enter away.
  const safeRef = useRef(null);
  useEffect(() => {
    safeRef.current?.focus();
    const onKey = (e) => e.key === "Escape" && onCancel();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onCancel]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-neutral-900/40 px-6"
      onClick={onCancel}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="discard-title"
        className="w-full max-w-md rounded-xl border border-neutral-200 bg-white p-6 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 id="discard-title" className="text-base font-semibold">
          Start a new recording?
        </h2>
        <p className="mt-2 text-sm text-neutral-600">
          The current transcript — {turns} turn{turns !== 1 ? "s" : ""}, {words}{" "}
          word{words !== 1 ? "s" : ""} — hasn't been downloaded. Starting a new
          recording clears it, and it can't be recovered.
        </p>
        <div className="mt-5 flex flex-wrap justify-end gap-2">
          <button
            ref={safeRef}
            onClick={onCancel}
            className="rounded-full border border-neutral-300 bg-white px-4 py-2 text-sm font-medium text-neutral-700 hover:bg-neutral-100"
          >
            Keep transcript
          </button>
          <button
            onClick={onDownload}
            className="rounded-full bg-neutral-900 px-4 py-2 text-sm font-medium text-white hover:bg-neutral-800"
          >
            Download, then start
          </button>
          <button
            onClick={onDiscard}
            className="rounded-full border border-red-300 bg-white px-4 py-2 text-sm font-medium text-red-700 hover:bg-red-50"
          >
            Discard and start
          </button>
        </div>
      </div>
    </div>
  );
}

function SpeakerSelect({ speakers, setSpeakers, disabled }) {
  return (
    <label
      className={
        "flex items-center gap-1.5 text-xs text-neutral-500 " +
        (disabled ? "opacity-50" : "")
      }
      title={
        "Auto — estimate how many people are speaking.\n" +
        "A number — assume EXACTLY that many speakers and never merge them.\n\n" +
        "Pick a number when you know the count (a two-person interview): it stops " +
        "two similar voices being collapsed into one speaker mid-recording.\n" +
        "Pick Auto when you don't: assuming 2 while only one person speaks will " +
        "split that person into two speakers."
      }
    >
      <span className="hidden sm:inline">Speakers</span>
      <select
        value={speakers}
        disabled={disabled}
        onChange={(e) => setSpeakers(Number(e.target.value))}
        className="rounded-full border border-neutral-300 bg-white px-2.5 py-1.5 text-xs font-medium text-neutral-700 disabled:cursor-not-allowed"
      >
        {SPEAKER_CHOICES.map((n) => (
          <option key={n} value={n}>
            {n === 0 ? "Auto" : `Exactly ${n}`}
          </option>
        ))}
      </select>
    </label>
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

/**
 * AccountMenu — which organization this transcript belongs to, and a way out.
 *
 * The organization name is shown rather than only the email because it is the
 * thing that changes what the session does: quota, plan and every stored
 * transcript are scoped to it, and a person with accounts at two customers
 * otherwise has no way to tell which one they are recording into.
 *
 * Sign out is disabled while recording. It drops the token, which closes the
 * socket mid-sentence and loses an undownloaded transcript.
 */
function AccountMenu({ user, onSignOut, recording }) {
  if (!user) return null;
  return (
    <div className="flex items-center gap-3 border-l border-neutral-200 pl-4">
      <div className="hidden text-right leading-tight sm:block">
        <div className="text-xs font-medium text-neutral-700">
          {user.organizationName}
        </div>
        <div className="text-[11px] text-neutral-400">{user.email}</div>
      </div>
      <button
        type="button"
        onClick={onSignOut}
        disabled={recording}
        title={
          recording
            ? "Stop the recording before signing out"
            : "Sign out of VoxLive"
        }
        className="rounded-full border border-neutral-300 px-3 py-1.5 text-xs font-medium text-neutral-600 transition-colors hover:bg-neutral-100 disabled:cursor-not-allowed disabled:opacity-40"
      >
        Sign out
      </button>
    </div>
  );
}
