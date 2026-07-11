"""
Central configuration — all values overridable via .env.
Copy .env.example -> .env, set credentials, done.

Two authentication modes for Gemini:

  A) API key (personal / quick start):
        GEMINI_API_KEY=...            # https://aistudio.google.com/apikey

  B) Vertex AI via service account (org deployments — the internship setup,
     where the raw API key can't be shared but a service account JSON exists):
        GEMINI_USE_VERTEX=true
        GOOGLE_CLOUD_PROJECT=your-gcp-project-id
        GOOGLE_CLOUD_LOCATION=us-central1     # or "global"
        GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

Speaker diarization (pyannote) — v3 dual-threshold model:
        DIARIZATION_MODE=pyannote
        HUGGINGFACE_TOKEN=hf_...      # hf.co/settings/tokens
        MAX_SPEAKERS=2                # set to the expected count when known!
        DIARIZATION_THRESHOLD=0.45            # match threshold (same speaker)
        DIARIZATION_NEW_SPEAKER_MARGIN=0.15   # + margin -> create threshold
        MIN_NEW_SPEAKER_SEC=2.0
   Distances <= threshold             -> same speaker
   threshold < d <= threshold+margin  -> nearest speaker, embedding NOT stored
   d > threshold+margin               -> new speaker (if long enough + capped)
   TUNE via the "diarize: ... dist=" log lines: same-speaker distances must
   fall below the threshold, cross-speaker above threshold+margin.
   NOTE: pyannote/embedding is a GATED model — you must (once) visit
   https://hf.co/pyannote/embedding while logged in and accept the conditions,
   otherwise loading fails with 401.

Latency: GEMINI_MODEL=gemini-2.5-flash-lite is noticeably faster/cheaper than
gemini-2.5-flash with a small accuracy cost — worth A/B testing for the live
demo. Diarization runs in parallel with the Gemini call (free).
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _csv(name: str, default: str) -> tuple:
    return tuple(x.strip() for x in os.getenv(name, default).split(",") if x.strip())


@dataclass
class Settings:
    # ---- ASR provider (Gemini only in this build) ----
    provider: str = os.getenv("ASR_PROVIDER", "gemini")

    # ---- Gemini auth (see module docstring for the two modes) ----
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_use_vertex: bool = _bool("GEMINI_USE_VERTEX", "false")
    google_cloud_project: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    google_cloud_location: str = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    # Model: gemini-2.5-flash (stable) | gemini-2.5-flash-lite (faster/cheaper)
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # ---- rolling ASR context ----
    # Number of recent transcribed segments fed back to Gemini as context for
    # the next segment. Recovers most of the accuracy advantage of whole-file
    # transcription that isolated 2-4s VAD chunks lose. 0 disables.
    context_segments: int = int(os.getenv("CONTEXT_SEGMENTS", "4"))

    # ---- language lock: only these are transcribed AND displayed ----
    allowed_languages: tuple = _csv("ALLOWED_LANGUAGES", "si,en,ta")

    # ---- ANTI-HALLUCINATION GUARDS ----
    # 1) Energy gate: segments quieter than this RMS (int16 scale, 0..32767)
    #    are never sent to Gemini. Silence/breath/AC-hum segments are the #1
    #    trigger for the model INVENTING fluent speech. Raise if hallucinations
    #    persist on quiet noise; lower if genuinely quiet speech gets dropped.
    min_segment_rms: int = int(os.getenv("MIN_SEGMENT_RMS", "120"))
    # 2) Density gate: real conversational speech is ~2-4 words/sec. If the
    #    "transcript" packs more than this many words per second of audio, the
    #    model invented text -> the segment is dropped.
    max_words_per_sec: float = float(os.getenv("MAX_WORDS_PER_SEC", "8.0"))

    # ---- audio / VAD segmentation ----
    sample_rate: int = int(os.getenv("SAMPLE_RATE", "16000"))
    # 0=least aggressive filter (keeps quiet speech) .. 3=most aggressive.
    # 2 filters more non-speech noise BEFORE it can be hallucinated.
    vad_aggressiveness: int = int(os.getenv("VAD_AGGRESSIVENESS", "2"))
    # Silence before finalizing a segment. v3 default 400ms (was 500):
    # broadcast turn-taking is FAST — with 500ms a speaker change often lands
    # inside one segment, which loses the new speaker's opening words to the
    # previous speaker (and gives the segment a single wrong label).
    silence_ms: int = int(os.getenv("SILENCE_MS", "400"))
    # Past this length, a monologue is cut at the next micro-gap (live feel).
    # ACCURACY/LATENCY KNOB: longer segments = better transcripts and fewer
    # hallucinations, slower "live" feel. 4000 = snappy; 6000-10000 = quality.
    soft_max_segment_ms: int = int(os.getenv("SOFT_MAX_SEGMENT_MS", "6000"))
    # Absolute ceiling (rarely hit; the only cut that may land mid-word).
    max_segment_ms: int = int(os.getenv("MAX_SEGMENT_MS", "10000"))
    # Sub-300ms blips are dropped as noise (classic hallucination trigger).
    min_segment_ms: int = int(os.getenv("MIN_SEGMENT_MS", "300"))

    # ---- diarization: off | pyannote | identify ----
    # pyannote  -> "Speaker 1..N" by voice, N capped at max_speakers
    # identify  -> enrolled names (voiceprints dir) + Speaker-N fallback
    diarization_mode: str = os.getenv("DIARIZATION_MODE", "pyannote")
    # SET THIS TO THE EXPECTED SPEAKER COUNT WHEN KNOWN (e.g., 2 for a
    # two-person interview). v3's ambiguous-zone rule means forced
    # assignments no longer pollute clusters, so capping is safe and is the
    # single most effective guard against phantom speakers.
    max_speakers: int = int(os.getenv("MAX_SPEAKERS", "10"))
    # MATCH threshold: cosine DISTANCE between pyannote embeddings; below it =
    # same speaker. TUNE using the "diarize:" log lines: same-speaker
    # distances must fall below it.
    diarization_threshold: float = float(os.getenv("DIARIZATION_THRESHOLD", "0.45"))
    # CREATE threshold = match + this margin. Distances in between are
    # "ambiguous": assigned to the nearest speaker but never stored, never
    # spawning a new speaker. Raising the margin = fewer phantom speakers,
    # at the cost of a genuinely new voice needing to be more distinct.
    diarization_new_speaker_margin: float = float(
        os.getenv("DIARIZATION_NEW_SPEAKER_MARGIN", "0.15")
    )
    # A NEW speaker is only created from a segment at least this long;
    # shorter segments are assigned to the nearest existing speaker.
    # v3 default 2.0s (was 1.0): one noisy second is not enough evidence to
    # invent a person.
    min_new_speaker_sec: float = float(os.getenv("MIN_NEW_SPEAKER_SEC", "2.0"))
    huggingface_token: str = os.getenv("HUGGINGFACE_TOKEN", "")
    if not huggingface_token and diarization_mode in ("pyannote", "identify"):
        raise ValueError(
            "DIARIZATION_MODE=pyannote or identify requires HUGGINGFACE_TOKEN "
            "in .env, and a one-time acceptance of the model conditions at "
            "hf.co/pyannote/embedding."
        )
    voiceprints_dir: str = os.getenv("VOICEPRINTS_DIR", "voiceprints")

    # ---- optional pipeline stages ----
    enable_postprocess: bool = _bool("ENABLE_POSTPROCESS", "false")

    # ---- CORS ----
    cors_origins: tuple = _csv("CORS_ORIGINS", "http://localhost:5173")


settings = Settings()

# Validate AFTER instantiation — a `raise` inside a dataclass class body runs
# at class-definition time and only sees the class-level default, which is
# fragile and breaks importing this module in tests.
if settings.gemini_use_vertex:
    if not settings.google_cloud_project:
        raise ValueError(
            "GEMINI_USE_VERTEX=true requires GOOGLE_CLOUD_PROJECT in .env, and "
            "GOOGLE_APPLICATION_CREDENTIALS must point at the service account "
            "JSON (the same one provisioned for Chirp works once Vertex AI API "
            "is enabled and the account has the 'Vertex AI User' role)."
        )
elif not settings.gemini_api_key:
    raise ValueError(
        "GEMINI_API_KEY must be set in .env "
        "(or set GEMINI_USE_VERTEX=true to authenticate via a service account)."
    )
