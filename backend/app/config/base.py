"""
Central configuration — all values overridable via .env.

Gemini auth (two modes):
  A) API key:                GEMINI_API_KEY=...
  B) Vertex service account: GEMINI_USE_VERTEX=true + GOOGLE_CLOUD_PROJECT
                             + GOOGLE_APPLICATION_CREDENTIALS

DIARIZATION (v10)
  There is one mode now, plus off. v5-v7 shipped four
  (pipeline/pyannote/identify/off) that shared the same defect: a speaker was
  decided once, greedily, from one short embedding, and never revisited.
  DIARIZATION_THRESHOLD could not fix that, because embedding distance depends
  on turn DURATION as strongly as on speaker identity — same-speaker distance
  is ~0.27 at 3 s but ~0.67 at 0.8 s. No single threshold is correct for both,
  which is why short turns spawned phantom speakers.

  v10 clusters the WHOLE session from scratch on every pass, using only
  windows long enough to carry identity. There is no threshold to tune and
  nothing to poison, so the knobs below are about LATENCY, not accuracy.

  EXPECTED_SPEAKERS=K   Live systems usually know their speaker count. It is
                        treated as a CEILING, not a quota: with K=2 and only
                        one person talking, you get one speaker, not a
                        monologue chopped between two.
  MAX_SPEAKERS          Ceiling when EXPECTED_SPEAKERS is 0 (auto).
  DIARIZE_INTERVAL_SEC  How often a background pass runs.
  DIARIZE_WAIT_MS       How long a finished segment waits for the timeline to
                        cover it before shipping with a best-effort label.

  Requires pyannote.audio>=3.1 and a one-time acceptance of ONE gated model:
      https://hf.co/pyannote/wespeaker-voxceleb-resnet34-LM
  (speaker-diarization-community-1 is no longer used — v10 does not run a
  whole-file pipeline on three-second chunks.)
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
    # ---- ASR provider ----
    provider: str = os.getenv("ASR_PROVIDER", "gemini")

    # ---- Gemini auth ----
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_use_vertex: bool = _bool("GEMINI_USE_VERTEX", "false")
    google_cloud_project: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    # NOT "global": the global endpoint adds routing variance to every call,
    # and latency is the whole point here. Pin the closest region.
    google_cloud_location: str = os.getenv("GOOGLE_CLOUD_LOCATION", "asia-south1")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    # Flash-tier models think by default. For transcription that is pure
    # latency for zero gain — the task has no reasoning in it.
    gemini_thinking_budget: int = int(os.getenv("GEMINI_THINKING_BUDGET", "0"))

    # ---- rolling ASR context ----
    context_segments: int = int(os.getenv("CONTEXT_SEGMENTS", "4"))

    # ---- language lock ----
    allowed_languages: tuple = _csv("ALLOWED_LANGUAGES", "si,en,ta")

    # ---- anti-hallucination guards ----
    min_segment_rms: int = int(os.getenv("MIN_SEGMENT_RMS", "120"))
    max_words_per_sec: float = float(os.getenv("MAX_WORDS_PER_SEC", "8.0"))
    # Companion to MAX_WORDS_PER_SEC. Sinhala and Tamil agglutinate, so a
    # runaway segment in either stays well under the WORD limit while its
    # character count explodes. ~28 c/s is roughly double a fast speaker.
    max_chars_per_sec: float = float(os.getenv("MAX_CHARS_PER_SEC", "28.0"))

    # ---- audio / VAD segmentation ----
    sample_rate: int = int(os.getenv("SAMPLE_RATE", "16000"))
    vad_aggressiveness: int = int(os.getenv("VAD_AGGRESSIVENESS", "2"))
    # Shorter than v7's 400 ms. Diarization no longer depends on the VAD
    # catching every turn change, so this is now purely a responsiveness
    # setting — and 320 ms is still comfortably above a within-sentence pause.
    silence_ms: int = int(os.getenv("SILENCE_MS", "320"))
    # v7 used 6000. Every segment waited for the soft cap before ANY text
    # appeared, so this alone put the transcript up to six seconds behind.
    soft_max_segment_ms: int = int(os.getenv("SOFT_MAX_SEGMENT_MS", "3500"))
    max_segment_ms: int = int(os.getenv("MAX_SEGMENT_MS", "9000"))
    min_segment_ms: int = int(os.getenv("MIN_SEGMENT_MS", "300"))

    # ---- diarization ----
    diarization_enabled: bool = os.getenv("DIARIZATION_MODE", "on").lower() not in (
        "off",
        "false",
        "0",
        "none",
    )
    expected_speakers: int = int(os.getenv("EXPECTED_SPEAKERS", "0"))
    max_speakers: int = int(os.getenv("MAX_SPEAKERS", "6"))
    diarize_interval_sec: float = float(os.getenv("DIARIZE_INTERVAL_SEC", "1.5"))
    diarize_wait_ms: int = int(os.getenv("DIARIZE_WAIT_MS", "900"))
    # embedding | sortformer | auto  — see diarizer_factory.py
    diarization_backend: str = os.getenv("DIARIZATION_BACKEND", "embedding")
    sortformer_window_sec: float = float(os.getenv("SORTFORMER_WINDOW_SEC", "90"))
    huggingface_token: str = os.getenv("HUGGINGFACE_TOKEN", "")

    # ---- optional pipeline stages ----
    enable_postprocess: bool = _bool("ENABLE_POSTPROCESS", "false")

    # ---- CORS ----
    cors_origins: tuple = _csv("CORS_ORIGINS", "http://localhost:5173")


settings = Settings()

# Validate AFTER instantiation — a `raise` in a dataclass class body runs at
# class-definition time and only ever sees the class-level default.
if (
    settings.diarization_enabled
    and settings.diarization_backend.lower() != "sortformer"
    and not settings.huggingface_token
):
    raise ValueError(
        "Diarization requires HUGGINGFACE_TOKEN in .env, plus a one-time "
        "acceptance of the gated model conditions at "
        "https://hf.co/pyannote/wespeaker-voxceleb-resnet34-LM "
        "(set DIARIZATION_MODE=off to run transcription only)."
    )

if settings.gemini_use_vertex:
    if not settings.google_cloud_project:
        raise ValueError(
            "GEMINI_USE_VERTEX=true requires GOOGLE_CLOUD_PROJECT in .env, and "
            "GOOGLE_APPLICATION_CREDENTIALS must point at the service account "
            "JSON with the 'Vertex AI User' role."
        )
elif not settings.gemini_api_key:
    raise ValueError(
        "GEMINI_API_KEY must be set in .env "
        "(or set GEMINI_USE_VERTEX=true to authenticate via a service account)."
    )
