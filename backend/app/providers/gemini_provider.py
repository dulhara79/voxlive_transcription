"""
Gemini provider - LLM-based transcription via the batch generateContent API.

WHY THIS EXISTS
  Whisper (OpenAI) and Google Chirp / si-LK both failed for conversational
  Sri Lankan speech. Gemini is a multimodal LLM, so it transcribes from
  semantic context, not pure acoustics - which handles Sinhala+English+Tamil
  code-switching inside one sentence better than acoustic-only engines.

ROLLING CONTEXT (accuracy)
  A whole-file upload gives Gemini the full conversation as context; our VAD
  pipeline sends isolated 2-4 s chunks, which is exactly the condition where
  an LLM transcriber mishears or HALLUCINATES plausible sentences on unclear
  audio. To recover most of that context advantage, the pipeline passes the
  last few transcribed segments in `context`, appended to the SYSTEM
  instruction (never to `contents`, so the no-echo design is preserved).

ANTI-HALLUCINATION (layered — Gemini is an assistant, not an ASR engine; on
noisy/quiet/unclear audio it will happily INVENT fluent speech. No single
prompt fixes that, so we stack independent guards):
  1. PROMPT     verbatim-only, "empty over guessing", partial-transcription
                allowed (so it doesn't fill gaps to make sentences whole).
  2. ECHO GUARD output that parrots the task instruction -> dropped.
  3. CONTEXT-ECHO GUARD  output that merely repeats/continues the rolling
                context (a classic LLM failure on near-silent audio) -> dropped.
  4. DENSITY GUARD  real speech is ~2-4 words/sec. If the "transcript" packs
                > max_words_per_sec into the clip's duration, it was invented
                -> dropped.
  Upstream (main.py/config): an RMS energy gate stops silent/hum segments from
  ever reaching Gemini, and VAD aggressiveness/min-segment length are raised —
  quiet noise segments are the #1 hallucination trigger.

AUTHENTICATION (two modes)
  1. API key (default):        GEMINI_API_KEY in .env
  2. Vertex AI service account: GEMINI_USE_VERTEX=true + GOOGLE_CLOUD_PROJECT
     + GOOGLE_APPLICATION_CREDENTIALS (service account JSON, "Vertex AI User"
     role, Vertex AI API enabled).

LATENCY
  gemini-2.5-flash runs "thinking" by default, which adds SECONDS per call for
  no benefit on transcription. We disable it (thinking_budget=0) and disable
  automatic function calling. That typically turns 6-9s calls into ~1-2s.

SETUP
  pip install google-genai
"""

import asyncio
import io
import json
import logging
import os
import re
import wave

from google import genai
from google.genai import types

from .base import SpeechProvider, TranscriptResult

log = logging.getLogger("voxlive.gemini")

TARGET_LANGS = ("si", "en", "ta")
LANG_DISPLAY = {
    "si": "\u0dc3\u0dd2\u0d82\u0dc4\u0dbd",
    "en": "English",
    "ta": "\u0ba4\u0bae\u0bbf\u0bb4\u0bcd",
}

# Phrases the model might echo from the task instruction. If the "transcript"
# is exactly one of these, it parroted the prompt -> treat as no speech.
_ECHO_GUARD = {
    "transcribe this audio.",
    "transcribe this audio",
    "transcribe the audio.",
    "transcribe the audio",
}

# Cap how much rolling context we append (chars). Enough for several
# segments of vocabulary/topic, small enough to keep calls fast and cheap.
_MAX_CONTEXT_CHARS = 700

SYSTEM_PROMPT = (
    "You are a strict speech-to-text transcription engine, not an assistant. "
    "Transcribe ONLY the words that are audibly spoken in THIS audio clip, "
    "then stop.\n"
    "Rules:\n"
    "- Output only the verbatim words actually spoken. Do NOT translate, "
    "summarize, paraphrase, correct grammar, complete sentences, or clean up "
    "the speech.\n"
    "- The speakers are Sri Lankan and mix Sinhala, English and Tamil, often "
    "within one sentence. Keep every word in the language and script it was "
    "actually spoken in: Sinhala in Sinhala script, English in Latin script, "
    "Tamil in Tamil script. Never convert one language into another.\n"
    "- Keep filler words, repetitions and false starts as spoken.\n"
    "- If only PART of the clip is intelligible, transcribe only that part. "
    "Never fill gaps with guessed words to make a sentence complete.\n"
    "- Set 'language' to the DOMINANT language of the segment: si, en, or ta.\n"
    "- Only if the audio is clearly a language OTHER than Sinhala, English or "
    "Tamil, return an empty 'text' and language 'other'.\n"
    "- If there is no intelligible speech (silence, breathing, background "
    "noise, music, keyboard sounds), return an empty 'text'. This is the "
    "CORRECT answer for such audio — never describe the sounds, never invent "
    "speech for them.\n"
    "- NEVER guess or invent speech. If the audio is too unclear to transcribe "
    "confidently, return an empty 'text' rather than a plausible-sounding "
    "sentence. An omission is acceptable; a fabrication is not.\n"
    "- Never add commentary, notes, brackets, or explanations."
)

_CONTEXT_PREFIX = (
    "\n\nFor context ONLY, here is the most recent transcript of this SAME "
    "ongoing conversation. Use it ONLY to recognize the topic, names, and "
    "code-switched vocabulary in the new audio. NEVER repeat, continue, "
    "complete, or paraphrase this context in your output. If the new audio "
    "contains no clear speech, return empty text — do NOT reuse words from "
    "this context. Transcribe ONLY the words actually spoken in the new "
    "audio:\n"
)

RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "text": types.Schema(type=types.Type.STRING),
        "language": types.Schema(
            type=types.Type.STRING, enum=["si", "en", "ta", "other"]
        ),
    },
    required=["text", "language"],
)


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    buf.seek(0)
    return buf.read()


def _norm(s: str) -> str:
    """Normalize for fuzzy comparison: strip punctuation/whitespace, lowercase.
    \\w matches Unicode letters in Python 3, so Sinhala/Tamil are preserved."""
    return re.sub(r"[\W_]+", "", s, flags=re.UNICODE).lower()


class GeminiProvider(SpeechProvider):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        allowed_languages: tuple = TARGET_LANGS,
        use_vertex: bool = False,
        project: str | None = None,
        location: str = "us-central1",
        max_words_per_sec: float = 8.0,
    ):
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.allowed = (
            set(allowed_languages) if allowed_languages else set(TARGET_LANGS)
        )
        self.max_words_per_sec = max_words_per_sec

        if use_vertex:
            # Vertex AI path: authenticates via Application Default Credentials
            # (GOOGLE_APPLICATION_CREDENTIALS -> service account JSON). No API
            # key involved; usage is billed to the GCP project.
            project = project or os.getenv("GOOGLE_CLOUD_PROJECT")
            if not project:
                raise ValueError(
                    "Vertex AI mode requires GOOGLE_CLOUD_PROJECT (and "
                    "GOOGLE_APPLICATION_CREDENTIALS pointing at the service "
                    "account JSON)."
                )
            self.client = genai.Client(
                vertexai=True, project=project, location=location
            )
            auth = f"Vertex AI (project={project}, location={location}, ADC)"
        else:
            key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not key:
                raise ValueError(
                    "GEMINI_API_KEY is not set.\n"
                    "Create one at https://aistudio.google.com/apikey, or set "
                    "GEMINI_USE_VERTEX=true to use a service account."
                )
            self.client = genai.Client(api_key=key)
            auth = "API key"

        self._config = self._make_config(None)
        log.info(
            "Gemini provider ready (model=%s, auth=%s, thinking disabled)",
            self.model,
            auth,
        )

    def _make_config(self, context: str | None) -> "types.GenerateContentConfig":
        instruction = SYSTEM_PROMPT
        if context:
            instruction += _CONTEXT_PREFIX + context[-_MAX_CONTEXT_CHARS:]
        return types.GenerateContentConfig(
            system_instruction=instruction,
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            # Kill latency: no internal "thinking", no function-calling probe.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )

    async def transcribe_segment(
        self, pcm_bytes, sample_rate, context: str | None = None
    ) -> TranscriptResult:
        return await asyncio.to_thread(
            self._transcribe, pcm_bytes, sample_rate, context
        )

    def _transcribe(
        self, pcm_bytes: bytes, sample_rate: int, context: str | None = None
    ) -> TranscriptResult:
        wav = _pcm_to_wav(pcm_bytes, sample_rate)
        config = self._make_config(context) if context else self._config

        # Audio only in contents -> nothing for the model to echo as "text".
        response = self.client.models.generate_content(
            model=self.model,
            contents=[types.Part.from_bytes(data=wav, mime_type="audio/wav")],
            config=config,
        )

        raw = (getattr(response, "text", None) or "").strip()
        if not raw:
            return TranscriptResult(text="", language="other", confidence=None)

        text, language = self._parse(raw)
        if not text:
            return TranscriptResult(text="", language="other", confidence=None)

        # --- GUARD 2: instruction echo ---
        if text.lower() in _ECHO_GUARD:
            log.info("echo guard: model parroted the instruction, dropping")
            return TranscriptResult(text="", language="other", confidence=None)

        # --- GUARD 3: context echo (hallucination on unclear audio often
        # just replays/continues the rolling context). Only checked for
        # non-trivial outputs so short genuine repeats like "හරි හරි" pass. ---
        if context:
            nt, nc = _norm(text), _norm(context)
            if len(nt) > 20 and nt and nt in nc:
                log.info(
                    "context-echo guard: output repeats rolling context, "
                    "dropping: %r",
                    text[:60],
                )
                return TranscriptResult(text="", language="other", confidence=None)

        # --- GUARD 4: word-density sanity check. Real speech ~2-4 words/sec;
        # a 2s clip "containing" a 25-word sentence was invented. ---
        duration = len(pcm_bytes) / 2 / sample_rate
        if duration > 0:
            wps = len(text.split()) / duration
            if wps > self.max_words_per_sec:
                log.info(
                    "density guard: %.1f words/sec over %.1fs clip -> "
                    "hallucination, dropping: %r",
                    wps,
                    duration,
                    text[:60],
                )
                return TranscriptResult(text="", language="other", confidence=None)

        if language not in self.allowed:
            log.info(
                "dropped non-target language: lang=%s text=%r", language, text[:40]
            )
            return TranscriptResult(text="", language=language, confidence=None)

        log.info(
            "lang=%s (%s) text=%r", language, LANG_DISPLAY.get(language, "?"), text[:80]
        )
        return TranscriptResult(text=text, language=language, confidence=None)

    @staticmethod
    def _parse(raw: str):
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned[4:] if cleaned.lower().startswith("json") else cleaned
            cleaned = cleaned.strip()
        try:
            obj = json.loads(cleaned)
            return (obj.get("text", "") or "").strip(), (
                obj.get("language") or "other"
            ).lower()
        except (json.JSONDecodeError, AttributeError):
            log.warning("non-JSON response, dropping: %r", raw[:80])
            return "", "other"
