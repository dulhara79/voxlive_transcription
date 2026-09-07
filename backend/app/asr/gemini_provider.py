"""
gemini_provider.py — Gemini ASR for Sinhala / Tamil / English, including
code-switched speech.

CHANGES IN THIS REVISION
------------------------
1. THE CONTEXT HINT IS NOW CHECKED AGAINST THE OUTPUT. `transcribe_segment`
   passes `context` down to `_parse`, so `clean_transcript` can tell whether
   the model transcribed the audio or continued the prompt. Previously the
   hint was sent and never audited; a model that echoed it produced fluent,
   correctly-scripted, non-repetitive text describing audio from ten seconds
   ago, and nothing in the pipeline could see that.

2. A CHARACTER-RATE GUARD SITS BESIDE THE WORD-RATE GUARD. `MAX_WORDS_PER_SEC`
   barely functions for Sinhala and Tamil: both agglutinate, so a sentence
   that would be eight English words is three or four tokens. A model looping
   in Sinhala can emit 400 characters in three seconds and still measure ~2
   words/sec, well under the 8.0 limit. Characters per second is the scale-free
   version of the same check.

3. THE LANGUAGE LABEL IS DERIVED FROM THE SCRIPT, NOT TRUSTED. `language` and
   `text` are produced independently — the response schema constrains the
   first and cannot constrain the second — so Sinhala-script text labelled
   `en` is routine. The frontend colours turns by that label, so a wrong label
   is visible to the user even when the text is perfect.

LATENCY NOTES (this file is where most of the remaining wall-clock lives)
------------------------------------------------------------------------
1. THINKING IS OFF. Flash-tier models reason before answering by default.
   Transcription has no reasoning in it — the model is reading audio — so a
   thinking budget is pure added latency. Setting it to 0 typically removes
   400-900 ms per call.

2. THE CLIENT IS BUILT ONCE. Constructing a genai.Client per request
   re-resolves credentials and re-opens the connection pool.

3. THE REGION IS PINNED. GOOGLE_CLOUD_LOCATION=global routes each call
   through Google's global front door, which is fine for throughput and bad
   for tail latency. From Sri Lanka, asia-south1 (Mumbai) is normally the
   closest low-latency region; asia-southeast1 (Singapore) is the usual
   second choice.

4. THE RESPONSE IS SCHEMA-CONSTRAINED. Asking for JSON in prose and then
   parsing it invites preambles and markdown fences; a response schema makes
   the output shape a decoding constraint instead of a request.
"""

from __future__ import annotations

import io
import json
import logging
import struct
from dataclasses import dataclass, field
from typing import Optional

from .language_spans import LanguageSpan, language_spans, languages_in
from .validation import clean_transcript, dominant_language

log = logging.getLogger("voxlive.gemini")

LANG_NAMES = {"si": "Sinhala", "ta": "Tamil", "en": "English"}


@dataclass
class ASRResult:
    text: str
    language: str  # DOMINANT language, for the gate and the paragraph colour
    # Phase 4: where each language actually sits inside `text`. Derived from
    # Unicode script ranges, so it is exact rather than predicted — si, ta and
    # en occupy disjoint blocks. Character offsets only: this provider's model
    # returns a plain string with no timings. See language_spans.py for why
    # they are not interpolated.
    language_spans: list[LanguageSpan] = field(default_factory=list)


def _wav(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap raw mono int16 PCM in a WAV container.

    Gemini accepts raw PCM, but a container removes any ambiguity about rate
    and endianness — and a wrong sample rate is silent: the model returns
    fluent, confident, completely wrong text rather than an error.
    """
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(pcm)))
    buf.write(b"WAVEfmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", len(pcm)))
    buf.write(pcm)
    return buf.getvalue()


class GeminiProvider:
    def __init__(
        self,
        api_key: str = "",
        model: str = "gemini-3.6-flash",
        allowed_languages: tuple = ("si", "en", "ta"),
        use_vertex: bool = False,
        project: str = "",
        location: str = "asia-south1",
        max_words_per_sec: float = 8.0,
        max_chars_per_sec: float = 28.0,
        thinking_budget: int = 0,
    ):
        from google import genai

        self.model = model
        # Anything outside si/ta/en is discarded rather than silently accepted:
        # the script filter, the prompt and the response enum all derive from
        # this tuple, so letting an unsupported code through here would quietly
        # widen all three.
        self.allowed = tuple(c for c in allowed_languages if c in LANG_NAMES) or (
            "si",
            "en",
            "ta",
        )
        if len(self.allowed) != len(tuple(allowed_languages)):
            log.warning(
                "ignoring unsupported language codes in ALLOWED_LANGUAGES=%s; "
                "this build supports only %s",
                ",".join(allowed_languages),
                ",".join(sorted(LANG_NAMES)),
            )
        self.max_wps = float(max_words_per_sec)
        self.max_cps = float(max_chars_per_sec)
        self.thinking_budget = int(thinking_budget)

        if use_vertex:
            if location == "global":
                log.warning(
                    "GOOGLE_CLOUD_LOCATION=global adds routing latency to every "
                    "call; pin a region (asia-south1 from Sri Lanka)."
                )
            self.client = genai.Client(
                vertexai=True, project=project, location=location
            )
            log.info("Gemini via Vertex AI (%s, %s)", project, location)
        else:
            self.client = genai.Client(api_key=api_key)
            log.info("Gemini via API key")

        self._system = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        names = ", ".join(LANG_NAMES.get(c, c) for c in self.allowed)
        return (
            "You are a verbatim speech transcriber for live Sri Lankan audio.\n"
            f"The audio contains {names}, and speakers frequently CODE-SWITCH "
            "mid-sentence.\n"
            "\n"
            "RULES\n"
            "1. Transcribe exactly what is said. Do NOT translate, summarise, "
            "correct grammar, or complete unfinished sentences.\n"
            "2. Write each language in its own script: Sinhala in Sinhala "
            "script, Tamil in Tamil script, English in Latin script. If a "
            "sentence mixes languages, keep the mix and keep each part in its "
            "own script — do not romanise Sinhala or Tamil.\n"
            "2b. Use ONLY Sinhala, Tamil and Latin script. Never output "
            "Thaana/Dhivehi, Devanagari, Arabic, Kannada, Malayalam, Telugu, "
            "Bengali or any other script. Sinhala is frequently confused with "
            "Thaana — if audio sounds like an unfamiliar South Asian "
            "language, it is Sinhala or Tamil, or it is not speech.\n"
            "2c. The speaker is speaking one of these three languages. If you "
            "are unsure which, choose between them — never fall back to a "
            "fourth language, and never output a transliteration.\n"
            "3. `language` is the language of the MAJORITY of the words. Use "
            "exactly one of: " + ", ".join(self.allowed) + ".\n"
            "4. If the audio is silence, noise, breathing or music, return an "
            "empty string for `text`. Never invent speech. An empty result is "
            "always better than a plausible guess.\n"
            "4b. Never repeat a word or phrase over and over. If you find "
            "yourself repeating, the audio is unintelligible: stop and return "
            "what you were certain of, or an empty string.\n"
            "5. The clip is a fragment of a longer conversation. It may begin "
            "or end mid-word. Transcribe the fragment as heard; do not pad it.\n"
            "6. No preamble, no commentary, no speaker labels, no timestamps."
        )

    async def transcribe_segment(
        self, pcm: bytes, sample_rate: int, context: Optional[str] = None
    ) -> ASRResult:
        from google.genai import types

        parts = [
            types.Part.from_bytes(data=_wav(pcm, sample_rate), mime_type="audio/wav")
        ]
        hint = context[-1200:] if context else None
        if hint:
            # Rolling context materially improves proper nouns and code-switch
            # boundaries. It is a HINT, and must be fenced as one, or the model
            # will happily continue the previous sentence instead of
            # transcribing the audio. `_parse` now verifies that it didn't.
            parts.append(
                types.Part(
                    text=(
                        "Context — the immediately preceding transcript, for "
                        "vocabulary and spelling consistency ONLY. Do not "
                        "repeat, continue or transcribe it:\n"
                        f"<<<{hint}>>>"
                    )
                )
            )

        cfg = types.GenerateContentConfig(
            system_instruction=self._system,
            temperature=0.0,
            response_mime_type="application/json",
            response_schema={
                "type": "OBJECT",
                "properties": {
                    "language": {"type": "STRING", "enum": list(self.allowed)},
                    "text": {"type": "STRING"},
                },
                "required": ["language", "text"],
            },
        )
        # Not every model exposes a thinking config; never let that be fatal.
        try:
            cfg.thinking_config = types.ThinkingConfig(
                thinking_budget=self.thinking_budget
            )
        except Exception:  # noqa: BLE001
            pass

        resp = await self.client.aio.models.generate_content(
            model=self.model,
            contents=[types.Content(role="user", parts=parts)],
            config=cfg,
        )

        return self._parse(resp, len(pcm) / 2 / sample_rate, hint)

    def _parse(self, resp, duration: float, context: Optional[str]) -> ASRResult:
        raw = (getattr(resp, "text", "") or "").strip()
        if not raw:
            return ASRResult("", self.allowed[0])
        try:
            data = json.loads(
                raw.removeprefix("```json").removeprefix("```").removesuffix("```")
            )
        except json.JSONDecodeError:
            log.warning("non-JSON response: %.120s", raw)
            return ASRResult("", self.allowed[0])

        text = " ".join(str(data.get("text", "")).split())
        lang = str(data.get("language", "")).lower()[:2]
        if lang not in self.allowed:
            lang = self.allowed[0]
        if not text:
            return ASRResult("", lang)

        # Guard 1 — RATE. Two measures of the same thing, because neither one
        # covers all three languages. Words-per-second catches English loops;
        # Sinhala and Tamil agglutinate, so a runaway Sinhala segment stays
        # under the word limit while its character count explodes. A segment
        # only has to trip ONE of them.
        words = len(text.split())
        chars = sum(1 for ch in text if not ch.isspace())
        if duration > 0.4:
            wps = words / duration
            cps = chars / duration
            if wps > self.max_wps or cps > self.max_cps:
                log.info(
                    "rate guard: %d words / %d chars in %.1fs "
                    "(%.1f w/s, %.1f c/s), dropping",
                    words,
                    chars,
                    duration,
                    wps,
                    cps,
                    extra={"event": "asr_rate_rejected"},
                )
                return ASRResult("", lang)

        # Guard 2 — CONTENT. Catches the slow failures the rate guard cannot
        # see: fluent output in a script we never asked for, a phrase repeating
        # like a stuck decoder, or the context hint read back to us. `language`
        # being a valid enum value says nothing about the characters in `text`,
        # so this is the only place the actual script is ever checked.
        cleaned, dropped = clean_transcript(text, self.allowed, context)
        if dropped:
            log.warning(
                "content guard: dropping %.1fs segment (%s)",
                duration,
                dropped,
                extra={"event": "asr_content_rejected", "reason": dropped},
            )
            return ASRResult("", lang)

        # The label is derived from the surviving text rather than taken on
        # trust. The schema constrains `language` to the enum but cannot make
        # it agree with `text`, and a mislabelled turn is visible in the UI:
        # the frontend colours and tags each paragraph by this field.
        detected = dominant_language(cleaned)
        if detected and detected in self.allowed and detected != lang:
            log.info(
                "language label corrected: model said %s, script says %s",
                lang,
                detected,
                extra={"event": "asr_language_corrected"},
            )
            lang = detected

        # Phase 4. Built from the CLEANED text, after the script filter has
        # run: spans must index the string the user will actually see, or the
        # offsets are meaningless. `lang` is the fallback for text that opens
        # with a neutral token.
        spans = language_spans(cleaned, lang)
        present = languages_in(spans)
        if len(present) > 1:
            log.info(
                "code-switched segment: %s across %d span(s)",
                "+".join(present),
                len(spans),
                extra={
                    "event": "asr_code_switch",
                    "languages": present,
                    "spans": len(spans),
                },
            )

        return ASRResult(cleaned, lang, spans)
