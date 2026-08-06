"""
gemini_provider.py — Gemini ASR for Sinhala / Tamil / English, including
code-switched speech.

LATENCY NOTES (this file is where most of the remaining wall-clock lives)
------------------------------------------------------------------------
1. THINKING IS OFF. Flash-tier models reason before answering by default.
   Transcription has no reasoning in it — the model is reading audio — so a
   thinking budget is pure added latency. Setting it to 0 typically removes
   400-900 ms per call. This is the single largest ASR-side win available.

2. THE CLIENT IS BUILT ONCE. Constructing a genai.Client per request
   re-resolves credentials and re-opens the connection pool.

3. THE REGION IS PINNED. GOOGLE_CLOUD_LOCATION=global routes each call
   through Google's global front door, which is fine for throughput and bad
   for tail latency. From Sri Lanka, asia-south1 (Mumbai) is normally the
   closest low-latency region; asia-southeast1 (Singapore) is the usual
   second choice. Measure both — it is a two-line experiment worth doing.

4. THE RESPONSE IS SCHEMA-CONSTRAINED. Asking for JSON in prose and then
   parsing it invites preambles and markdown fences; a response schema makes
   the output shape a decoding constraint instead of a request.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import struct
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("voxlive.gemini")

LANG_NAMES = {"si": "Sinhala", "ta": "Tamil", "en": "English"}


@dataclass
class ASRResult:
    text: str
    language: str


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
        thinking_budget: int = 0,
    ):
        from google import genai

        self.model = model
        self.allowed = tuple(allowed_languages)
        self.max_wps = float(max_words_per_sec)
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
            "3. `language` is the language of the MAJORITY of the words. Use "
            "exactly one of: " + ", ".join(self.allowed) + ".\n"
            "4. If the audio is silence, noise, breathing or music, return an "
            "empty string for `text`. Never invent speech. An empty result is "
            "always better than a plausible guess.\n"
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
        if context:
            # Rolling context materially improves proper nouns and code-switch
            # boundaries. It is a HINT, and must be fenced as one, or the model
            # will happily continue the previous sentence instead of
            # transcribing the audio.
            parts.append(
                types.Part(
                    text=(
                        "Context — the immediately preceding transcript, for "
                        "vocabulary and spelling consistency ONLY. Do not "
                        "repeat, continue or transcribe it:\n"
                        f"<<<{context[-1200:]}>>>"
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

        return self._parse(resp, len(pcm) / 2 / sample_rate)

    def _parse(self, resp, duration: float) -> ASRResult:
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

        # Hallucination guard: nobody speaks 8 words a second. A burst well
        # above human rate means the model looped on noise.
        words = len(text.split())
        if duration > 0.4 and words / duration > self.max_wps:
            log.info(
                "hallucination guard: %d words in %.1fs (%.1f w/s), dropping",
                words,
                duration,
                words / duration,
            )
            return ASRResult("", lang)

        return ASRResult(text, lang)
