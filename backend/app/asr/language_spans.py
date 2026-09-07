"""
language_spans.py — word/phrase-level language structure for code-switched text.

WHY THIS EXISTS
---------------
A Sri Lankan meeting turn is routinely not in one language:

    මචං, where are we going for lunch today? මට ගොඩක් බඩගිනියි, ரொம்ப பசிக்குது!

Until now the system carried ONE language code per chunk (`Chunk.language`)
and `TranscriptStore.paragraphs()` reported a paragraph's language as the
"+"-joined set of its chunks' labels — `si+en`. That says the paragraph
contains Sinhala and English somewhere. It does not say WHERE, so nothing
downstream can render, search, count or evaluate the switch points, and a
switch that happens INSIDE one chunk is invisible entirely.

This module derives the missing structure.

HOW LANGUAGE IS DECIDED: SCRIPT, NOT A MODEL CLAIM
--------------------------------------------------
Sinhala, Tamil and English occupy three disjoint Unicode blocks:

    Sinhala   U+0D80-U+0DFF
    Tamil     U+0B80-U+0BFF  (+ Tamil Supplement U+11FC0-U+11FFF)
    English   Latin

So for THESE three languages, and only because they happen not to share a
script, language identification is a deterministic character-range lookup
rather than a prediction. That is worth stating plainly: `validation.py`
already relies on it to correct the model's own `language` label, and
everything here is exact for the same reason. This approach would NOT
generalise to, say, distinguishing Hindi from Marathi.

WHY TOKENS AND NOT CHARACTERS
-----------------------------
Spans are built from whitespace-separated tokens, then adjacent tokens of the
same language are merged into one span. Cutting inside a word would produce
spans that are not words in any language — and `strip_foreign_scripts()`
already made the same call for the same reason: a reader can see a gap, but
cannot see a corruption.

Tokens carrying no script information at all — "2025", "—", "?" — are NEUTRAL.
They are attached to the surrounding span rather than being given a language
of their own, because a digit is not English and a comma is not Sinhala.

WHAT THIS MODULE DELIBERATELY DOES NOT DO: INVENT TIMINGS
---------------------------------------------------------
The supervisor's Phase 4 schema carries `start` and `end` SECONDS per span.
This module emits `start_char`/`end_char` and leaves `start`/`end` as None,
with `timing="none"`.

That is not an omission, it is the honest answer to what the ASR actually
returns. Verified against current Google documentation (2 Sep 2026):

  * The general Gemini audio path used by `GeminiProvider` returns a plain
    string. No timings of any kind.
  * `gemini-3.5-transcribe` CAN return word-level timings, via
    `timestamp_granularities: ["word"]` -> `word_info` annotations carrying
    `text`, `start_offset`, `end_offset`. But its published supported-language
    table lists NEITHER Sinhala NOR Tamil, and enabling word timestamps is
    documented as degrading transcription accuracy.
  * `word_info` has no `language` field even when it is available, so the
    language attribution below stays ours regardless of provider.

WHY NOT INTERPOLATE TIMINGS FROM CHARACTER POSITION
---------------------------------------------------
It would be one line to map a character offset onto the chunk's [start, end]
proportionally. It is wrong here for a specific, measurable reason, and the
reason is CODE SWITCHING ITSELF.

Proportional interpolation assumes a constant characters-per-second rate. This
codebase already knows that assumption is false across these languages —
`MAX_WORDS_PER_SEC` and `MAX_CHARS_PER_SEC` exist as two separate guards
precisely because Sinhala and Tamil agglutinate and English does not, so the
same duration of speech yields very different character counts. Interpolating
across a switch boundary is therefore biased exactly AT the boundary, which is
the only place these timings would ever be used.

A wrong timestamp is worse than an absent one: absent is visibly absent, wrong
is invisibly wrong, and Phase 15 would score it as if it were measured.

So: `start_char`/`end_char` are exact and always present. `start`/`end` are
present only when a provider genuinely supplied them, and `timing` records
which. `spans_from_word_info()` below is the path that fills them in, ready
for the day a verified provider supplies word timings for si/ta.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

from .validation import SCRIPT_RANGES, _in_ranges, _is_neutral

# Provenance of a span's `start`/`end` seconds.
#   "none"  no timing available — the provider returned text only
#   "api"   timings came from the provider's own word-level annotations
TIMING_NONE = "none"
TIMING_API = "api"


@dataclass
class LanguageSpan:
    """One contiguous run of a single language inside an utterance.

    `start_char`/`end_char` index the utterance text as a Python slice, so
    `text[span.start_char:span.end_char]` is always exactly the span. They are
    derived deterministically from Unicode ranges and are never estimated.

    `start`/`end` are absolute session seconds and are None unless a provider
    supplied word timings. See the module docstring for why they are not
    interpolated.
    """

    language: str  # "si" | "ta" | "en"
    start_char: int
    end_char: int
    start: Optional[float] = None
    end: Optional[float] = None
    timing: str = TIMING_NONE

    def text_of(self, text: str) -> str:
        return text[self.start_char : self.end_char]

    def as_dict(self) -> dict:
        """Wire form. Time fields are emitted even when None, so a client can
        tell 'no timing available' apart from 'field not implemented'."""
        return {
            "language": self.language,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "start": self.start,
            "end": self.end,
            "timing": self.timing,
        }

    def shifted(self, offset: int) -> "LanguageSpan":
        """Same span, character offsets moved by `offset`.

        Needed when chunks are concatenated into a paragraph: each chunk's
        spans were computed against its own text and must be rebased onto the
        joined string. Timings are absolute session seconds and do not move.
        """
        return LanguageSpan(
            language=self.language,
            start_char=self.start_char + offset,
            end_char=self.end_char + offset,
            start=self.start,
            end=self.end,
            timing=self.timing,
        )


def token_language(token: str) -> Optional[str]:
    """Which of si / ta / en a single token is written in, or None if neutral.

    A token mixing scripts — which happens with a Sinhala word carrying a
    Latin suffix, or an English word with a stray mark — is assigned to
    whichever script holds the MAJORITY of its script-bearing characters, with
    Sinhala and Tamil winning ties over Latin. That tie rule matches
    `dominant_language()`: a Sinhala token containing one Latin character is a
    Sinhala token.
    """
    counts = {code: 0 for code in SCRIPT_RANGES}
    for ch in token:
        if _is_neutral(ch):
            continue
        for code, ranges in SCRIPT_RANGES.items():
            if _in_ranges(ch, ranges):
                counts[code] += 1
                break
    if not any(counts.values()):
        return None
    for code in ("si", "ta"):
        if counts[code] and counts[code] >= counts["en"]:
            return code
    best = max(counts, key=lambda c: counts[c])
    return best if counts[best] else None


def _tokens_with_offsets(text: str) -> list[tuple[int, int, str]]:
    """Whitespace-separated tokens as (start_char, end_char, token).

    Written by hand rather than with `re.split` so the offsets are exact even
    when the text contains runs of whitespace, which `str.split()` collapses.
    """
    out: list[tuple[int, int, str]] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i].isspace():
            i += 1
            continue
        j = i
        while j < n and not text[j].isspace():
            j += 1
        out.append((i, j, text[i:j]))
        i = j
    return out


def language_spans(text: str, fallback: Optional[str] = None) -> list[LanguageSpan]:
    """Segment `text` into contiguous single-language spans.

    `fallback` is used only for leading neutral tokens that have no span to
    attach to yet — e.g. a turn opening with a number. Pass the chunk's
    overall language label. If it is None and the whole text is neutral, the
    result is an empty list, which is the correct representation of "this
    string carries no language information" (a bare "2025" or "...").

    Spans are contiguous in token order and never overlap. The gaps between
    them are whitespace only.
    """
    tokens = _tokens_with_offsets(text)
    if not tokens:
        return []

    # Pass 1: classify. Neutral tokens inherit from the previous classified
    # token so that "lunch today? මට" is [en][si] and not [en][neutral][si].
    labelled: list[tuple[int, int, Optional[str]]] = []
    last: Optional[str] = None
    for a, b, tok in tokens:
        lang = token_language(tok)
        if lang is None:
            lang = last  # may still be None at the head of the text
        else:
            last = lang
        labelled.append((a, b, lang))

    # Pass 2: leading neutral tokens look BACKWARDS to nothing, so give them
    # the first real language that appears (or the caller's fallback).
    first_real = next((l for _, _, l in labelled if l is not None), fallback)
    if first_real is None:
        return []
    labelled = [(a, b, l if l is not None else first_real) for a, b, l in labelled]

    # Pass 3: merge adjacent tokens sharing a language. `end_char` extends to
    # the end of the last token in the run, so trailing whitespace is never
    # included in a span.
    spans: list[LanguageSpan] = []
    for a, b, lang in labelled:
        if spans and spans[-1].language == lang:
            spans[-1].end_char = b
        else:
            spans.append(LanguageSpan(language=lang, start_char=a, end_char=b))
    return spans


def spans_from_word_info(
    words: Sequence[Any],
    text: str,
    fallback: Optional[str] = None,
    offset: float = 0.0,
) -> list[LanguageSpan]:
    """Build TIMED spans from a provider's word-level annotations.

    This is the path that produces the supervisor's Phase 4 schema in full —
    `language`, `start`, `end` — and it is the ONLY path in this module
    permitted to set `timing=TIMING_API`.

    `words` is a sequence of objects or dicts carrying `text`, `start_offset`
    and `end_offset`, which is the shape `gemini-3.5-transcribe` returns under
    `timestamp_granularities: ["word"]`. Note that `word_info` has no
    `language` field, so language is still derived from the script here.

    Offsets arrive as strings like "0.450s"; `offset` is added to convert a
    provider-relative time into an absolute session time.

    NOT WIRED TO A PROVIDER YET, deliberately. The only verified Gemini model
    exposing word timings is `gemini-3.5-transcribe`, whose published language
    table lists neither Sinhala nor Tamil. This function exists so that the
    data model genuinely supports timed spans — Phase 4 asks for the model to
    be CAPABLE of it — and so that enabling them later is a provider change,
    not a schema change. It is covered by tests.
    """
    parsed: list[tuple[str, Optional[float], Optional[float]]] = []
    for w in words:
        if isinstance(w, dict):
            wt, ws, we = w.get("text"), w.get("start_offset"), w.get("end_offset")
        else:
            wt = getattr(w, "text", None)
            ws = getattr(w, "start_offset", None)
            we = getattr(w, "end_offset", None)
        if not wt:
            continue
        parsed.append((str(wt), _seconds(ws), _seconds(we)))

    if not parsed:
        # No usable annotations. Fall back to untimed spans rather than
        # returning nothing — the language structure is still correct.
        return language_spans(text, fallback)

    spans: list[LanguageSpan] = []
    cursor = 0
    for word, ws, we in parsed:
        lang = token_language(word)
        # Locate the word in `text` so char offsets stay authoritative even
        # when the provider's tokenisation differs from ours.
        idx = text.find(word, cursor)
        if idx < 0:
            a = b = cursor
        else:
            a, b = idx, idx + len(word)
            cursor = b

        start = None if ws is None else ws + offset
        end = None if we is None else we + offset
        timing = TIMING_API if (start is not None or end is not None) else TIMING_NONE

        if lang is None:
            if spans:
                # Neutral word: extend the current span rather than opening one.
                spans[-1].end_char = max(spans[-1].end_char, b)
                if end is not None:
                    spans[-1].end = end
                continue
            lang = fallback
            if lang is None:
                continue

        if spans and spans[-1].language == lang:
            spans[-1].end_char = max(spans[-1].end_char, b)
            if end is not None:
                spans[-1].end = end
                spans[-1].timing = timing
        else:
            spans.append(
                LanguageSpan(
                    language=lang,
                    start_char=a,
                    end_char=b,
                    start=start,
                    end=end,
                    timing=timing,
                )
            )
    return spans


def _seconds(value: Any) -> Optional[float]:
    """Parse a duration. Accepts 1.5, "1.5" and the API's "1.500s" form."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if s.endswith("s"):
        s = s[:-1]
    try:
        return float(s)
    except ValueError:
        return None


def merge_spans(spans: Iterable[LanguageSpan]) -> list[LanguageSpan]:
    """Join adjacent spans that share a language.

    Used when concatenating chunks into a paragraph: chunk A ending in English
    followed by chunk B beginning in English is ONE English span across the
    join, not two. Without this, every chunk boundary would look like a
    code-switch, and the switch COUNT — which Phase 15 asks us to measure as
    'code-switch accuracy' — would be inflated by the segmentation.
    """
    out: list[LanguageSpan] = []
    for s in spans:
        if out and out[-1].language == s.language:
            prev = out[-1]
            prev.end_char = max(prev.end_char, s.end_char)
            if s.end is not None:
                prev.end = s.end
            if prev.start is None and s.start is not None:
                prev.start = s.start
            if prev.timing == TIMING_NONE and s.timing != TIMING_NONE:
                prev.timing = s.timing
        else:
            out.append(
                LanguageSpan(
                    language=s.language,
                    start_char=s.start_char,
                    end_char=s.end_char,
                    start=s.start,
                    end=s.end,
                    timing=s.timing,
                )
            )
    return out


def switch_count(spans: Sequence[LanguageSpan]) -> int:
    """Number of language changes. Two spans of one language = 0 switches."""
    return max(0, len(spans) - 1)


def languages_in(spans: Iterable[LanguageSpan]) -> list[str]:
    """Distinct languages, in order of first appearance.

    This is what a paragraph's summary `language` field should be built from:
    it reflects the SPANS, so it cannot disagree with them.
    """
    seen: list[str] = []
    for s in spans:
        if s.language not in seen:
            seen.append(s.language)
    return seen
