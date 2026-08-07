"""
validation.py — content checks on ASR output.

WHY THIS EXISTS
---------------
A real transcript from this system contained a paragraph labelled `si`
(Sinhala) whose characters measured:

    Sinhala   232 chars   50.0%
    Thaana    219 chars   47.2%   <- Dhivehi / Maldivian
    Arabic      7 chars    1.5%
    Kannada     5 chars    1.1%

Every guard in the pipeline passed it, because each one was checking the
wrong thing:

  ALLOWED_LANGUAGES     `state.py` checks `result.language not in allowed`.
                        The model returned "si". That is a LABEL check; it
                        says nothing about the characters in `text`.

  response_schema       Constrains `language` to an enum. `text` is an
                        unconstrained STRING — no JSON schema can restrict
                        which script a string is written in.

  MAX_WORDS_PER_SEC     The whole paragraph is ~3.3 words/sec over 33 s; the
                        Thaana burst alone is ~5 w/s in a 9 s segment. Both
                        are far under the 8.0 limit. That guard catches FAST
                        hallucination (a model looping on noise); this one is
                        slow and fluent, so the guard could never fire.

So there was no check on the characters themselves. That is what this module
adds.

WHY THE MODEL DID IT
--------------------
The garbage is a textbook degenerate decoding loop — the token `ހކއހވ`
appears 7 times and a ~12-token phrase repeats nearly verbatim 4 times. The
model lost the audio and started generating from its own prior. Contributing
factors, in rough order of importance:

  1. No script constraint existed, so nothing pushed back.
  2. Sinhala and Thaana are both low-resource South Asian scripts. On unclear
     or noisy Sinhala the audio encoder has weak separation between them.
  3. The segment that produced it ended on `max_len` (9 s), not on a natural
     pause — the longest, least-bounded input the pipeline can produce.
  4. `temperature=0.0`. Greedy decoding is MORE prone to repetition loops
     than low-temperature sampling, not less. This is counter-intuitive and
     worth testing separately.

TWO CHECKS, IN ORDER
--------------------
    1. SCRIPT      Strip tokens written in a script we did not ask for.
                   Drop the segment entirely if most of it was foreign.
    2. REPETITION  Drop text that repeats itself like a stuck decoder —
                   this also catches loops that stay in Sinhala script.

Measured on the real sample: real speech has a unique-token ratio of 0.82 and
a top 3-gram repeated twice; the hallucination scores 0.33 and 4. The
thresholds below sit between those, well clear of both.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Iterable

log = logging.getLogger("voxlive.asr.validation")

# Unicode ranges we accept, per language code.
SCRIPT_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "si": ((0x0D80, 0x0DFF),),  # Sinhala
    "ta": ((0x0B80, 0x0BFF), (0x11FC0, 0x11FFF)),  # Tamil, Tamil Supplement
    "en": ((0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F)),  # Latin
}

# Characters that belong to no script and must never count against a token.
# ZWJ/ZWNJ matter: Sinhala uses U+200D for conjuncts such as ක්‍රියාත්මක.
# Treating it as foreign would reject perfectly good Sinhala.
NEUTRAL_CODEPOINTS = frozenset(
    {
        0x200C,  # ZERO WIDTH NON-JOINER
        0x200D,  # ZERO WIDTH JOINER
        0x00B7,
        0x2018,
        0x2019,
        0x201C,
        0x201D,
        0x2013,
        0x2014,
        0x2026,
    }
)

# Strip a token if MORE than this share of its script-bearing characters are
# foreign. Above zero so a single stray mark inside an otherwise good word
# does not delete the word.
TOKEN_FOREIGN_RATIO = 0.5

# Foreign tokens are ALWAYS removed — they are definitionally wrong, so
# keeping the remainder cannot reintroduce garbage. The segment is therefore
# dropped only when nothing usable survives, not because the foreign SHARE
# was high. An earlier version dropped anything above 60% foreign; measured
# against the real sample that discarded 34 tokens of clean Sinhala (unique
# ratio 1.0, zero foreign characters) alongside the garbage. Losing real
# speech is the worse failure.
MIN_TOKENS_AFTER_STRIP = 3

# Above this share, log at WARNING rather than INFO. Not a drop trigger — a
# monitoring signal. A rising foreign-script rate is the earliest evidence
# that the model, the region or the audio quality has degraded, and it is
# far more useful as a metric than as a silent deletion.
FOREIGN_SHARE_ALARM = 0.25

# Repetition. Only applied once there are enough tokens for the ratio to mean
# something; short utterances legitimately repeat ("ලණු කන්න එපා").
REPETITION_MIN_TOKENS = 12
REPETITION_UNIQUE_RATIO = 0.45  # real speech measured 0.82, garbage 0.33
REPETITION_MAX_NGRAM_REPEATS = 3  # real speech 2, garbage 4


def _is_neutral(ch: str) -> bool:
    """Digits, punctuation, spaces and joiners carry no script information."""
    o = ord(ch)
    if o in NEUTRAL_CODEPOINTS:
        return True
    return not ch.isalpha()


def _in_ranges(ch: str, ranges: Iterable[tuple[int, int]]) -> bool:
    o = ord(ch)
    return any(lo <= o <= hi for lo, hi in ranges)


def allowed_ranges(languages: Iterable[str]) -> tuple[tuple[int, int], ...]:
    """Union of the accepted ranges for the configured languages.

    English is ALWAYS included even if not configured: Sri Lankan speech is
    full of Latin-script acronyms, brand names and numerals, and rejecting
    them would damage genuine code-switched output.
    """
    out: list[tuple[int, int]] = list(SCRIPT_RANGES["en"])
    for code in languages:
        out.extend(SCRIPT_RANGES.get(code, ()))
    return tuple(dict.fromkeys(out))


def script_profile(text: str, ranges: Iterable[tuple[int, int]]) -> tuple[int, int]:
    """(allowed_chars, foreign_chars) among script-bearing characters."""
    allowed = foreign = 0
    for ch in text:
        if _is_neutral(ch):
            continue
        if _in_ranges(ch, ranges):
            allowed += 1
        else:
            foreign += 1
    return allowed, foreign


def strip_foreign_scripts(
    text: str, ranges: Iterable[tuple[int, int]]
) -> tuple[str, float]:
    """Remove whitespace-separated tokens written in an unexpected script.

    Returns the cleaned text and the share of script-bearing characters in the
    ORIGINAL text that were foreign. Token-level rather than character-level:
    deleting individual characters out of a word produces a misspelling, which
    is worse than deleting the word — a reader can see a gap, but cannot see a
    corruption.
    """
    kept: list[str] = []
    total_allowed = total_foreign = 0

    for token in text.split():
        allowed, foreign = script_profile(token, ranges)
        total_allowed += allowed
        total_foreign += foreign
        scripted = allowed + foreign
        if scripted == 0:
            kept.append(token)  # pure punctuation or digits
        elif foreign / scripted <= TOKEN_FOREIGN_RATIO:
            kept.append(token)

    scripted_total = total_allowed + total_foreign
    share = (total_foreign / scripted_total) if scripted_total else 0.0
    return " ".join(kept), share


def looks_degenerate(text: str) -> bool:
    """True when the text repeats itself like a stuck decoder.

    Catches loops that stay INSIDE an allowed script, which the script filter
    cannot see. Two independent signals, either of which is enough:

      * low unique-token ratio — the model is cycling a small vocabulary
      * one 3-gram appearing many times — the model is cycling a phrase
    """
    tokens = text.split()
    if len(tokens) < REPETITION_MIN_TOKENS:
        return False

    if len(set(tokens)) / len(tokens) < REPETITION_UNIQUE_RATIO:
        return True

    grams = Counter(tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2))
    if grams and grams.most_common(1)[0][1] >= REPETITION_MAX_NGRAM_REPEATS:
        return True

    return False


def clean_transcript(text: str, languages: Iterable[str]) -> tuple[str, str | None]:
    """Validate one ASR result.

    Returns `(cleaned_text, reason_if_dropped)`. An empty string with a reason
    means the whole segment should be discarded; an empty reason with shorter
    text means contamination was stripped and the rest is usable.
    """
    if not text.strip():
        return "", None

    ranges = allowed_ranges(languages)
    cleaned, foreign_share = strip_foreign_scripts(text, ranges)

    if len(cleaned.split()) < MIN_TOKENS_AFTER_STRIP:
        return "", f"foreign script {foreign_share:.0%}, nothing usable left"

    if looks_degenerate(cleaned):
        return "", "repetition loop"

    if foreign_share > 0:
        log.log(
            logging.WARNING if foreign_share > FOREIGN_SHARE_ALARM else logging.INFO,
            "stripped %.0f%% foreign-script content from segment",
            foreign_share * 100,
            extra={"event": "asr_foreign_script", "foreign_share": foreign_share},
        )

    return cleaned, None
