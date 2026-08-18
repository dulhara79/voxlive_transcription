"""
validation.py — content checks on ASR output.

WHAT CHANGED IN THIS REVISION, AND WHY
--------------------------------------
The previous version had the drop rule backwards. It dropped on TOKEN COUNT:

    if len(cleaned.split()) < MIN_TOKENS_AFTER_STRIP:   # 3
        return "", "foreign script ..., nothing usable left"

`MIN_TOKENS_AFTER_STRIP` was written to mean "too little survived the strip to
trust", but it was applied to text that had never been stripped at all. The
measured consequence:

    'ඔව්'       -> dropped   ("foreign script 0%, nothing usable left")
    'හරි'       -> dropped
    'thank you' -> dropped
    'ஆம் சரி'   -> dropped
    'ok ok ok ok ok ok ok' -> KEPT

Every one- and two-word turn was silently deleted while a seven-token decoder
loop passed. In live Sinhala/Tamil conversation short answers are a large
share of all turns, so this was removing real speech continuously and only
ever surfacing in the logs as a foreign-script warning.

The rule is now conditional on CONTAMINATION, not on length:

    * nothing survived the strip                     -> drop
    * most of the text was foreign AND the remnant
      is too short to stand on its own               -> drop
    * otherwise                                      -> keep what survived

A clean short utterance has a foreign share of 0.0, so it is never eligible
for the second branch and is always kept.

THE ORIGINAL FAILURE THIS MODULE EXISTS FOR
-------------------------------------------
A real transcript from this system contained a paragraph labelled `si`
(Sinhala) whose characters measured:

    Sinhala   232 chars   50.0%
    Thaana    219 chars   47.2%   <- Dhivehi / Maldivian
    Arabic      7 chars    1.5%
    Kannada     5 chars    1.1%

Every guard in the pipeline passed it, because each was checking the wrong
thing:

  ALLOWED_LANGUAGES     `state.py` checks `result.language not in allowed`.
                        The model returned "si". That is a LABEL check; it
                        says nothing about the characters in `text`.

  response_schema       Constrains `language` to an enum. `text` is an
                        unconstrained STRING — no JSON schema can restrict
                        which script a string is written in.

  MAX_WORDS_PER_SEC     The paragraph is ~3.3 words/sec over 33 s. That guard
                        catches FAST hallucination (a model looping on noise);
                        this one was slow and fluent, so it could never fire.

FOUR CHECKS, IN ORDER
---------------------
    1. SCRIPT       Strip tokens written in a script we did not ask for.
    2. RUN          Drop text containing an immediate run of one repeated
                    token — the shortest, most obvious decoder loop, and the
                    one the n-gram check below is too coarse to see.
    3. REPETITION   Drop text that repeats itself across a longer window.
    4. ECHO         Drop text that is mostly a copy of the rolling context we
                    sent as a hint. The model continued the prompt instead of
                    reading the audio.

Checks 2 and 4 are new. Check 4 closes a gap the README already claimed was
closed: there was no context-echo guard anywhere in the pipeline, only an
exact-match comparison against the single previous segment in `state.py`.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Iterable, Optional

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

# Drop the whole segment only when the text was BADLY contaminated and what
# survived is too short to stand on its own. BOTH conditions are required.
# A clean segment has foreign_share == 0.0 and can never reach this branch,
# which is precisely the bug this replaced.
HEAVY_CONTAMINATION_SHARE = 0.5
MIN_TOKENS_AFTER_HEAVY_STRIP = 3

# Above this share, log at WARNING rather than INFO. Not a drop trigger — a
# monitoring signal. A rising foreign-script rate is the earliest evidence
# that the model, the region or the audio quality has degraded.
FOREIGN_SHARE_ALARM = 0.25

# --- repetition -------------------------------------------------------------
# An immediate run of the SAME token. Catches "ok ok ok ok" and
# "හරි හරි හරි හරි", which the 3-gram check cannot see because it needs 12
# tokens before it will look at anything. Four is deliberately conservative:
# natural speech repeats a word twice for emphasis and occasionally three
# times, but four identical tokens in a row is a decoder loop.
MAX_TOKEN_RUN = 4

# Longer-window repetition. Only applied once there are enough tokens for the
# ratio to mean something; short utterances legitimately repeat.
REPETITION_MIN_TOKENS = 12
REPETITION_UNIQUE_RATIO = 0.45  # real speech measured 0.82, garbage 0.33
REPETITION_MAX_NGRAM_REPEATS = 3  # real speech 2, garbage 4

# --- context echo -----------------------------------------------------------
# The provider sends recent transcript as a spelling hint. When the model
# loses the audio it sometimes continues that hint instead. Measured as the
# share of the output's 3-grams that already appear in the context.
ECHO_MIN_TOKENS = 5
ECHO_CONTAINMENT = 0.80


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


def dominant_language(text: str) -> Optional[str]:
    """Which of si / ta / en the text is mostly written in, or None.

    Used to CORRECT the model's own `language` label. The label and the text
    are produced independently — the response schema constrains one and not
    the other — so a Sinhala-script paragraph labelled `en` is a routine
    outcome, and the frontend colours turns by that label.
    """
    counts: dict[str, int] = {}
    for code, ranges in SCRIPT_RANGES.items():
        counts[code] = sum(
            1 for ch in text if not _is_neutral(ch) and _in_ranges(ch, ranges)
        )
    total = sum(counts.values())
    if total == 0:
        return None
    # Sinhala and Tamil beat Latin on ties: a code-switched Sinhala sentence
    # carrying two English brand names is a Sinhala turn, not an English one.
    for code in ("si", "ta"):
        if counts[code] and counts[code] >= counts["en"]:
            return code
    best = max(counts, key=lambda c: counts[c])
    return best if counts[best] else None


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


def longest_token_run(tokens: list[str]) -> int:
    """Length of the longest run of consecutive identical tokens."""
    best = run = 0
    previous: Optional[str] = None
    for token in tokens:
        key = token.strip(".,!?…:;").casefold()
        if key and key == previous:
            run += 1
        else:
            run = 1
            previous = key
        best = max(best, run)
    return best


def looks_degenerate(text: str) -> bool:
    """True when the text repeats itself like a stuck decoder.

    Catches loops that stay INSIDE an allowed script, which the script filter
    cannot see. Three independent signals, any of which is enough:

      * an immediate run of one token — the shortest possible loop
      * low unique-token ratio — the model is cycling a small vocabulary
      * one 3-gram appearing many times — the model is cycling a phrase
    """
    tokens = text.split()
    if not tokens:
        return False

    if longest_token_run(tokens) >= MAX_TOKEN_RUN:
        return True

    if len(tokens) < REPETITION_MIN_TOKENS:
        return False

    if len(set(tokens)) / len(tokens) < REPETITION_UNIQUE_RATIO:
        return True

    grams = Counter(tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2))
    if grams and grams.most_common(1)[0][1] >= REPETITION_MAX_NGRAM_REPEATS:
        return True

    return False


def echoes_context(text: str, context: Optional[str]) -> bool:
    """True when the output is mostly a copy of the context hint.

    The provider sends recent transcript so the model spells names
    consistently. A model that has lost the audio will sometimes transcribe
    that hint instead — producing fluent, correct-looking text that describes
    audio from ten seconds ago. Nothing else in the pipeline can see this:
    the script is right, the language is right, and it does not repeat itself.

    Containment rather than similarity, because the echo is usually a SUBSET
    of the context, not the whole of it.
    """
    if not context:
        return False

    tokens = text.split()
    if len(tokens) < ECHO_MIN_TOKENS:
        # Short confirmations legitimately recur across turns. Dropping "හරි"
        # because it appeared thirty seconds ago is the same class of mistake
        # this module was rewritten to remove.
        return False

    ctx_tokens = context.split()
    if len(ctx_tokens) < 3:
        return False

    ctx_grams = {
        tuple(t.casefold() for t in ctx_tokens[i : i + 3])
        for i in range(len(ctx_tokens) - 2)
    }
    grams = [
        tuple(t.casefold() for t in tokens[i : i + 3]) for i in range(len(tokens) - 2)
    ]
    if not grams:
        return False

    contained = sum(1 for g in grams if g in ctx_grams) / len(grams)
    return contained >= ECHO_CONTAINMENT


def clean_transcript(
    text: str,
    languages: Iterable[str],
    context: Optional[str] = None,
) -> tuple[str, str | None]:
    """Validate one ASR result.

    Returns `(cleaned_text, reason_if_dropped)`. An empty string with a reason
    means the whole segment should be discarded; an empty reason with shorter
    text means contamination was stripped and the rest is usable.
    """
    if not text.strip():
        return "", None

    ranges = allowed_ranges(languages)
    cleaned, foreign_share = strip_foreign_scripts(text, ranges)
    tokens = cleaned.split()

    if not tokens:
        return "", f"foreign script {foreign_share:.0%}, nothing survived"

    # Only heavy contamination justifies discarding the remainder. A clean
    # segment has foreign_share == 0.0 and never reaches this branch, however
    # short it is.
    if (
        foreign_share >= HEAVY_CONTAMINATION_SHARE
        and len(tokens) < MIN_TOKENS_AFTER_HEAVY_STRIP
    ):
        return "", f"foreign script {foreign_share:.0%}, remnant too short to trust"

    if looks_degenerate(cleaned):
        return "", "repetition loop"

    if echoes_context(cleaned, context):
        return "", "echoed the context hint instead of the audio"

    if foreign_share > 0:
        log.log(
            logging.WARNING if foreign_share > FOREIGN_SHARE_ALARM else logging.INFO,
            "stripped %.0f%% foreign-script content from segment",
            foreign_share * 100,
            extra={"event": "asr_foreign_script", "foreign_share": foreign_share},
        )

    return cleaned, None