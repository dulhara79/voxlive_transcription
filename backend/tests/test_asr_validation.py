# -*- coding: utf-8 -*-
"""Content guards on ASR output.

The sample in `CONTAMINATED` is REAL output from this system: a paragraph the
UI labelled සිංහල whose characters measured 50% Sinhala, 47% Thaana
(Maldivian), 1.5% Arabic and 1.1% Kannada. Every existing guard passed it.
These tests exist so that specific failure cannot come back.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.asr.validation import (
    allowed_ranges,
    clean_transcript,
    looks_degenerate,
    script_profile,
    strip_foreign_scripts,
)

LANGS = ("si", "en", "ta")

CLEAN_SINHALA = (
    "ඔබතුමලා ඔබතුමියලා ආණ්ඩුව හැසිරෙන ඕන විදිහට හැසිරෙන්න. ඔබ දැන් ඉන්නේ "
    "ආණ්ඩුවේ. මිනිහාට අධිකරණ පද්ධතිය ස්වාධීන වෙන්න ඕනේ. හැබැයි පොලිසිය දක්වන "
    "කරුණු මත තමයි ගරු විනිශ්චකාරවරු ක්‍රියාත්මක වෙන්නේ."
)

# The complete run exactly as it appeared: 45 tokens, the phrase repeating
# four times. Measured unique-token ratio 0.33 against 0.82 for real speech.
THAANA_LOOP = (
    "ހކއހޖ ޕނވމ ކނ އހވ އއ ކއރމނވހ އވމނ ހކއހވ ނއހވމ ހހހކހކރމ ހހކހހކ ހވއމ ކމރމނ "
    "އއވމ ހކއހވ އއ ކއރމނވހ އވމނ ހކއހވ ނއހވމ ހހހކހކރމ ހހކހހކ ހވއމ ކމރމނ "
    "އއވމ ހކއހވ އއ އވމނ ހކއހވ ނއހވމ ހހހކހކރމ ހހކހހކ ހވއމ ކމރމނ "
    "އއވމ ހކއހވ އއ ކއރމނވހ އވމނ ހކއހވ ނއހވމ ހހހކހކރރމ ހހކހހކ ހވއމ ކމރމނ"
)

# The same loop cut to two repetitions. This must NOT trip the repetition
# guard: real speech in the same transcript repeats a 3-gram twice, so a
# threshold that flagged this would delete legitimate Sinhala. The script
# filter still removes it — the two guards cover different failures.
THAANA_SHORT = (
    "ހކއހޖ ޕނވމ ކނ އހވ އއ ކއރމނވހ އވމނ ހކއހވ ނއހވމ ހހހކހކރމ ހހކހހކ ހވއމ "
    "ކމރމނ އއވމ ހކއހވ އއ ކއރމނވހ އވމނ ހކއހވ ނއހވމ ހހހކހކރމ ހހކހހކ ހވއމ ކމރމނ"
)

CONTAMINATED = (
    CLEAN_SINHALA
    + " پولیسیہ ಸಂಪೂರ್ಣ "
    + THAANA_LOOP
    + " මෙන්න මේ දේශපාලාඥයින්ට ඔය ලණු කන්න එපා."
)


def test_zwj_is_not_treated_as_foreign():
    """Sinhala conjuncts use U+200D. Rejecting it would break real Sinhala."""
    assert "\u200d" in CLEAN_SINHALA
    allowed, foreign = script_profile(CLEAN_SINHALA, allowed_ranges(LANGS))
    assert foreign == 0, "clean Sinhala must contain zero foreign characters"
    assert allowed > 100


def test_clean_sinhala_passes_untouched():
    out, dropped = clean_transcript(CLEAN_SINHALA, LANGS)
    assert dropped is None
    assert out == " ".join(CLEAN_SINHALA.split())


def test_code_switched_text_survives():
    """Latin acronyms and English words inside Sinhala must NOT be stripped."""
    mixed = "ඔබ දැන් ඉන්නේ IMF එකේ program එකට යටත්ව 2025 වසරේ."
    out, dropped = clean_transcript(mixed, LANGS)
    assert dropped is None
    for word in ("IMF", "program", "2025"):
        assert word in out


def test_tamil_passes():
    out, dropped = clean_transcript("இது ஒரு தமிழ் வாக்கியம் ஆகும்.", LANGS)
    assert dropped is None and out


def test_pure_thaana_loop_is_dropped():
    out, dropped = clean_transcript(THAANA_LOOP, LANGS)
    assert out == ""
    assert dropped is not None


def test_pipeline_sees_segments_not_paragraphs():
    """The UI paragraph spanned five segments; the guard runs per segment.

    A segment that is entirely garbage is dropped; a clean segment beside it
    is untouched. This is the shape of the real failure.
    """
    assert clean_transcript(THAANA_LOOP, LANGS)[0] == ""
    assert clean_transcript(CLEAN_SINHALA, LANGS)[1] is None


def test_contaminated_paragraph_is_cleaned():
    """Worst case — garbage and speech inside ONE segment.

    Real speech must survive. Dropping the whole segment because the foreign
    share was high would discard 34 tokens of clean Sinhala.
    """
    out, dropped = clean_transcript(CONTAMINATED, LANGS)
    assert dropped is None, f"should clean, not drop: {dropped}"

    for garbage in ("ހކއހވ", "پولیسیہ", "ಸಂಪೂರ್ಣ"):
        assert garbage not in out, f"{garbage!r} survived the filter"

    assert "ආණ්ඩුව" in out and "දේශපාලාඥයින්ට" in out

    allowed, foreign = script_profile(out, allowed_ranges(LANGS))
    assert foreign == 0, "cleaned output still contains foreign script"


def test_repetition_separates_real_speech_from_loops():
    """Measured on the real sample: 0.82 unique ratio vs 0.33."""
    assert not looks_degenerate(CLEAN_SINHALA)
    assert looks_degenerate(THAANA_LOOP)


def test_two_repetitions_are_not_flagged_as_a_loop():
    """The repetition guard is deliberately conservative.

    Real speech in this transcript repeats a 3-gram twice. A threshold tight
    enough to catch a 2x repetition would delete genuine Sinhala, so the
    script filter — not the repetition guard — is what removes this one.
    """
    assert not looks_degenerate(THAANA_SHORT)
    out, dropped = clean_transcript(THAANA_SHORT, LANGS)
    assert out == "" and dropped is not None


def test_legitimate_short_repetition_is_kept():
    """'ලණු කන්න එපා' repeats twice in the real transcript. Must survive."""
    text = (
        "මෙන්න මේ දේශපාලාඥයින්ට ඔය ලණු කන්න එපා. මේ පොලිස්පොතුමාට කියන්නේ "
        "මේ ලණු කන්න එපා. මීට පෙර ඔය අපි දීපු ලණු කෑවොත් වලට වැටෙන්න "
        "වෙන්නේ නැද්ද කියලා තමයි ඔය කියන්නේ."
    )
    out, dropped = clean_transcript(text, LANGS)
    assert dropped is None, f"legitimate repetition was rejected: {dropped}"
    assert "ලණු" in out


def test_sinhala_script_loop_is_still_caught():
    """A loop that stays in Sinhala is invisible to the script filter."""
    loop = "ඔව් ඔව් ඔව් ඔව් ඔව් ඔව් ඔව් ඔව් ඔව් ඔව් ඔව් ඔව් ඔව් ඔව්"
    out, dropped = clean_transcript(loop, LANGS)
    assert out == "" and dropped == "repetition loop"


def test_empty_and_whitespace_are_safe():
    for value in ("", "   ", "\n"):
        out, dropped = clean_transcript(value, LANGS)
        assert out == "" and dropped is None


def test_punctuation_only_is_not_foreign():
    out, dropped = clean_transcript("... ,,, !!!", LANGS)
    assert dropped is None
