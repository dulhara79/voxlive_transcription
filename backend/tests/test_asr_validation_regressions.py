"""
Regression tests for the ASR content guards.

Each test here corresponds to a bug that reached the transcript. The first
class is the important one: it locks in behaviour that was WRONG for the whole
life of the previous validation module, and wrong in the direction that
deletes real speech rather than the direction that lets garbage through — so
it never appeared as a complaint about hallucination, only as turns that
quietly never showed up.
"""

from __future__ import annotations

import pytest

from app.asr.validation import (
    clean_transcript,
    dominant_language,
    echoes_context,
    longest_token_run,
    looks_degenerate,
)

LANGS = ("si", "en", "ta")


class TestShortUtterancesSurvive:
    """`MIN_TOKENS_AFTER_STRIP = 3` dropped every one- and two-word turn.

    The constant was meant to express "too little survived the foreign-script
    strip to be trustworthy", but it was applied to the token count of text
    that had never been stripped. A clean 'ඔව්' measured one token, tripped
    the check, and was discarded with the reason 'foreign script 0%, nothing
    usable left' — a message whose own numbers contradict the decision.

    Short answers are a large share of turns in conversational Sinhala and
    Tamil, so this removed real speech continuously.
    """

    @pytest.mark.parametrize(
        "text",
        [
            "ඔව්",  # yes
            "නෑ",  # no
            "හරි",  # ok / right
            "ස්තූතියි",  # thank you
            "ஆம்",  # yes (Tamil)
            "சரி",  # ok (Tamil)
            "yes okay",
            "thank you",
            "mm hmm",
            "2024",
        ],
    )
    def test_clean_short_text_is_kept(self, text):
        cleaned, dropped = clean_transcript(text, LANGS)
        assert dropped is None, f"{text!r} was dropped: {dropped}"
        assert cleaned == text

    def test_a_clean_segment_is_never_dropped_for_being_short(self):
        """The drop branch requires contamination, so 0% foreign cannot reach it."""
        for text in ("a", "ඔ", "ok"):
            _, dropped = clean_transcript(text, LANGS)
            assert dropped is None


class TestForeignScriptStillRejected:
    """The behaviour the module was built for must not regress."""

    def test_wholly_foreign_text_is_dropped(self):
        cleaned, dropped = clean_transcript("ހކއހވ ހކއހވ ހކއހވ", LANGS)
        assert cleaned == ""
        assert dropped and "foreign script" in dropped

    def test_heavily_contaminated_short_remnant_is_dropped(self):
        """One good token surviving a mostly-Thaana segment is not a transcript."""
        cleaned, dropped = clean_transcript("ඔව් ހކއހވ ހކއހވ", LANGS)
        assert cleaned == ""
        assert dropped and "remnant too short" in dropped

    def test_lightly_contaminated_long_text_keeps_the_good_part(self):
        text = "අපි කරන්න ඕන දේ ހކއހވ තමයි මේක වගේ දෙයක්"
        cleaned, dropped = clean_transcript(text, LANGS)
        assert dropped is None
        assert "ހކއހވ" not in cleaned
        assert "අපි" in cleaned

    def test_sinhala_zwj_conjuncts_are_not_foreign(self):
        """U+200D is structural in Sinhala; treating it as foreign ate words."""
        text = "මෙය ක්‍රියාත්මක කරන්න ඕන දෙයක් වෙනවා"
        cleaned, dropped = clean_transcript(text, LANGS)
        assert dropped is None
        assert "ක්‍රියාත්මක" in cleaned


class TestRepetitionLoops:
    """The n-gram check needed 12 tokens, so short loops passed untouched."""

    @pytest.mark.parametrize(
        "text",
        [
            "ok ok ok ok",
            "හරි හරි හරි හරි හරි",
            "yes. yes. yes. yes.",
        ],
    )
    def test_short_runs_are_caught(self, text):
        _, dropped = clean_transcript(text, LANGS)
        assert dropped == "repetition loop"

    @pytest.mark.parametrize(
        "text",
        [
            "no no I meant the other one",  # doubling for emphasis
            "very very very good",  # tripling is still human
        ],
    )
    def test_natural_emphasis_survives(self, text):
        _, dropped = clean_transcript(text, LANGS)
        assert dropped is None

    def test_run_detector_ignores_trailing_punctuation(self):
        assert longest_token_run("ok, ok. ok! ok".split()) == 4

    def test_long_phrase_loop_still_caught(self):
        text = " ".join(["the budget meeting is on friday"] * 4)
        assert looks_degenerate(text)


class TestContextEcho:
    """The rolling hint was sent to the model and never audited on the way back."""

    def test_output_that_repeats_the_hint_is_dropped(self):
        context = "so anyway we need to finalise the budget before friday please"
        text = "we need to finalise the budget before friday"
        _, dropped = clean_transcript(text, LANGS, context)
        assert dropped and "echoed the context" in dropped

    def test_genuinely_new_speech_passes_with_context_present(self):
        context = "so anyway we need to finalise the budget before friday please"
        text = "I think Nimal already sent the revised figures this morning"
        _, dropped = clean_transcript(text, LANGS, context)
        assert dropped is None

    def test_short_confirmations_are_exempt(self):
        """'හරි' recurring across turns is a person agreeing, not a loop."""
        context = "හරි එහෙනම් අපි හෙට කතා කරමු හරි"
        assert not echoes_context("හරි", context)


class TestLanguageLabel:
    """`language` and `text` are generated independently; only one is constrained."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("ඔව් හරි", "si"),
            ("hello there", "en"),
            ("வணக்கம் நண்பரே", "ta"),
            # A Sinhala sentence carrying an English brand name is a Sinhala
            # turn — Latin must not win on a near-tie.
            ("අපි Google Meet එකෙන් කතා කරමු", "si"),
        ],
    )
    def test_dominant_language(self, text, expected):
        assert dominant_language(text) == expected

    def test_no_script_returns_none(self):
        assert dominant_language("123 456 ...") is None
