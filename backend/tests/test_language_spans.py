"""Phase 4 — word/phrase-level code-switching representation.

Run: python tests/test_language_spans.py

No torch, no pyannote, no network. Everything here is deterministic Unicode
range arithmetic, which is the whole reason the language attribution can be
tested with exact assertions rather than tolerances.

The fixtures are the supervisor's own target conversation, verbatim.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.asr.language_spans import (  # noqa: E402
    TIMING_API,
    TIMING_NONE,
    language_spans,
    languages_in,
    merge_spans,
    spans_from_word_info,
    switch_count,
    token_language,
)
from app.session.transcript import Chunk, TranscriptStore  # noqa: E402

# The target conversation from the brief.
NIMAL_1 = "මචං, where are we going for lunch today? මට ගොඩක් බඩගිනියි, ரொம்ப பசிக்குது!"
SIVA_1 = (
    "நாங்க அந்த புது கடைக்கு போகலாமா? The food there is awesome, පට්ට කෑම තියෙනවා ඒකේ."
)
SARAH_1 = "ஐயோ, I don't want to eat rice today. වෙන මොනවා හරි කමු, எனக்கு பரோட்டா தான் வேணும்."


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_token_language():
    check(token_language("මචං") == "si", "Sinhala token")
    check(token_language("சரி") == "ta", "Tamil token")
    check(token_language("lunch") == "en", "English token")
    check(token_language("2025") is None, "digits are neutral")
    check(token_language("...") is None, "punctuation is neutral")
    # Sinhala uses U+200D ZWJ for conjuncts; it must not count as foreign.
    check(token_language("ක්‍රියාත්මක") == "si", "ZWJ conjunct stays Sinhala")
    print("  token_language: si/ta/en/neutral/ZWJ  OK")


def test_three_languages_in_one_utterance():
    """The headline Phase 4 requirement: si -> en -> si -> ta in ONE turn."""
    spans = language_spans(NIMAL_1, "si")
    langs = [s.language for s in spans]
    check(langs == ["si", "en", "si", "ta"], f"expected si/en/si/ta, got {langs}")
    check(switch_count(spans) == 3, "three switches")
    check(languages_in(spans) == ["si", "en", "ta"], "all three languages present")

    # Offsets must slice back to the exact source text.
    check(spans[0].text_of(NIMAL_1) == "මචං,", "span 0 slices correctly")
    check(
        spans[1].text_of(NIMAL_1) == "where are we going for lunch today?",
        "span 1 slices correctly",
    )
    check(spans[3].text_of(NIMAL_1) == "ரொம்ப பசிக்குது!", "span 3 slices correctly")
    print("  one turn, three languages, four spans, offsets exact  OK")


def test_spans_are_contiguous_and_ordered():
    for text in (NIMAL_1, SIVA_1, SARAH_1):
        spans = language_spans(text, "si")
        for a, b in zip(spans, spans[1:]):
            check(a.end_char <= b.start_char, "spans must not overlap")
            gap = text[a.end_char : b.start_char]
            check(gap.strip() == "", f"gap between spans must be whitespace: {gap!r}")
        check(spans[0].start_char == 0, "first span starts at 0")
        check(spans[-1].end_char == len(text.rstrip()), "last span reaches the end")
    print("  spans ordered, non-overlapping, gap-free across all three turns  OK")


def test_tamil_leading_turn():
    spans = language_spans(SIVA_1, "ta")
    check(
        [s.language for s in spans] == ["ta", "en", "si"],
        f"Siva's turn is ta->en->si, got {[s.language for s in spans]}",
    )
    print("  Tamil-leading turn ta->en->si  OK")


def test_short_protected_utterances_keep_a_language():
    """Phase 5's protected list must never come back as 'no language'."""
    for text, expected in [
        ("ඔව්", "si"),
        ("හරි", "si"),
        ("ஆம்", "ta"),
        ("சரி", "ta"),
        ("yes", "en"),
        ("okay", "en"),
        ("හ්ම්", "si"),
    ]:
        spans = language_spans(text, None)
        check(len(spans) == 1, f"{text!r} should be one span")
        check(spans[0].language == expected, f"{text!r} should be {expected}")
    print("  short backchannels keep their language  OK")


def test_scriptless_text_yields_no_spans():
    """Empty is a real answer, not a failure: a bare number has no language."""
    for text in ("2025", "...", "", "   ", "42 -- 17"):
        check(language_spans(text, None) == [], f"{text!r} should have no spans")
    # ...but with a fallback the caller can still attribute it.
    spans = language_spans("2025", "si")
    check(len(spans) == 1 and spans[0].language == "si", "fallback attributes digits")
    print("  scriptless text -> no spans; fallback honoured  OK")


def test_neutral_tokens_attach_rather_than_split():
    text = "where are we going for lunch today ? මට බඩගිනියි"
    spans = language_spans(text, "en")
    check(
        [s.language for s in spans] == ["en", "si"],
        f"a lone '?' must not split the English run: {[s.language for s in spans]}",
    )
    print("  standalone punctuation does not fragment a run  OK")


def test_no_timings_by_default():
    for s in language_spans(NIMAL_1, "si"):
        check(s.start is None and s.end is None, "no timings from a text-only provider")
        check(s.timing == TIMING_NONE, "timing must be reported as 'none'")
        d = s.as_dict()
        check("start" in d and d["start"] is None, "null timing is emitted explicitly")
    print("  text-only provider -> timing='none', start/end null  OK")


def test_word_info_produces_timed_spans():
    """The Phase 4 schema in full, when a provider actually supplies timings."""
    text = "මචං where are சரி"
    words = [
        {"text": "මචං", "start_offset": "0.10s", "end_offset": "0.70s"},
        {"text": "where", "start_offset": "0.70s", "end_offset": "1.00s"},
        {"text": "are", "start_offset": "1.00s", "end_offset": "2.20s"},
        {"text": "சரி", "start_offset": "2.20s", "end_offset": "3.50s"},
    ]
    spans = spans_from_word_info(words, text, "si", offset=100.0)
    check([s.language for s in spans] == ["si", "en", "ta"], "si/en/ta from word_info")
    check(all(s.timing == TIMING_API for s in spans), "timings marked as API-sourced")
    # offset=100 converts provider-relative into absolute session seconds.
    check(abs(spans[0].start - 100.10) < 1e-6, "start rebased onto the session clock")
    check(abs(spans[1].end - 102.20) < 1e-6, "adjacent words merge, end extends")
    check(abs(spans[2].end - 103.50) < 1e-6, "final span end")
    print("  word_info -> timed si/en/ta spans on the session clock  OK")


def test_word_info_without_timings_falls_back():
    text = "hello හරි"
    spans = spans_from_word_info([{"text": "hello"}, {"text": "හරි"}], text, "en")
    check([s.language for s in spans] == ["en", "si"], "language still derived")
    check(all(s.timing == TIMING_NONE for s in spans), "no timings claimed")
    print("  word_info lacking offsets -> spans without invented timings  OK")


def test_merge_across_chunk_boundary():
    """A VAD cut inside one language must not read as a code-switch."""
    store = TranscriptStore()
    a = store.new_chunk(1, 0.0, 2.0)
    a.text, a.language = "where are we", "en"
    a.language_spans = language_spans(a.text, "en")
    b = store.new_chunk(2, 2.0, 4.0)
    b.text, b.language = "going for lunch", "en"
    b.language_spans = language_spans(b.text, "en")
    a.speaker = b.speaker = 0

    para = store.paragraphs()[0]
    check(para["text"] == "where are we going for lunch", "texts concatenated")
    check(len(para["language_spans"]) == 1, "one English span, not two")
    check(para["language"] == "en", "summary label stays 'en'")
    print("  same-language chunk join merges into one span  OK")


def test_offsets_rebased_when_chunks_concatenate():
    """The silent-failure case: spans stay well-formed but point at the wrong
    words if the join offset is missed."""
    store = TranscriptStore()
    a = store.new_chunk(1, 0.0, 2.0)
    a.text, a.language = "මචං,", "si"
    a.language_spans = language_spans(a.text, "si")
    b = store.new_chunk(2, 2.0, 4.0)
    b.text, b.language = "where are we going?", "en"
    b.language_spans = language_spans(b.text, "en")
    c = store.new_chunk(3, 4.0, 6.0)
    c.text, c.language = "ரொம்ப பசிக்குது", "ta"
    c.language_spans = language_spans(c.text, "ta")
    a.speaker = b.speaker = c.speaker = 0

    para = store.paragraphs()[0]
    text = para["text"]
    spans = para["language_spans"]
    check([s["language"] for s in spans] == ["si", "en", "ta"], "si/en/ta preserved")
    for s in spans:
        sliced = text[s["start_char"] : s["end_char"]]
        check(sliced.strip() != "", f"span slices to real text, got {sliced!r}")
    check(text[spans[0]["start_char"] : spans[0]["end_char"]] == "මචං,", "span 0")
    check(
        text[spans[1]["start_char"] : spans[1]["end_char"]] == "where are we going?",
        "span 1 rebased past the join",
    )
    check(
        text[spans[2]["start_char"] : spans[2]["end_char"]] == "ரொம்ப பசிக்குது",
        "span 2 rebased past two joins",
    )
    check(para["language"] == "si+en+ta", "summary reports all three")
    print("  offsets rebased correctly across two chunk joins  OK")


def test_summary_label_derived_from_spans():
    """A switch INSIDE one chunk is invisible to the per-chunk label. The
    spans see it, and the summary must follow the spans."""
    store = TranscriptStore()
    c = store.new_chunk(1, 0.0, 3.0)
    c.text = NIMAL_1
    c.language = "si"  # one chunk, one label — cannot express the mix
    c.language_spans = language_spans(c.text, "si")
    c.speaker = 0

    para = store.paragraphs()[0]
    check(
        para["language"] == "si+en+ta",
        f"summary must come from spans, got {para['language']!r}",
    )
    check(len(para["language_spans"]) == 4, "four spans survive to the wire")
    print("  intra-chunk switch surfaces in the summary label  OK")


def test_paragraph_without_spans_still_works():
    """Back-compat: a chunk written by anything that does not set spans."""
    store = TranscriptStore()
    c = store.new_chunk(1, 0.0, 1.0)
    c.text, c.language, c.speaker = "hello", "en", 0
    para = store.paragraphs()[0]
    check(para["language_spans"] == [], "no spans is a valid state")
    check(para["language"] == "en", "falls back to the chunk label")
    print("  chunk with no spans degrades cleanly  OK")


def test_merge_spans_preserves_timing_provenance():
    timed = spans_from_word_info(
        [
            {"text": "hello", "start_offset": "0.0s", "end_offset": "0.5s"},
            {"text": "there", "start_offset": "0.5s", "end_offset": "1.0s"},
        ],
        "hello there",
        "en",
    )
    untimed = language_spans("friend", "en")
    merged = merge_spans(timed + [s.shifted(11) for s in untimed])
    check(len(merged) == 1, "all English, one span")
    check(merged[0].timing == TIMING_API, "API provenance survives the merge")
    print("  merge keeps timing provenance  OK")


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
    print("\nall passed")
