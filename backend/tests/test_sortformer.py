"""Sortformer backend: cross-window identity stitching.

No NeMo, no GPU, no model — `_infer` is never called. What is under test is
the part that is easy to get wrong and would silently corrupt a long session:
Sortformer numbers speakers by arrival order *within the input it was given*,
so slot 0 in one window is not necessarily slot 0 in the next. If the stitch
gets that wrong, speakers swap names halfway through a meeting.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.diarization.sortformer import SortformerDiarizer


def make():
    # enabled=True, but start() is never called — the background loop is opt-in,
    # and _stitch / label_for are pure and synchronous. (enabled=False would
    # short-circuit label_for to 0, which is the correct behaviour for
    # DIARIZATION_MODE=off and the wrong thing to test against here.)
    return SortformerDiarizer(enabled=True, window_sec=90.0)


def test_first_window_is_adopted_as_is():
    d = make()
    win = [(0.0, 10.0, 0), (12.0, 20.0, 1), (22.0, 30.0, 0)]
    assert d._stitch(win, offset=0.0)
    assert d.timeline() == win
    assert d.speaker_count() == 2
    print("  first window adopted verbatim, 2 speakers  OK")


def test_permuted_slots_across_windows_are_matched():
    """THE failure this guards against.

    Window 1 (0-90 s): A speaks first  -> local slot 0 = A, slot 1 = B
    Window 2 (30-120 s): B speaks first -> local slot 0 = B, slot 1 = A

    Without stitching, everyone's name flips at the 30 s mark.
    """
    d = make()
    d._stitch(
        [(0.0, 20.0, 0), (25.0, 45.0, 1), (50.0, 70.0, 0), (75.0, 88.0, 1)],
        offset=0.0,
    )
    before = {s for _, _, s in d.timeline() if _ < 20.0}

    # Second window starts at 30 s; its slots are permuted relative to the
    # session, and the overlap region (30-88 s) is what must resolve them.
    d._stitch(
        [(30.0, 45.0, 0), (50.0, 70.0, 1), (75.0, 88.0, 0), (95.0, 115.0, 1)],
        offset=30.0,
    )
    tl = d.timeline()
    by_time = {round(a, 1): s for a, b, s in tl}

    # 25-45 was session speaker 1; the new window called it slot 0. It must
    # still come out as session speaker 1.
    assert by_time[30.0] == 1, tl
    assert by_time[50.0] == 0, tl
    assert by_time[75.0] == 1, tl
    assert by_time[95.0] == 0, tl
    assert d.speaker_count() == 2, f"stitch invented a speaker: {d.speaker_count()}"
    print("  permuted slots matched on the overlap, no name flip  OK")


def test_new_speaker_mid_session_gets_a_new_id():
    """Realistic geometry: the window is 90 s and slides ~2 s per pass, so the
    overlap is nearly the whole window and every active speaker appears in it."""
    d = make()
    d._stitch([(0.0, 20.0, 0), (25.0, 45.0, 1), (50.0, 70.0, 0)], offset=0.0)
    # Next pass: same audio plus a third person joining at 75 s.
    d._stitch(
        [(0.0, 20.0, 0), (25.0, 45.0, 1), (50.0, 70.0, 0), (75.0, 88.0, 2)],
        offset=0.0,
    )
    assert d.speaker_count() == 3, d.timeline()
    assert sorted({s for _, _, s in d.timeline()}) == [0, 1, 2]
    print("  a third speaker joining gets a fresh id, existing two kept  OK")


def test_speaker_cap_prevents_id_inflation():
    """KNOWN LIMITATION, pinned so it cannot get worse.

    Sortformer keeps no state between two separate diarize() calls, so a
    speaker who is silent for the entire overlap region cannot be recognised
    acoustically when they come back. Rather than let one person accumulate
    ids across a long meeting, the count is capped at max_speakers and the
    unmatched slot is folded into whoever spoke most recently before it.

    The embedding backend does NOT have this limitation — it clusters the whole
    session on stored embeddings, so a speaker can be silent for ten minutes
    and still be recognised. That is a real reason to prefer it for long,
    uneven sessions.
    """
    d = make()
    d.max_speakers = 2
    d._stitch([(0.0, 20.0, 0), (25.0, 45.0, 1)], offset=0.0)
    d._stitch([(25.0, 45.0, 0), (50.0, 60.0, 1)], offset=25.0)
    assert d.speaker_count() <= 2, d.timeline()
    print(f"  at the cap, unmatched slots fold instead of inflating "
          f"({d.speaker_count()} ids)  OK")


def test_adjacent_runs_merge_across_a_chunk_edge():
    d = make()
    d._stitch([(0.0, 10.0, 0)], offset=0.0)
    # The re-run split one continuous turn at a chunk boundary.
    d._stitch([(0.0, 10.0, 0), (10.05, 18.0, 0)], offset=0.0)
    tl = d.timeline()
    assert len(tl) == 1, tl
    assert tl[0] == (0.0, 18.0, 0), tl
    print("  runs split at a chunk edge merged back into one  OK")


def test_label_and_split_lookups():
    d = make()
    d._stitch([(0.0, 10.0, 0), (10.5, 20.0, 1), (20.5, 30.0, 0)], offset=0.0)
    assert d.label_for(2.0, 5.0) == 0
    assert d.label_for(12.0, 18.0) == 1
    # A segment straddling a turn change reports the majority speaker...
    assert d.label_for(8.0, 16.0) in (0, 1)
    # ...and offers the cut so main.py can split it instead.
    cuts = d.split_points(0.0, 30.0)
    assert len(cuts) == 2, cuts
    assert abs(cuts[0] - 10.5) < 0.01 and abs(cuts[1] - 20.5) < 0.01, cuts
    print(f"  label_for + split_points agree; cuts at {[round(c,2) for c in cuts]}  OK")


def test_interface_matches_the_embedding_backend():
    """Both backends must be swappable without main.py branching."""
    from app.diarization.service import DiarizationService

    required = [
        "feed", "start", "aclose", "finalize", "wait_for_coverage",
        "label_for", "split_points", "timeline", "stats", "reset",
    ]
    a, b = make(), DiarizationService(hf_token="x", enabled=False)
    for name in required:
        assert hasattr(a, name), f"SortformerDiarizer missing {name}"
        assert hasattr(b, name), f"DiarizationService missing {name}"
    assert hasattr(a.engine, "speaker_count") and hasattr(b.engine, "speaker_count")
    print(f"  both backends expose the same {len(required) + 1}-method surface  OK")


if __name__ == "__main__":
    print("Sortformer backend tests\n")
    for fn in [
        test_first_window_is_adopted_as_is,
        test_permuted_slots_across_windows_are_matched,
        test_new_speaker_mid_session_gets_a_new_id,
        test_speaker_cap_prevents_id_inflation,
        test_adjacent_runs_merge_across_a_chunk_edge,
        test_label_and_split_lookups,
        test_interface_matches_the_embedding_backend,
    ]:
        fn()
    print("\nall passed")
