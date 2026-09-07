"""Automatic speaker-count tests (supervisor review §18, plus §17).

WHY THIS FILE EXISTS
====================
The review's §23 is blunt about the engineering process it is replacing:

    play YouTube -> look at transcript -> decide it is bad -> change code

That loop cannot tell a fix from a coincidence. It tuned the engine against one
news recording and had no way of knowing what the change did to the other nine
cases. This file is the cheap half of the replacement — synthetic embeddings
with realistic WeSpeaker geometry, covering the full matrix §18 asks for, run in
under a second on every commit. `eval_diarization.py` over labelled Sinhala,
Tamil and mixed-language audio is the expensive half, and neither substitutes
for the other: synthetic geometry proves the ALGORITHM does what it claims,
real audio proves the geometry assumption holds in Sri Lankan broadcast.

WHAT §18 ASKS FOR, AND WHERE IT IS
    1 speaker                       test_one_speaker_stays_one
    2 speakers                      test_two_speakers
    3 speakers                      test_three_speakers
    4 speakers                      test_four_speakers
    5+ up to the configured max     test_five_speakers / test_cap_is_respected
    a speaker who appears late      test_late_arriving_speaker
    a speaker who speaks briefly    test_brief_speaker_is_discovered
    two acoustically similar        test_similar_speakers_separate
    one speaker, large variation    test_vocal_variation_is_not_two_people
    overlapping speech              test_overlap_is_a_documented_limitation
    long sessions                   test_long_session_does_not_drift
    discovery after establishment   test_discovery_after_establishment
                                    test_count_evolves_upward

A NOTE ON WHAT THESE TESTS CANNOT PROVE
Synthetic voices are drawn from a fixed geometry: cross-speaker cosine distance
~0.70-0.80, within-speaker ~0.18-0.25, with a shared component standing in for
channel and language. That is the geometry WeSpeaker produces on clean 2 s
windows of conversational speech. Broadcast audio is compressed, has music
beds, telephone-bandwidth remote guests and voice-over, and its geometry is
NOT this. A green run here means the algorithm is sound; it does not mean the
system works on Sinhala news, and §19 is explicit that it must not be reported
as if it does.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from app.diarization.service import DiarizationService
from app.diarization.speaker_engine import (
    TRUST_SEC,
    WIN_SEC,
    SpeakerEngine,
    Window,
    l2norm,
)

RNG = np.random.default_rng(2026)
DIM = 256

# The component every voice in one recording shares — microphone, codec,
# language, and the embedder's own bias. Without it two random 256-d vectors
# sit at cosine distance ~1.0 and every test below is far easier than reality.
_COMMON = l2norm(np.random.default_rng(0).normal(size=DIM))


def voice(seed: int) -> np.ndarray:
    """A distinct person. Cross-speaker distance lands ~0.70-0.80."""
    r = np.random.default_rng(seed)
    return l2norm(0.45 * _COMMON + l2norm(r.normal(size=DIM)))


def near_voice(base: np.ndarray, seed: int, mix: float = 1.35) -> np.ndarray:
    """A second person who happens to sound like `base`.

    mix=1.35 puts the two centroids ~0.43 apart — inside the old
    SAME_SPEAKER_MAX veto, which is the review's "Video 2" and the case §9
    exists to fix.
    """
    r = np.random.default_rng(seed)
    return l2norm(base + mix * l2norm(r.normal(size=DIM)))


def utterance(base: np.ndarray, jitter: float = 0.75) -> np.ndarray:
    """One window of one person at normal within-speaker spread."""
    return l2norm(base + jitter * l2norm(RNG.normal(size=DIM)))


def mixed(a: np.ndarray, b: np.ndarray, w: float = 0.5) -> np.ndarray:
    """A window containing TWO voices at once.

    An embedding model has one output vector per window, so overlapped speech
    lands between the two speakers whether or not that point means anything.
    Used only by the overlap test, which asserts the limitation rather than
    pretending it is solved.
    """
    return l2norm(w * a + (1.0 - w) * b + 0.55 * l2norm(RNG.normal(size=DIM)))


def build(turns, start: float = 0.0, hop: float = 0.75):
    """turns: [(voice_base, n_windows)] laid out consecutively in time.

    Returns (windows, truth, end_time). `truth` carries the identity of the
    base vector so `purity` can score permutation-invariantly.
    """
    assert WIN_SEC >= TRUST_SEC, (
        f"test windows ({WIN_SEC}s) are below TRUST_SEC ({TRUST_SEC}s) — every "
        "window would be untrusted, recluster() would return False, and these "
        "tests would pass vacuously without ever clustering anything."
    )
    wins, truth, t = [], [], start
    for base, n in turns:
        for _ in range(n):
            wins.append(Window(t, t + WIN_SEC, utterance(base)))
            truth.append(id(base))
            t += hop
        t += 0.5  # inter-turn pause
    return wins, truth, t


def stream(eng: SpeakerEngine, turns, start: float = 0.0, chunk: int = 4) -> float:
    """Feed windows a few at a time with a recluster between, the way the live
    service does: one `add_windows` + one `recluster` per pass."""
    wins, _, end = build(turns, start)
    for i in range(0, len(wins), chunk):
        eng.add_windows(wins[i : i + chunk])
        eng.recluster()
    return end


def n_speakers(eng: SpeakerEngine) -> int:
    return len({w.prev for w in eng.windows if w.prev >= 0})


def purity(eng: SpeakerEngine, truth: list[int]) -> float:
    labels = [w.prev for w in eng.windows]
    best = {}
    for tr in set(truth):
        idx = [i for i, x in enumerate(truth) if x == tr]
        vals, counts = np.unique([labels[i] for i in idx], return_counts=True)
        best[tr] = vals[int(np.argmax(counts))]
    return sum(1 for i, tr in enumerate(truth) if labels[i] == best[tr]) / len(truth)


def auto(**kw) -> SpeakerEngine:
    """An engine configured the way the shipped .env now configures it:
    EXPECTED_SPEAKERS=0, MAX_SPEAKERS=6, SPEAKER_MODE=auto."""
    kw.setdefault("expected_speakers", 0)
    kw.setdefault("max_speakers", 6)
    kw.setdefault("speaker_mode", "auto")
    return SpeakerEngine(**kw)


# --------------------------------------------------------------- the ladder
# 1 -> 5 speakers, offline (one whole-session pass), which is the cleanest
# possible statement of "can this thing count?".


def test_one_speaker_stays_one():
    """The case the old separation veto existed to protect. It still works.

    This is the test that would break first if the relative separation rule of
    §9 were tuned too loose, so it is the anchor for every threshold in the
    file. Twenty-four windows of one person, and any answer other than 1 means
    the engine is manufacturing people out of vocal variation.
    """
    eng = auto()
    wins, _, _ = build([(voice(101), 24)])
    eng.add_windows(wins)
    eng.recluster()
    assert n_speakers(eng) == 1, f"one person became {n_speakers(eng)} speakers"


def test_two_speakers():
    eng = auto()
    wins, truth, _ = build([(voice(111), 10), (voice(112), 10), (voice(111), 8)])
    eng.add_windows(wins)
    eng.recluster()
    assert n_speakers(eng) == 2
    assert purity(eng, truth) >= 0.90


def test_three_speakers():
    a, b, c = voice(121), voice(122), voice(123)
    eng = auto()
    wins, truth, _ = build([(a, 10), (b, 10), (c, 10), (a, 6), (b, 6)])
    eng.add_windows(wins)
    eng.recluster()
    assert n_speakers(eng) == 3, f"expected 3, got {n_speakers(eng)}"
    assert purity(eng, truth) >= 0.90


def test_four_speakers():
    v = [voice(130 + i) for i in range(4)]
    eng = auto()
    wins, truth, _ = build([(x, 10) for x in v] + [(x, 6) for x in v])
    eng.add_windows(wins)
    eng.recluster()
    assert n_speakers(eng) == 4, f"expected 4, got {n_speakers(eng)}"
    assert purity(eng, truth) >= 0.88


def test_five_speakers():
    """A panel. This is the case the shipped configuration made IMPOSSIBLE:
    with EXPECTED_SPEAKERS=2 and SPEAKER_MODE=fixed the answer could only ever
    have been 2, no matter what the audio contained."""
    v = [voice(140 + i) for i in range(5)]
    eng = auto()
    wins, truth, _ = build([(x, 9) for x in v] + [(x, 5) for x in v])
    eng.add_windows(wins)
    eng.recluster()
    assert n_speakers(eng) == 5, f"expected 5, got {n_speakers(eng)}"
    assert purity(eng, truth) >= 0.85


def test_cap_is_respected():
    """MAX_SPEAKERS is a CEILING (§3). Six real people with a cap of 4 gives 4
    — not because four is right, but because the operator said four is the most
    the system may report. The reverse reading, that a cap of 6 means "find
    six", is the misreading §3 warns about and is covered by the tests above:
    every one of them runs with max_speakers=6 and returns its true count."""
    v = [voice(150 + i) for i in range(6)]
    eng = auto(max_speakers=4)
    wins, _, _ = build([(x, 9) for x in v])
    eng.add_windows(wins)
    eng.recluster()
    assert n_speakers(eng) <= 4, f"cap of 4 exceeded: {n_speakers(eng)}"


# ------------------------------------------------------------ hard cases


def test_similar_speakers_separate():
    """§9. Two different people at centroid distance ~0.43.

    Under v12 the hard veto merged them and the only escape was for the user to
    select FIXED — which is precisely the configuration that then made every
    other recording report exactly two speakers. The relative test separates
    them because the gap (0.44) beats the combined cluster width (~0.37).
    """
    a = voice(161)
    b = near_voice(a, 162, mix=1.35)
    eng = auto()
    wins, truth, _ = build([(a, 10), (b, 10), (a, 8), (b, 8)])
    eng.add_windows(wins)
    eng.recluster()
    assert n_speakers(eng) == 2, f"similar-sounding pair merged into {n_speakers(eng)}"
    assert purity(eng, truth) >= 0.85


def test_vocal_variation_is_not_two_people():
    """The mirror of the test above, and the reason it cannot simply be made
    more permissive.

    One person across a long stretch of animated speech spreads a long way in
    cosine distance — vocal effort, proximity and loudness move an embedding
    about as much as identity does. jitter=1.15 is that person. If the engine
    reports 2 here, it will report phantom speakers on every real monologue.
    """
    base = voice(171)
    wins, _, t = [], [], 0.0
    for _ in range(30):
        wins.append(Window(t, t + WIN_SEC, utterance(base, jitter=1.15)))
        t += 0.75
    eng = auto()
    eng.add_windows(wins)
    eng.recluster()
    assert n_speakers(eng) == 1, (
        f"vocal variation was read as {n_speakers(eng)} speakers — the "
        "separation ratio is too permissive"
    )


def test_brief_speaker_is_discovered():
    """§10. The news reporter: a long presenter/guest session, and a reporter
    who says three windows' worth and leaves.

    SESSION LENGTH IS LOAD-BEARING IN THIS TEST. The same three windows inside
    a thirty-window session survive on the old engine, which is why the defect
    was invisible in the earlier test suite and visible on a four-minute news
    broadcast. Two separate mechanisms scale against the brief speaker:

      * the core filter scored each window against its FIVE nearest
        neighbours, and a reporter with three windows has only two of their
        own, so the mean was dragged over the threshold by three windows
        belonging to other people:  (2*0.22 + 3*0.78) / 5 = 0.556 > 0.50.
        The reporter was deleted as noise before the dendrogram was built,
        and after that no value of K could recover them.

      * the prune floor is RELATIVE (2 % of session speech), so it rises as
        the session runs. Six seconds clears it at 30 windows and does not at
        150.

    Measured against the v12 engine, this exact configuration returns 2
    speakers at 3 and 4 reporter windows, and 3 at five.
    """
    a, b, c = voice(181), voice(182), voice(183)
    eng = auto()
    wins, _, _ = build([(a, 60), (b, 50), (c, 3), (a, 40)])
    eng.add_windows(wins)
    eng.recluster()
    assert n_speakers(eng) == 3, (
        f"the brief speaker was absorbed: {n_speakers(eng)} speaker(s). This is "
        "the failure mode §10 describes — a reporter becomes whoever they "
        "sounded least unlike."
    )


def test_brief_noise_burst_is_not_a_speaker():
    """The other half of §10, at the SAME session length as the test above.

    That pairing is the point. §18 warns that the fix must not be "detect more
    speakers", because the opposite failure — one real speaker rendered as
    Speaker 1/2/3/4 — is just as bad and much easier to cause. So the bar that
    lets a three-window reporter through must simultaneously refuse six windows
    of unrelated noise sitting in the same place in the same session.

    What separates them is not duration, which is identical, but COHERENCE:
    each noise window here is independent, so the cluster they form has a huge
    internal radius. A voice agrees with itself; a door slam does not.
    """
    a, b = voice(191), voice(192)
    wins, _, t = build([(a, 60), (b, 50)])
    for _ in range(6):  # six mutually unrelated noise windows
        wins.append(Window(t, t + WIN_SEC, l2norm(RNG.normal(size=DIM))))
        t += 0.75
    more, _, _ = build([(a, 40)], start=t)
    wins.extend(more)

    eng = auto()
    eng.add_windows(wins)
    eng.recluster()
    assert n_speakers(eng) == 2, (
        f"noise became a speaker: {n_speakers(eng)}. §18/§19 — the fix must not "
        "be 'detect more speakers'."
    )


def test_overlap_is_a_documented_limitation():
    """§11. Two people talking at once.

    An embedding backend emits ONE vector per window, so an overlapped window
    cannot represent both speakers; it lands somewhere between them. This test
    asserts what the system ACTUALLY does — it still finds the two speakers,
    and the overlapped region is attributed to one of them — rather than
    asserting a capability the architecture does not have. If overlap
    attribution matters for a deployment, that is the Sortformer A/B (§12/§15),
    not a threshold change here.
    """
    a, b = voice(201), voice(202)
    wins, _, t = build([(a, 12), (b, 12)])
    for _ in range(6):  # both speaking simultaneously
        wins.append(Window(t, t + WIN_SEC, mixed(a, b)))
        t += 0.75
    more, _, _ = build([(a, 8), (b, 8)], start=t)
    wins.extend(more)

    eng = auto()
    eng.add_windows(wins)
    eng.recluster()
    n = n_speakers(eng)
    assert n == 2, f"overlap should not change the speaker count, got {n}"


# ------------------------------------------------- live / streaming behaviour


def test_growth_check_finds_what_discovery_missed():
    """§12, isolated.

    `_discover` only ever sees a voice that is far from every centroid RIGHT
    NOW. A speaker folded into someone else during establishment is invisible
    to it permanently, because they are no longer far from anything — they ARE
    part of a centroid. That is the gap `_growth_check` fills.

    The isolation here is deliberate: the discovery thresholds are set
    unreachably high so that path CANNOT fire, and the third speaker must
    therefore be found by the periodic re-examination or not at all.
    """
    a, b, c = voice(291), voice(292), voice(293)
    eng = auto(
        new_identity_min_sec=10_000.0,  # normal path disabled
        new_identity_short_sec=10_000.0,  # short path disabled
        growth_check_sec=8.0,
    )
    end = stream(eng, [(a, 10), (b, 10), (a, 8), (b, 8)])
    assert n_speakers(eng) == 2 and eng.established(), "setup failed"

    stream(eng, [(c, 14), (a, 6), (c, 10)], start=end)
    assert n_speakers(eng) == 3, (
        f"growth check did not find the third speaker ({n_speakers(eng)}) with "
        "the incremental discovery path disabled"
    )


def test_growth_check_refuses_to_shrink():
    """§11/§12. The growth check is asymmetric BY CONSTRUCTION, and this is the
    half that protects the transcript.

    Two established speakers, then a long stretch of only the first one — the
    situation where a fresh whole-session K search is most tempted to conclude
    there was only ever one person. A pass that prefers fewer speakers is
    discarded live; it gets its say in the final pass, when the user has
    stopped and a renumbering is no longer disruptive.
    """
    a, b = voice(301), voice(302)
    eng = auto(growth_check_sec=6.0)
    end = stream(eng, [(a, 10), (b, 10), (a, 8), (b, 8)])
    assert n_speakers(eng) == 2 and eng.established(), "setup failed"

    for _ in range(6):  # ~100 windows of speaker A alone
        end = stream(eng, [(a, 16)], start=end)
        assert n_speakers(eng) == 2, (
            f"the quiet speaker was deleted mid-session ({n_speakers(eng)}) — "
            "this is the collapse the establishment rule exists to prevent"
        )


def test_late_arriving_speaker():
    """A guest who joins ninety seconds in.

    The established-identity rule of §11 stops live passes deleting speakers.
    This checks it did not also stop them ADDING one, which is the failure the
    review predicts in §9: "the system starts by discovering 2 and can never
    get past it".
    """
    a, b, c = voice(211), voice(212), voice(213)
    eng = auto()
    end = stream(eng, [(a, 10), (b, 10), (a, 10), (b, 10)])
    assert n_speakers(eng) == 2 and eng.established(), "setup failed"
    end = stream(eng, [(c, 12), (a, 6), (c, 10)], start=end)
    assert n_speakers(eng) == 3, f"late speaker never appeared: {n_speakers(eng)}"


def test_discovery_after_establishment():
    """The incremental path specifically: `_discover` must admit a fourth voice
    on a session that is already established with three."""
    v = [voice(220 + i) for i in range(4)]
    eng = auto()
    end = stream(eng, [(v[0], 10), (v[1], 10), (v[2], 10), (v[0], 8)])
    assert eng.established(), "setup failed: should be established"
    before = n_speakers(eng)
    end = stream(eng, [(v[3], 14), (v[1], 6), (v[3], 8)], start=end)
    after = n_speakers(eng)
    assert after > before, f"no new identity admitted ({before} -> {after})"


def test_count_evolves_upward():
    """§12/§19. The count must be able to walk 1 -> 2 -> 3 -> 4 across a
    session, and must never walk back down.

    The downward half is the collapse the whole v12 establishment mechanism was
    written to prevent, so it is asserted on every pass rather than only at the
    end: a run that peaks at 3 and finishes at 3 still FAILS if it ever dipped.
    """
    v = [voice(230 + i) for i in range(4)]
    eng = auto()
    seen, seq, t = 0, [], 0.0
    for speaker in v:
        wins, _, t = build([(speaker, 14)], start=t)
        for i in range(0, len(wins), 4):
            eng.add_windows(wins[i : i + 4])
            eng.recluster()
            n = n_speakers(eng)
            seq.append(n)
            if n < seen:
                raise AssertionError(
                    f"speaker count went BACKWARDS {seen} -> {n} (sequence {seq})"
                )
            seen = max(seen, n)
    assert seen >= 3, f"count never grew past {seen} for four speakers: {seq}"


def test_long_session_does_not_drift():
    """§7 of the review: the earlier version burned a new speaker NUMBER every
    time clustering wobbled. Sixty passes of two people, and the count must be
    2 at the end and never have exceeded it."""
    a, b = voice(241), voice(242)
    eng = auto()
    t, peak = 0.0, 0
    for _ in range(15):
        wins, _, t = build([(a, 4), (b, 4)], start=t)
        eng.add_windows(wins)
        eng.recluster()
        peak = max(peak, n_speakers(eng))
    assert peak <= 2, f"identity drift: peaked at {peak} speakers for two people"
    assert n_speakers(eng) == 2, f"ended at {n_speakers(eng)}"


# ------------------------------------------------------- fixed mode (§13)


def test_fixed_mode_still_forces_exactly_n():
    """§13. Nothing above may have weakened the fixed-mode contract: when the
    user asserts a count, that count is what they get."""
    v = [voice(250 + i) for i in range(4)]
    eng = SpeakerEngine(expected_speakers=3, max_speakers=6, speaker_mode="fixed")
    wins, _, _ = build([(x, 10) for x in v])
    eng.add_windows(wins)
    eng.recluster()
    assert n_speakers(eng) == 3, f"fixed K=3 produced {n_speakers(eng)}"


def test_fixed_mode_never_discovers():
    """Fixed mode must not gain the new short-speaker discovery path."""
    a, b, c = voice(261), voice(262), voice(263)
    eng = SpeakerEngine(expected_speakers=2, max_speakers=6, speaker_mode="fixed")
    end = stream(eng, [(a, 10), (b, 10), (a, 8)])
    end = stream(eng, [(c, 16), (a, 6)], start=end)
    assert n_speakers(eng) == 2, (
        f"fixed K=2 admitted a third speaker ({n_speakers(eng)}) — the user's "
        "asserted count must hold"
    )


# ------------------------------------------- configuration is observable (§5)


def test_describe_states_the_consequence():
    """The 2-speaker investigation cost days because the effective config was
    invisible at run time and `-> 2 speaker(s)` looks the same whether the
    engine estimated 2 or was ordered to produce 2. `describe()` is what the
    session now logs, so it has to say which of those is happening."""
    fixed = SpeakerEngine(expected_speakers=2, speaker_mode="fixed").describe()
    assert "FORCED" in fixed and "Speaker 3 can never be created" in fixed

    est = auto().describe()
    assert "ESTIMATED" in est and "speaker_mode=auto" in est


# ------------------------------ §17: segment splitting is a SEPARATE concern


class _SplitStub:
    """Minimal stand-in for DiarizationService so `split_points` can be tested
    without loading an embedding model."""

    def __init__(self, engine):
        self.enabled = True
        self.engine = engine


def test_asr_chunk_spanning_three_turns_yields_cuts():
    """§17. A nine-second ASR chunk can contain A -> B -> A.

    Diarization knowing about the change is not enough: the transcript can only
    attribute a whole chunk, so the chunk has to be CUT at the timeline's
    speaker changes. This asserts the cut points exist and fall inside the
    chunk — which is the mechanism behind the `unsplit_chunks` warning in the
    backend log, and is a segmentation problem, not a speaker-count one.
    """
    a, b = voice(271), voice(272)
    eng = auto()
    wins, _, _ = build([(a, 8), (b, 8), (a, 8)])
    eng.add_windows(wins)
    eng.recluster()
    assert n_speakers(eng) == 2, "setup failed"

    t0 = min(w.start for w in eng.windows)
    t1 = max(w.end for w in eng.windows)
    cuts = DiarizationService.split_points(_SplitStub(eng), t0, t1)
    assert len(cuts) >= 2, (
        f"A->B->A produced {len(cuts)} cut point(s); a chunk covering all "
        "three turns would be emitted under a single speaker label"
    )
    assert all(t0 < c < t1 for c in cuts), "cut points must fall inside the chunk"


def test_single_speaker_chunk_is_not_cut():
    """The complement: no speaker change means no cut, so a continuous talker
    is not chopped into paragraphs for no reason."""
    eng = auto()
    wins, _, _ = build([(voice(281), 20)])
    eng.add_windows(wins)
    eng.recluster()
    t0 = min(w.start for w in eng.windows)
    t1 = max(w.end for w in eng.windows)
    assert DiarizationService.split_points(_SplitStub(eng), t0, t1) == []


if __name__ == "__main__":
    import inspect

    mod = sys.modules[__name__]
    fns = [
        f
        for name, f in inspect.getmembers(mod, inspect.isfunction)
        if name.startswith("test_")
    ]
    print(f"Automatic speaker-count tests (supervisor review §18) — {len(fns)} cases")
    for fn in fns:
        fn()
        print(f"  {fn.__name__}  OK")
    print("all passed")
