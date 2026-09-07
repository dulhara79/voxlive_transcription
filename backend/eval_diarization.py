#!/usr/bin/env python3
"""
eval_diarization.py — how good is diarization on YOUR audio?

WHY THIS IS NOT bench_diarization.py
------------------------------------
`bench_diarization.py` answers "does this box keep up?" — batching speedups,
pass budget, real-time factor. It prints a timeline, but it has nothing to
compare that timeline against, so it can tell you the system is fast and
cannot tell you it is right.

This script answers the other question: "is the speaker labelling CORRECT?",
by scoring the timeline against one you annotated by hand.

It also evaluates diarization ON ITS OWN, with ASR nowhere in the process.
That separation is the point. When you judge "Speaker 2 said the wrong thing"
from the transcript, you are looking at two systems multiplied together and
cannot tell which one failed. Here there is no text to be wrong.

    python eval_diarization.py meeting.wav --truth meeting.txt --speakers 2
    python eval_diarization.py meeting.wav --truth meeting.txt --backend both
    python eval_diarization.py --compare truth.txt system.txt

MAKING THE TRUTH FILE
---------------------
Audacity: play the recording, press Ctrl+B at each speaker change, type a
name, then File > Export > Export Labels. That writes exactly the format this
reads:

    0.000000    4.120000    Nimal
    4.120000    7.480000    Kamala
    7.480000   10.900000    Nimal

Whitespace, tab or comma separated; `#` starts a comment. RTTM is also
accepted. Ten minutes of careful annotation is worth more than a week of
tuning against an impression.

The names are arbitrary — they are matched to the system's numbering
automatically, so a run that got every turn right but swapped Speaker 1 and
Speaker 2 scores as perfect, because it is.

READING THE RESULT
------------------
    DER near 0        the partition is right
    high CONFUSION    turns detected, wrong identity  -> a clustering problem
    high MISS         speech the system never labelled -> VAD, or the diarizer
                      falling behind
    high FALSE ALARM  labels over silence/noise        -> VAD too permissive
    speaker MISMATCH  over- or under-clustering        -> set --speakers

ONE LIMITATION, STATED UP FRONT
-------------------------------
Overlapping speech is not scored as overlap; see `app/diarization/scoring.py`.
The embedding backend structurally cannot emit two speakers at one instant, so
this metric UNDERSTATES what overlap costs. Do not use it to conclude that
overlap is a solved problem here — for that, listen to the overlapped regions
and check which speaker was dropped.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.diarization.scoring import (  # noqa: E402
    DEFAULT_COLLAR_SEC,
    parse_annotation,
    score_timelines,
    timeline_to_turns,
)

SR = 16000
# Fed in chunks rather than one blob so the diarizer sees the same arrival
# pattern it sees live. Feeding a whole file at once would exercise a code
# path no real session ever takes.
FEED_CHUNK_SEC = 1.0


def read_wav_bytes(path: str) -> bytes:
    with wave.open(path, "rb") as w:
        if w.getnchannels() != 1 or w.getframerate() != SR:
            raise SystemExit(
                f"{path}: need mono 16 kHz, got {w.getnchannels()}ch "
                f"@ {w.getframerate()} Hz.\n"
                f"  ffmpeg -i {path} -ac 1 -ar 16000 -c:a pcm_s16le out.wav"
            )
        return w.readframes(w.getnframes())


def load_truth(path: str):
    turns = parse_annotation(Path(path).read_text(encoding="utf-8"))
    if not turns:
        raise SystemExit(
            f"{path}: no usable annotations found.\n"
            "  Expected lines of 'start end speaker', or NIST RTTM."
        )
    return turns


async def run_backend(pcm: bytes, backend: str, speakers: int, duration: float):
    """Diarize a whole file through the REAL session-time code path.

    Deliberately not a reimplementation: this builds the same object
    `SessionState` builds, feeds it the same way, and calls the same
    `finalize()` the WebSocket route calls on `stop`. A benchmark that scores
    a parallel implementation tells you about the benchmark.
    """
    import os

    # resolve_backend() reads this, and build_diarizer() is what the session
    # uses, so overriding here keeps the A/B honest.
    os.environ["DIARIZATION_BACKEND"] = backend

    from app.config import settings
    from app.diarization.factory import build_diarizer, resolve_backend, warmup_backend

    settings.diarization_backend = backend
    # An .env carrying DIARIZATION_MODE=off would otherwise produce an empty
    # timeline and a 100% miss rate, which reads as a catastrophic diarization
    # failure rather than as a setting. Running this script IS the request to
    # diarize.
    if not settings.diarization_enabled:
        print("  (DIARIZATION_MODE=off in your .env — enabling it for this run)")
        settings.diarization_enabled = True

    resolved = resolve_backend(backend)
    if resolved != backend:
        print(f"  ! requested {backend}, running {resolved} (see the warning above)")

    t0 = time.perf_counter()
    await warmup_backend(settings)
    warm_sec = time.perf_counter() - t0

    diarizer = build_diarizer(settings, speakers)
    diarizer.start()

    t0 = time.perf_counter()
    step = int(SR * FEED_CHUNK_SEC) * 2
    for i in range(0, len(pcm), step):
        diarizer.feed(pcm[i : i + step])
        # Let the background pass actually run, as it would between audio
        # frames arriving over a WebSocket.
        await asyncio.sleep(0)

    await diarizer.finalize()
    process_sec = time.perf_counter() - t0

    timeline = diarizer.timeline()
    stats = diarizer.stats() if hasattr(diarizer, "stats") else {}
    await diarizer.aclose()

    return {
        "backend": resolved,
        "timeline": timeline,
        "warm_sec": warm_sec,
        "process_sec": process_sec,
        "rtf": process_sec / duration if duration else 0.0,
        "stats": stats,
    }


def print_timeline(timeline, limit: int = 40) -> None:
    if not timeline:
        print("  (empty — the diarizer produced no speaker runs at all)")
        return
    print(f"  {len(timeline)} run(s):")
    for a, b, s in timeline[:limit]:
        print(f"    {a:7.2f} - {b:7.2f}s   Speaker {int(s) + 1}")
    if len(timeline) > limit:
        print(f"    ... {len(timeline) - limit} more")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score diarization against a hand-annotated timeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("wav", nargs="?", help="mono 16 kHz WAV")
    ap.add_argument("--truth", help="annotation file (Audacity labels or RTTM)")
    ap.add_argument(
        "--compare",
        nargs=2,
        metavar=("TRUTH", "SYSTEM"),
        help="score two annotation files against each other; no audio, no model",
    )
    ap.add_argument(
        "--speakers",
        type=int,
        default=None,
        help="known speaker count (a CEILING, not a quota). "
        "Set it when you know it: measuring identity and count at the same "
        "time tells you which one failed only by luck.",
    )
    ap.add_argument(
        "--backend",
        default=None,
        choices=("embedding", "sortformer", "auto", "both"),
        help="'both' runs the A/B on identical audio",
    )
    ap.add_argument(
        "--collar",
        type=float,
        default=DEFAULT_COLLAR_SEC,
        help=f"boundary tolerance in seconds (default {DEFAULT_COLLAR_SEC}); "
        "0 for the unforgiving number",
    )
    args = ap.parse_args()

    # ---- offline mode: score two files, no model, no audio -----------------
    if args.compare:
        truth = load_truth(args.compare[0])
        system = load_truth(args.compare[1])
        score = score_timelines(truth, system, collar=args.collar)
        print("=" * 62)
        print(f"{args.compare[1]}  vs  {args.compare[0]}")
        print("=" * 62)
        print(score.report())
        return

    if not args.wav:
        ap.error("give a WAV file, or use --compare TRUTH SYSTEM")

    pcm = read_wav_bytes(args.wav)
    duration = len(pcm) / 2 / SR
    truth = load_truth(args.truth) if args.truth else None

    from app.config import settings

    speakers = (
        args.speakers if args.speakers is not None else settings.expected_speakers
    )
    requested = args.backend or settings.diarization_backend

    print("=" * 62)
    print("VoxLive diarization evaluation")
    print("=" * 62)
    print(f"audio      {args.wav}  ({duration:.1f}s)")
    print(f"speakers   {speakers or 'auto'}")
    print(f"collar     {args.collar:.2f}s")
    if truth:
        ref_speakers = len({t.speaker for t in truth})
        ref_speech = sum(t.duration for t in truth)
        print(
            f"truth      {args.truth}  "
            f"({len(truth)} turn(s), {ref_speakers} speaker(s), "
            f"{ref_speech:.1f}s speech)"
        )
    else:
        print("truth      (none — timeline only; pass --truth to score it)")

    backends = ("embedding", "sortformer") if requested == "both" else (requested,)
    results = []

    for backend in backends:
        print("\n" + "-" * 62)
        print(f"backend: {backend}")
        print("-" * 62)
        try:
            result = asyncio.run(run_backend(pcm, backend, speakers, duration))
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {exc}")
            continue

        print(
            f"  warmup {result['warm_sec']:.1f}s   "
            f"diarize {result['process_sec']:.1f}s   "
            f"RTF {result['rtf']:.3f}"
        )
        if result["stats"]:
            print(f"  stats  {result['stats']}")
        print_timeline(result["timeline"])

        if truth:
            hyp = timeline_to_turns(result["timeline"])
            score = score_timelines(truth, hyp, duration=duration, collar=args.collar)
            print()
            print(score.report())
            results.append((result["backend"], score))

    # ---- A/B summary -------------------------------------------------------
    if len(results) > 1:
        print("\n" + "=" * 62)
        print("A/B")
        print("=" * 62)
        print(f"{'backend':<14}{'DER':>9}{'confusion':>12}{'miss':>9}{'speakers':>10}")
        for name, score in results:
            print(
                f"{name:<14}{score.der:>8.1%}"
                f"{score._share(score.confusion):>11.1%}"
                f"{score._share(score.miss):>8.1%}"
                f"{score.hypothesis_speakers:>10}"
            )
        best = min(results, key=lambda r: r[1].der)
        print(f"\nlower DER: {best[0]}")
        print(
            "One recording is not a decision. Run this over quiet speech, "
            "fast speech,\nthree speakers, overlapping speech, and "
            "Sinhala-English code-switching\nbefore choosing a production "
            "backend."
        )

    print("\n" + "=" * 62)
    if not truth:
        print("No --truth given, so nothing was scored. Annotate the file in")
        print("Audacity (Ctrl+B at each speaker change, then Export Labels)")
        print("and re-run to get a DER.")
    print("=" * 62)


if __name__ == "__main__":
    main()
