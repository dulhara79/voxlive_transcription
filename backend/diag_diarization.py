#!/usr/bin/env python3
"""
diag_diarization.py — is it the MODEL or is it our STITCHING?

READ THIS BEFORE CHANGING ANY DIARIZATION SETTING
-------------------------------------------------
This is the experiment the supervisor review asks for in point 16, and it is
meant to be run BEFORE the next code change, not after it.

The live Sortformer path is two things multiplied together:

    a model, run repeatedly on a rolling window
    a stitching layer, gluing each window's local speaker slots onto the
    session timeline

When the transcript names the wrong person, those two are indistinguishable
from the outside — which is precisely why raising SORTFORMER_WINDOW_SEC from
20 to 90 and hoping is not an experiment. So this script pulls them apart by
running the SAME audio two ways:

    RAW      one Sortformer call over the whole file.
             No window. No stitching. No cap fallback.
             This is the model's honest opinion.

    ROLLING  the real production path — `build_diarizer()`, fed in chunks,
             `finalize()` — exactly as `session/state.py` drives it.

Then, with a hand annotation, the verdict falls out of two numbers:

    RAW bad,  ROLLING bad   -> the MODEL (or the audio) is the problem.
                               Tuning the window will not save you; look at
                               point 13's alternatives instead.
    RAW good, ROLLING bad   -> the STITCHING is the problem. The model already
                               knew the answer and our layer lost it. Fix
                               `stitching.py`; changing models will not help.
    RAW good, ROLLING good  -> diarization is not your bug. Look at the ASR
                               alignment in `TranscriptStore.label_for()`.

Without `--truth` it still runs, and `reconcile.py` will tell you how far the
two disagree and exactly where — but it CANNOT tell you which one is right.
Two systems can be wrong together. Ten minutes annotating in Audacity buys you
the verdict; nothing else does.

USAGE
-----
    python diag_diarization.py news.wav --truth news.txt
    python diag_diarization.py news.wav --truth news.txt --also sortformer_offline
    python diag_diarization.py news.wav --truth news.txt --also pyannote_community1
    python diag_diarization.py news.wav --out diag/                 # no truth

`--also` runs the whole-file candidates from `offline.py`. Point 13 puts
pyannote Community-1 first in the queue for the final slot; this is where you
find out whether that holds on YOUR recordings rather than on a benchmark.

WHAT IT WRITES  (review point 17)
---------------------------------
    raw_sortformer.json     the model's per-pass output, verbatim
    stitched_timeline.json  every renaming decision, and the timeline after
    summary.json            every candidate's timeline, DER and agreement

MAKING THE TRUTH FILE
---------------------
Audacity: play the recording, Ctrl+B at each speaker change, type a name,
File > Export > Export Labels. Same format `eval_diarization.py` reads.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.diarization import offline as offline_backends  # noqa: E402
from app.diarization.reconcile import reconcile  # noqa: E402
from app.diarization.scoring import (  # noqa: E402
    DEFAULT_COLLAR_SEC,
    score_timelines,
    timeline_to_turns,
)
from eval_diarization import SR, load_truth, read_wav_bytes  # noqa: E402

# --------------------------------------------------------------------- RAW


def run_raw_whole_file(pcm: bytes, max_speakers: int) -> list[tuple[float, float, int]]:
    """One Sortformer call over the entire recording, with the LIVE checkpoint.

    Deliberately reuses `SortformerDiarizer._infer` rather than reimplementing
    the model call. Same checkpoint, same streaming config, same output
    parsing as production — the ONLY difference is that it gets the whole file
    instead of a window, and nothing stitches the result. If this were a
    parallel implementation, a difference between RAW and ROLLING would tell
    you about this script rather than about the system.
    """
    import numpy as np

    from app.diarization.sortformer import SortformerDiarizer

    diarizer = SortformerDiarizer(sample_rate=SR, max_speakers=max_speakers)
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    segs = diarizer._infer(audio)
    if segs is None:
        raise RuntimeError(
            "Sortformer inference returned nothing. Check the NeMo install and "
            "that a GPU is visible — on CPU this call can take many minutes."
        )
    return sorted((float(a), float(b), int(s)) for a, b, s in segs)


# ----------------------------------------------------------------- ROLLING


async def run_rolling(pcm: bytes, max_speakers: int, out_dir: Path) -> dict:
    """The production path, unmodified, with diagnostics turned on."""
    import os

    os.environ["DIARIZATION_BACKEND"] = "sortformer"

    from app.config import settings
    from app.diarization.factory import build_diarizer

    previous = (
        settings.diarization_backend,
        settings.diarization_enabled,
        getattr(settings, "diarize_diagnostics_dir", ""),
    )
    settings.diarization_backend = "sortformer"
    if not settings.diarization_enabled:
        print("  (DIARIZATION_MODE=off in your .env — enabling it for this run)")
        settings.diarization_enabled = True
    settings.diarize_diagnostics_dir = str(out_dir)

    try:
        diarizer = build_diarizer(settings, expected_speakers=0)
        if type(diarizer).__name__ != "SortformerDiarizer":
            raise RuntimeError(
                "the factory did not build the Sortformer backend — NeMo or CUDA "
                "is missing, so there is nothing here to diagnose. See the "
                "warning above."
            )
        diarizer.start()

        t0 = time.perf_counter()
        step = int(SR * 1.0) * 2  # 1 s chunks, as a WebSocket delivers them
        for i in range(0, len(pcm), step):
            diarizer.feed(pcm[i : i + step])
            await asyncio.sleep(0)

        rolling_before_final = diarizer.timeline()
        await diarizer.finalize()
        elapsed = time.perf_counter() - t0

        result = {
            "final": diarizer.timeline(),
            # `finalize()` keeps this when the offline pass overwrites the
            # timeline; if it did not run, the live timeline IS the final one.
            "rolling": diarizer.rolling_timeline() or rolling_before_final,
            "stats": diarizer.stats(),
            "seconds": elapsed,
        }
        await diarizer.aclose()
        return result
    finally:
        (
            settings.diarization_backend,
            settings.diarization_enabled,
            settings.diarize_diagnostics_dir,
        ) = previous


# --------------------------------------------------------------- reporting


def summarise(name: str, timeline, truth, duration: float, collar: float) -> dict:
    row = {
        "name": name,
        "runs": len(timeline),
        "speakers": len({s for _, _, s in timeline}),
        "timeline": [[round(a, 3), round(b, 3), s] for a, b, s in timeline],
    }
    if truth:
        score = score_timelines(
            truth, timeline_to_turns(timeline), duration=duration, collar=collar
        )
        row.update(
            der=score.der,
            confusion=score._share(score.confusion),
            miss=score._share(score.miss),
            false_alarm=score._share(score.false_alarm),
            reference_speakers=score.reference_speakers,
        )
    return row


def print_scores(rows: list[dict], scored: bool) -> None:
    print()
    if scored:
        print(
            f"{'candidate':<24}{'K':>4}{'runs':>7}{'DER':>9}{'conf':>8}{'miss':>8}{'FA':>8}"
        )
        print("-" * 68)
        for r in rows:
            print(
                f"{r['name']:<24}{r['speakers']:>4}{r['runs']:>7}"
                f"{r['der']:>8.1%}{r['confusion']:>8.1%}"
                f"{r['miss']:>8.1%}{r['false_alarm']:>8.1%}"
            )
    else:
        print(f"{'candidate':<24}{'K':>4}{'runs':>7}")
        print("-" * 36)
        for r in rows:
            print(f"{r['name']:<24}{r['speakers']:>4}{r['runs']:>7}")
        print("\n(no --truth, so none of these is scored — see the note below)")


def print_verdict(
    rows: list[dict], truth_speakers, threshold: float, final_name
) -> None:
    """Say plainly which layer to work on. This is the whole point of the run."""
    by_name = {r["name"]: r for r in rows}
    raw = by_name.get("raw_whole_file")
    # Compare against what the user ACTUALLY GETS, not the mid-session
    # timeline. `rolling_live` trails by up to one pass interval by design, and
    # scoring that against full-length truth charges the system for a lag that
    # never reaches the transcript.
    shipped = by_name.get(final_name) if final_name else None

    print("\n" + "=" * 68)
    print("VERDICT")
    print("=" * 68)

    if not (raw and shipped) or "der" not in raw or "der" not in shipped:
        print("Not enough to attribute the error.")
        print("  Needs BOTH the raw whole-file run and the production run to")
        print("  have scored, which needs --truth. Annotate the file and run")
        print("  this again; without it you can see the two disagree but not")
        print("  who is wrong.")
        return

    raw_ok = raw["der"] <= threshold
    shipped_ok = shipped["der"] <= threshold
    gap = shipped["der"] - raw["der"]

    print(f"  raw whole-file DER        {raw['der']:.1%}")
    print(f"  shipped ({shipped['name']}) DER   {shipped['der']:.1%}")
    print(f"  cost of the production path       {gap:+.1%}")
    print()

    if raw_ok and shipped_ok:
        print("  DIARIZATION IS NOT YOUR BUG.")
        print("  Both paths are within threshold. If the transcript still names")
        print("  the wrong person, the fault is downstream: look at")
        print("  TranscriptStore.label_for() and the ASR segment boundaries,")
        print("  not at the diarizer.")
    elif raw_ok and not shipped_ok:
        print("  THE STITCHING IS THE PROBLEM.")
        print("  The model already knew the answer and our layer lost it")
        print("  between windows. Work in app/diarization/stitching.py:")
        print("  SpeakerAlignment (wrong match) or SessionIdentityManager")
        print("  (cap fallback folding two people into one). Check `cap_folds`")
        print("  in stitched_timeline.json first.")
        print("  Changing MODELS will not help. Changing SORTFORMER_WINDOW_SEC")
        print("  will not help either.")
    elif not raw_ok and not shipped_ok:
        print("  THE MODEL (OR THE AUDIO) IS THE PROBLEM.")
        print("  Sortformer is wrong even when handed the whole file with")
        print("  nothing in the way, so no amount of stitching work recovers")
        print("  this. Check the obvious first — is the audio 16 kHz mono, is")
        print("  the speech Sinhala/Tamil (this checkpoint was trained")
        print("  primarily on English), are there more than 4 speakers (a hard")
        print("  ceiling for this model) — then try the alternatives with")
        print("  --also, review point 13.")
    else:
        print("  ODD RESULT: the production path beat the whole-file one.")
        print("  That can happen legitimately — a long file can exceed the")
        print("  model's comfortable context while a window does not — but")
        print("  check the audio length and GPU memory before believing it.")

    # Both can pass the threshold while the production path still costs real
    # accuracy. Saying nothing there would hide a fixable regression behind a
    # threshold that is itself a judgement call.
    if raw_ok and shipped_ok and gap >= 0.05:
        print()
        print(f"  BUT the production path is still costing {gap:.1%} against the")
        print("  model's own opinion. Both are under your threshold, so this is")
        print("  not urgent — it is a real gap in the stitching all the same.")

    if truth_speakers is not None:
        for r in (raw, shipped):
            if r["speakers"] != truth_speakers:
                print(
                    f"\n  SPEAKER COUNT WRONG in {r['name']}: found "
                    f"{r['speakers']}, truth has {truth_speakers}."
                )
                if truth_speakers > 4:
                    print(
                        "  Note: this checkpoint caps at 4 speakers. That is "
                        "architecture,\n  not tuning — no setting in this repo "
                        "raises it."
                    )


# -------------------------------------------------------------------- main


async def amain() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("wav", help="mono 16 kHz WAV — one saved recording")
    ap.add_argument(
        "--truth", help="Audacity labels or RTTM. Without it there is no verdict."
    )
    ap.add_argument(
        "--out", default="diag", help="where the JSON goes (default: diag/)"
    )
    ap.add_argument(
        "--also",
        default="",
        help="comma-separated whole-file candidates: "
        + ", ".join(offline_backends.available()),
    )
    ap.add_argument("--max-speakers", type=int, default=4)
    ap.add_argument("--collar", type=float, default=DEFAULT_COLLAR_SEC)
    ap.add_argument(
        "--der-threshold",
        type=float,
        default=0.20,
        help="DER at or below which a path counts as 'good' in the verdict "
        "(default 0.20). This is a judgement call about YOUR use case, not a "
        "standard; set it where a listener would stop noticing.",
    )
    ap.add_argument("--skip-raw", action="store_true", help="rolling path only")
    args = ap.parse_args()

    pcm = read_wav_bytes(args.wav)
    duration = len(pcm) / 2 / SR
    truth = load_truth(args.truth) if args.truth else None
    truth_speakers = len({t.speaker for t in truth}) if truth else None

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("VoxLive diarization diagnostic — model vs stitching")
    print("=" * 68)
    print(f"audio    {args.wav}  ({duration:.1f}s)")
    print(f"truth    {args.truth or '(none — no verdict will be possible)'}")
    if truth_speakers is not None:
        print(f"         {truth_speakers} speaker(s) annotated")
    print(f"output   {out_dir}/")

    rows: list[dict] = []
    timelines: dict[str, list] = {}

    # ---- RAW ---------------------------------------------------------------
    if not args.skip_raw:
        print("\n→ raw whole-file Sortformer (no window, no stitching)")
        try:
            t0 = time.perf_counter()
            raw_timeline = await asyncio.to_thread(
                run_raw_whole_file, pcm, args.max_speakers
            )
            print(f"  {len(raw_timeline)} run(s) in {time.perf_counter() - t0:.1f}s")
            timelines["raw_whole_file"] = raw_timeline
            rows.append(
                summarise("raw_whole_file", raw_timeline, truth, duration, args.collar)
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            print("  Without this there is no attribution — fix it before reading on.")

    # ---- ROLLING -----------------------------------------------------------
    print("\n→ rolling live path (the production code, with diagnostics on)")
    final_name = None
    try:
        rolling = await run_rolling(pcm, args.max_speakers, out_dir)
        print(f"  {len(rolling['final'])} run(s) in {rolling['seconds']:.1f}s")
        print(f"  stats  {rolling['stats']}")

        # The timeline as it stood at `stop`, before finalize. This one
        # legitimately LAGS: the background pass runs every DIARIZE_INTERVAL_SEC,
        # so the last second or two of audio is usually unlabelled and shows up
        # as `miss`. Do not read that as a diarization failure — it is why the
        # final pass exists. It is here because a large gap between this and the
        # final row is what tells you the final pass is earning its keep.
        timelines["rolling_live"] = rolling["rolling"]
        rows.append(
            summarise("rolling_live", rolling["rolling"], truth, duration, args.collar)
        )

        # And what the user actually gets. Named for what finalize DID, because
        # "after_final_pass" is not the same claim when the offline checkpoint
        # failed to load and it quietly ran the rolling window again.
        mode = str(rolling["stats"].get("final_pass", "unknown"))
        final_name = "final_offline" if mode == "offline" else "final_rolling"
        timelines[final_name] = rolling["final"]
        rows.append(
            summarise(final_name, rolling["final"], truth, duration, args.collar)
        )
        if mode != "offline":
            print(f"  NOTE: the offline final pass did not run — {mode}")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: {type(exc).__name__}: {exc}")

    # ---- ALSO --------------------------------------------------------------
    for name in [n.strip() for n in args.also.split(",") if n.strip()]:
        print(f"\n→ {name} (whole file)")
        try:
            from app.config import settings

            t0 = time.perf_counter()
            timeline = await asyncio.to_thread(
                offline_backends.run,
                name,
                pcm,
                sample_rate=SR,
                max_speakers=args.max_speakers,
                hf_token=getattr(settings, "huggingface_token", ""),
            )
            print(f"  {len(timeline)} run(s) in {time.perf_counter() - t0:.1f}s")
            timelines[name] = timeline
            rows.append(summarise(name, timeline, truth, duration, args.collar))
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            print("  (nothing in offline.py has ever been run against a real")
            print("   checkpoint — suspect this file before the model)")

    if not rows:
        print("\nEverything failed. Nothing to report.")
        return 1

    print_scores(rows, scored=truth is not None)

    # ---- pairwise agreement ------------------------------------------------
    if len(timelines) > 1:
        print("\n" + "=" * 68)
        print("AGREEMENT  (review point 10: deterministic, not 'the newer one wins')")
        print("=" * 68)
        names = list(timelines)
        base = names[0]
        for other in names[1:]:
            print()
            print(reconcile(timelines[base], timelines[other]).report(base, other))

    print_verdict(rows, truth_speakers, args.der_threshold, final_name)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "audio": str(args.wav),
                "duration_sec": round(duration, 2),
                "truth": str(args.truth) if args.truth else None,
                "truth_speakers": truth_speakers,
                "collar_sec": args.collar,
                "candidates": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nwrote {summary_path}")
    for f in ("raw_sortformer.json", "stitched_timeline.json"):
        if (out_dir / f).exists():
            print(f"wrote {out_dir / f}")
    if truth is None:
        print(
            "\nNo --truth given. You can see WHERE the candidates disagree above,\n"
            "but not which is right. Annotate the disagreement regions listed\n"
            "under AGREEMENT — that is ten minutes of work aimed at exactly the\n"
            "audio that matters — and run this again."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
