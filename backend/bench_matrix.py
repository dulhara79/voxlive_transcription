#!/usr/bin/env python3
"""
bench_matrix.py — is diarization good enough ACROSS SCENARIOS, not on one file?

WHY THIS EXISTS
---------------
`bench_diarization.py` answers "does this box keep up?".
`eval_diarization.py` answers "is the labelling correct on THIS recording?".

Neither answers the question a review of this branch raised: the evaluation is
narrow. `tests/test_vs_v7.py` is a synthetic two-speaker regression, and the
real requirement is three, four and five speakers, short turns, cross-talk,
similar voices, drift over a long recording, Sinhala, and music that contains
no speakers at all. A system tuned until one interview "looks correct" is not
measured; it is anecdotal.

This script runs the whole scenario matrix in one pass and prints a table.
The matrix is data, not code — see benchmarks/matrix.json.

    python bench_matrix.py                                  # default manifest
    python bench_matrix.py --manifest benchmarks/matrix.json --backend both
    python bench_matrix.py --only news,panel --backend sortformer
    python bench_matrix.py --json results/2026-08-18.json

Scenarios whose audio or annotation is not on disk are reported as MISSING
rather than skipped silently, so the table doubles as a checklist of what still
needs to be recorded and annotated. An empty matrix is itself a finding.

EVERY SCENARIO RUNS IN AUTO MODE BY DEFAULT, and that is deliberate. Handing
the engine the true speaker count and then reporting that it found the true
speaker count measures nothing. Set "speaker_mode": "fixed" on a scenario only
when the point of that row is to test fixed mode.

Read the numbers as: DER is the headline, K is whether speaker DISCOVERY
worked, and the two fail independently. A row can have the right K and a
terrible DER (identities discovered, then assigned to the wrong audio), and a
row can have a low DER and the wrong K (one speaker missed entirely while
everyone else is labelled perfectly).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.diarization.scoring import (  # noqa: E402
    DEFAULT_COLLAR_SEC,
    score_timelines,
    timeline_to_turns,
)
from eval_diarization import SR, load_truth, read_wav_bytes, run_backend  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "benchmarks" / "matrix.json"


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def resolve(base: Path, value: str | None) -> Path | None:
    if not value:
        return None
    p = Path(value)
    return p if p.is_absolute() else (base / p)


def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"{path}: no manifest.\n"
            "  Copy benchmarks/matrix.json and point the wav/truth fields at "
            "your own recordings."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    scenarios = data["scenarios"] if isinstance(data, dict) else data
    for i, s in enumerate(scenarios):
        if "id" not in s:
            raise SystemExit(f"{path}: scenario {i} has no 'id'.")
    return scenarios


async def run_one(scenario: dict, backend: str, base: Path, collar: float) -> dict:
    """Score one scenario against one backend. Never raises for a bad row —
    a broken scenario must not take the rest of the matrix down with it."""
    row = {
        "id": scenario["id"],
        "backend": backend,
        "true_speakers": scenario.get("true_speakers"),
        "purpose": scenario.get("purpose", ""),
        "status": "ok",
    }

    wav = resolve(base, scenario.get("wav"))
    truth = resolve(base, scenario.get("truth"))

    missing = [str(p) for p in (wav, truth) if p is not None and not p.exists()] + (
        [] if wav else ["<no wav in manifest>"]
    )
    if missing:
        row["status"] = "missing"
        row["detail"] = ", ".join(missing)
        return row

    try:
        pcm = read_wav_bytes(str(wav))
    except SystemExit as e:  # read_wav_bytes raises this on a format mismatch
        row["status"] = "error"
        row["detail"] = str(e).splitlines()[0]
        return row

    duration = wav_duration(wav)

    # Per-scenario mode. Auto unless the row exists specifically to test fixed.
    from app.config import settings

    previous_mode = settings.speaker_mode
    settings.speaker_mode = scenario.get("speaker_mode", "auto")
    hint = int(scenario.get("speakers_hint", 0))

    try:
        result = await run_backend(pcm, backend, hint, duration)
    except Exception as e:  # a backend that will not load is a result, not a crash
        row["status"] = "error"
        row["detail"] = f"{type(e).__name__}: {e}"
        return row
    finally:
        settings.speaker_mode = previous_mode

    row["backend"] = result["backend"]
    row["rtf"] = result["rtf"]
    row["found_speakers"] = len({s for _, _, s in result["timeline"]})

    if truth:
        score = score_timelines(
            load_truth(str(truth)),
            timeline_to_turns(result["timeline"]),
            duration=duration,
            collar=collar,
        )
        row.update(
            der=score.der,
            confusion=(
                score.confusion / score.scored_speech if score.scored_speech else 0.0
            ),
            miss=score.miss / score.scored_speech if score.scored_speech else 0.0,
            false_alarm=(
                score.false_alarm / score.scored_speech if score.scored_speech else 0.0
            ),
            reference_speakers=score.reference_speakers,
            found_speakers=score.hypothesis_speakers,
        )
    else:
        # A speech-free row (the music scenario) has nothing to align against.
        # The only thing being asserted is that no speaker was invented.
        row["status"] = "counted"
    return row


def pct(x) -> str:
    return "-" if x is None else f"{x * 100:5.1f}%"


def print_table(rows: list[dict]) -> None:
    print()
    print(
        f"{'scenario':<20}{'backend':<12}{'true K':>7}{'K':>5}"
        f"{'DER':>9}{'conf':>8}{'miss':>8}{'FA':>8}{'RTF':>7}"
    )
    print("-" * 84)
    for r in rows:
        if r["status"] in ("missing", "error"):
            print(
                f"{r['id']:<20}{r['backend']:<12}{r['status'].upper():>7}  {r.get('detail', '')[:44]}"
            )
            continue
        true_k = r.get("true_speakers")
        print(
            f"{r['id']:<20}{r['backend']:<12}"
            f"{(true_k if true_k is not None else '-'):>7}"
            f"{r.get('found_speakers', '-'):>5}"
            f"{pct(r.get('der')):>9}{pct(r.get('confusion')):>8}"
            f"{pct(r.get('miss')):>8}{pct(r.get('false_alarm')):>8}"
            f"{r.get('rtf', 0.0):>7.2f}"
        )

    scored = [r for r in rows if r.get("der") is not None]
    wrong_k = [
        r
        for r in rows
        if r.get("true_speakers") is not None
        and r.get("found_speakers") is not None
        and r["true_speakers"] != r["found_speakers"]
    ]
    missing = [r for r in rows if r["status"] == "missing"]

    print()
    if scored:
        print(
            f"mean DER over {len(scored)} scored row(s): {pct(sum(r['der'] for r in scored) / len(scored))}"
        )
    if wrong_k:
        print(f"WRONG SPEAKER COUNT: {', '.join(sorted({r['id'] for r in wrong_k}))}")
    if missing:
        print(
            f"not yet recorded/annotated: "
            f"{', '.join(sorted({r['id'] for r in missing}))}"
        )
        print("  Until these exist, this matrix is not evidence about them.")


async def amain() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument(
        "--backend",
        default="both",
        choices=["embedding", "sortformer", "both"],
        help="'both' runs each scenario twice — the A/B the review asked for.",
    )
    ap.add_argument("--only", help="comma-separated scenario ids")
    ap.add_argument("--collar", type=float, default=DEFAULT_COLLAR_SEC)
    ap.add_argument("--json", help="write the raw rows here")
    ap.add_argument(
        "--max-der",
        type=float,
        default=None,
        help="exit non-zero if any scored row exceeds this (for CI)",
    )
    args = ap.parse_args()

    manifest_path = Path(args.manifest).resolve()
    scenarios = load_manifest(manifest_path)
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        scenarios = [s for s in scenarios if s["id"] in wanted]
        if not scenarios:
            raise SystemExit(f"--only {args.only}: no such scenario in {manifest_path}")

    backends = ["embedding", "sortformer"] if args.backend == "both" else [args.backend]

    rows: list[dict] = []
    for scenario in scenarios:
        for backend in backends:
            print(f"→ {scenario['id']} / {backend}", flush=True)
            rows.append(
                await run_one(scenario, backend, manifest_path.parent, args.collar)
            )

    print_table(rows)

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")

    if args.max_der is not None:
        over = [r for r in rows if (r.get("der") or 0.0) > args.max_der]
        if over:
            print(f"\nFAIL: {len(over)} row(s) over --max-der {args.max_der}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
