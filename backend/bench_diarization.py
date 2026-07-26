#!/usr/bin/env python3
"""
bench_diarization.py — does YOUR box keep up?

Unlike the tests in tests/, this one loads the real model. Run it on the
machine you will deploy on before tuning anything, because every latency knob
in .env is a trade against one number: how long a diarization pass takes.

    python bench_diarization.py                    # synthetic audio
    python bench_diarization.py meeting.wav        # your own recording
    python bench_diarization.py meeting.wav --speakers 2

With a WAV file it also prints the full speaker timeline, which is the fastest
way to see whether diarization is working on YOUR audio — Sinhala and Tamil
included — without running the whole server.
"""

import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import settings  # noqa: E402
from app.embedder import Embedder, warmup  # noqa: E402
from app.speaker_engine import (  # noqa: E402
    SpeakerEngine,
    Window,
    slice_windows,
    speech_regions,
)

SR = 16000


def read_wav(path: str):
    with wave.open(path, "rb") as w:
        if w.getnchannels() != 1 or w.getframerate() != SR:
            raise SystemExit(
                f"{path}: need mono 16 kHz, got {w.getnchannels()}ch "
                f"@ {w.getframerate()} Hz.\n"
                f"  ffmpeg -i {path} -ac 1 -ar 16000 -c:a pcm_s16le out.wav"
            )
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def synth(seconds=30.0):
    """Two alternating pseudo-voices. Not speech — this benchmarks THROUGHPUT,
    not accuracy. Use a real WAV for accuracy."""
    rng = np.random.default_rng(0)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    out = np.zeros(n, dtype=np.float32)
    for i, (a, b) in enumerate(
        zip(np.arange(0, seconds, 3.0), np.arange(3.0, seconds + 3.0, 3.0))
    ):
        f0 = 110 if i % 2 == 0 else 190
        i0, i1 = int(a * SR), min(n, int(b * SR))
        seg = t[i0:i1]
        v = sum(np.sin(2 * np.pi * f0 * k * seg) / k for k in range(1, 6))
        env = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * seg)
        out[i0:i1] = 0.25 * v * env + 0.01 * rng.normal(size=i1 - i0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", nargs="?", help="mono 16 kHz WAV (optional)")
    ap.add_argument(
        "--speakers",
        type=int,
        default=None,
        help="known speaker count (a ceiling, not a quota)",
    )
    args = ap.parse_args()

    print("=" * 62)
    print("VoxLive diarization benchmark")
    print("=" * 62)

    try:
        import torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"torch {torch.__version__}   device: {dev}")
        if dev == "cpu":
            print(f"threads: {torch.get_num_threads()}")
    except ImportError:
        raise SystemExit("torch is not installed — pip install -r requirements.txt")

    t0 = time.perf_counter()
    warmup(settings.huggingface_token)
    print(f"model load + warmup: {time.perf_counter() - t0:.1f}s\n")

    emb = Embedder(settings.huggingface_token, sample_rate=SR)

    # ---- 1. batching -------------------------------------------------------
    # The v7 embedder looped one window at a time. This is the number that
    # made that a mistake.
    print("--- batching ---")
    print(f"{'windows':>8} {'total ms':>10} {'per window':>12}")
    per_window = {}
    for k in (1, 2, 4, 8, 16, 32):
        w = [np.random.randn(int(1.5 * SR)).astype(np.float32) * 0.1] * k
        emb.embed_batch(w[:1])  # settle
        t = time.perf_counter()
        emb.embed_batch(w)
        ms = (time.perf_counter() - t) * 1000
        per_window[k] = ms / k
        print(f"{k:>8} {ms:>9.0f}  {ms / k:>10.1f} ms")
    if per_window.get(1) and per_window.get(16):
        print(
            f"\nbatching speedup at 16 windows: {per_window[1] / per_window[16]:.1f}x"
        )

    # ---- 2. can a pass keep up? -------------------------------------------
    interval = settings.diarize_interval_sec
    n_win = max(1, int(interval / 0.75))  # HOP_SEC = 0.75
    pass_ms = per_window.get(min(per_window, key=lambda k: abs(k - n_win)), 0) * n_win
    print(f"\n--- pass budget (DIARIZE_INTERVAL_SEC={interval}) ---")
    print(f"~{n_win} window(s) per pass  ->  ~{pass_ms:.0f} ms of model time")
    budget = interval * 1000
    print(f"budget: {budget:.0f} ms   headroom: {budget / max(pass_ms, 1):.1f}x")
    if pass_ms > budget * 0.5:
        print("  ⚠ under 2x headroom. Raise DIARIZE_INTERVAL_SEC or use a GPU.")
    else:
        print("  ✓ comfortable")

    # ---- 3. real audio -----------------------------------------------------
    if args.wav:
        pcm = read_wav(args.wav)
        label = args.wav
    else:
        pcm = synth(30.0)
        label = "synthetic 30 s"
        print(
            "\n(no WAV given — using synthetic tones. Pass a real recording "
            "to check ACCURACY; this section only checks throughput.)"
        )

    dur = len(pcm) / SR
    print(f"\n--- full pipeline on {label} ({dur:.1f}s) ---")

    t = time.perf_counter()
    regions = speech_regions(pcm, SR, 0.0, settings.vad_aggressiveness)
    vad_ms = (time.perf_counter() - t) * 1000
    speech = sum(b - a for a, b in regions)
    print(
        f"VAD: {len(regions)} region(s), {speech:.1f}s speech "
        f"({speech / dur:.0%}) in {vad_ms:.0f} ms"
    )

    t = time.perf_counter()
    waves, spans = slice_windows(pcm, SR, regions, 0.0)
    vecs = emb.embed_batch(waves)
    embed_ms = (time.perf_counter() - t) * 1000
    good = [(s, v) for s, v in zip(spans, vecs) if v is not None]
    print(
        f"embed: {len(good)}/{len(waves)} window(s) in {embed_ms:.0f} ms "
        f"({embed_ms / max(1, len(waves)):.1f} ms each)"
    )

    k = args.speakers if args.speakers is not None else settings.expected_speakers
    engine = SpeakerEngine(expected_speakers=k, max_speakers=settings.max_speakers)
    engine.add_windows([Window(a, b, v) for (a, b), v in good])
    t = time.perf_counter()
    engine.recluster()
    cluster_ms = (time.perf_counter() - t) * 1000
    print(f"cluster: {cluster_ms:.0f} ms  ->  {engine.speaker_count()} speaker(s)")

    total = vad_ms + embed_ms + cluster_ms
    print(f"\ntotal {total:.0f} ms for {dur:.1f}s audio   RTF {total / 1000 / dur:.3f}")
    print("(RTF is per-pass cost amortised; live, only NEW audio is embedded.)")

    tl = engine.timeline()
    if tl:
        print(f"\n--- timeline ({len(tl)} run(s)) ---")
        for a, b, s in tl[:40]:
            print(f"  {a:7.2f} - {b:7.2f}s   Speaker {s + 1}")
        if len(tl) > 40:
            print(f"  ... {len(tl) - 40} more")

    print("\n" + "=" * 62)
    if args.wav:
        print("Check the timeline against what you HEAR in the file. If turns")
        print("are right but names swap, that is clustering — set --speakers.")
        print("If turn BOUNDARIES are wrong, look at VAD_AGGRESSIVENESS.")
    else:
        print("Now run it again with a real recording:")
        print("  python bench_diarization.py your_meeting.wav --speakers 2")
    print("=" * 62)


if __name__ == "__main__":
    main()
