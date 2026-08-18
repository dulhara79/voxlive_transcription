#!/usr/bin/env python3
"""analyse_log.py — turn a VoxLive backend log into the numbers that matter.

    python tools/analyse_log.py podcast_test.log

WHY THIS EXISTS
---------------
Reading a diarization log by eye gives you the wrong answer confidently. The
three-speaker podcast log is the example: grepping it for `asr_completed`
shows every one of the 51 segments completing, which reads as "no speech was
lost" — and that conclusion is wrong, because the loss happened INSIDE a
segment. `_split` built a chunk per speaker turn and discarded any chunk that
came out too short or too quiet, and a discarded chunk is logged nowhere at
all. The segment still completes. The words are still gone.

So this tool reports what is actually checkable, and is explicit about what
the log cannot tell you:

  CEILING     the ceiling the session really ran with, which is the single
              thing that decided whether a third speaker was reachable
  SPEAKERS    the trajectory, not just the final count — a count that climbs
              and settles is healthy, one that oscillates is not
  SEPARATION  silhouette and closest-centroid distance over time. Silhouette
              sagging while closest-centroid RISES means the clusters are
              being stretched to cover someone who has no cluster of their own
  COVERAGE    segments created vs. ASR completions, per segment id. A segment
              id that never completes is speech that certainly vanished
  SPLITS      how many segments were cut, and into how many pieces
  FRAGMENTS   the segment-duration histogram. Sub-second segments are one
              Gemini call each, and they are what pushes you into the rate
              limiter
  LOSSES      every guard that fired, quoted with its reason

Nothing here is tuned or thresholded. It counts what happened.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict

# Log lines this understands. Anything unmatched is ignored, so a format change
# degrades to a smaller report rather than a traceback.
RE = {
    "session": re.compile(r"session_started \(expected_speakers=(?P<cap>\w+)"),
    "segment": re.compile(
        r"segment (?P<id>\d+): (?P<a>[\d.]+)-(?P<b>[\d.]+)s "
        r"\((?P<dur>[\d.]+)s, (?P<reason>\w+)\) rms=(?P<rms>\d+)"
    ),
    "split": re.compile(
        r"segment (?P<id>\d+) split into (?P<n>\d+) turn\(s\) at (?P<at>.+?) "
        r"\((?P<refused>\d+) cut\(s\) refused"
    ),
    "clusters": re.compile(
        r"clusters: (?P<win>\d+) trusted.*?-> (?P<k>\d+) speaker\(s\) "
        r"\(silhouette=(?P<sil>[\d.-]+), closest centroids=(?P<dmin>[\d.]+)\)"
    ),
    "timeline": re.compile(r"timeline: (?P<win>\d+) window\(s\) -> (?P<k>\d+) speaker"),
    "completed": re.compile(r"asr_completed\s+\[session_id=\S+ segment_id=(?P<id>\d+)"),
    "timeout": re.compile(r"ASR timeout after (?P<sec>\d+)s.*segment_id=(?P<id>\d+)"),
    "ratelimit": re.compile(r"ASR rate limited.*segment_id=(?P<id>\d+)"),
    "backlog": re.compile(
        r"segment backlog full \(\d+\) — dropped segment (?P<id>\d+)"
    ),
    "gated": re.compile(r"segment (?P<id>\d+) gated: rms=(?P<rms>\d+)"),
    "rateguard": re.compile(r"rate guard: (?P<detail>.+?), dropping"),
    "contentguard": re.compile(r"content guard: dropping (?P<detail>.+?)\s*\["),
    "langdrop": re.compile(r"dropped segment (?P<id>\d+): lang=(?P<lang>\w+)"),
    "repeat": re.compile(r"repeat guard: segment (?P<id>\d+) dropped"),
    "queuefull": re.compile(r"ASR capacity reached — segment (?P<id>\d+) dropped"),
    "done": re.compile(r"session done: (?P<stats>\{.+\})"),
}


class Report:
    def __init__(self) -> None:
        self.ceiling: str | None = None
        self.segments: dict[int, dict] = {}
        self.splits: dict[int, int] = {}
        self.refused = 0
        self.clusters: list[tuple[int, int, float, float]] = []
        self.timeline: list[tuple[int, int]] = []
        self.completed: Counter = Counter()
        self.events: dict[str, list[str]] = defaultdict(list)
        self.done: str | None = None

    def feed(self, line: str) -> None:
        for name, rx in RE.items():
            m = rx.search(line)
            if not m:
                continue
            self._apply(name, m)
            return

    def _apply(self, name: str, m: re.Match) -> None:
        g = m.groupdict()
        if name == "session":
            self.ceiling = g["cap"]
        elif name == "segment":
            self.segments[int(g["id"])] = {
                "dur": float(g["dur"]),
                "reason": g["reason"],
                "rms": int(g["rms"]),
            }
        elif name == "split":
            self.splits[int(g["id"])] = int(g["n"])
            self.refused += int(g["refused"])
        elif name == "clusters":
            self.clusters.append(
                (int(g["win"]), int(g["k"]), float(g["sil"]), float(g["dmin"]))
            )
        elif name == "timeline":
            self.timeline.append((int(g["win"]), int(g["k"])))
        elif name == "completed":
            self.completed[int(g["id"])] += 1
        elif name == "done":
            self.done = g["stats"]
        else:
            self.events[name].append(
                ", ".join(f"{k}={v}" for k, v in g.items() if v is not None)
            )


def bar(n: int, total: int, width: int = 28) -> str:
    return "█" * max(0, round(width * n / total)) if total else ""


def emit(r: Report) -> int:
    """Print the report. Returns a shell exit code: 1 if speech was lost."""
    out, problems = print, 0

    out("\n" + "=" * 66)
    out("  SPEAKER CEILING")
    out("=" * 66)
    if r.ceiling is None:
        out("  no session_started line found — is this the right log?")
        return 2
    out(f"  session ran with expected_speakers = {r.ceiling}")
    if r.ceiling.isdigit() and int(r.ceiling) > 0:
        out(
            f"  This is a HARD CEILING. The engine cannot return more than\n"
            f"  {r.ceiling} speaker(s) no matter what the audio contains."
        )
        problems += 1
    else:
        out("  Auto — the engine chose the count from the audio.")

    out("\n" + "=" * 66)
    out("  SPEAKER COUNT OVER TIME")
    out("=" * 66)
    if r.timeline:
        seq = [k for _, k in r.timeline]
        out(f"  final: {seq[-1]} speaker(s)   range: {min(seq)}-{max(seq)}")
        out("  trajectory: " + "".join(str(k) for k in seq[-60:]))
        flips = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
        out(f"  changes: {flips} (a count that settles is healthy)")
    else:
        out("  no timeline lines")

    if r.clusters:
        out("\n" + "=" * 66)
        out("  CLUSTER SEPARATION")
        out("=" * 66)
        first, last = r.clusters[0], r.clusters[-1]
        out(
            f"  first pass  K={first[1]}  silhouette={first[2]:.3f}  closest={first[3]:.3f}"
        )
        out(
            f"  last  pass  K={last[1]}  silhouette={last[2]:.3f}  closest={last[3]:.3f}"
        )
        sils = [c[2] for c in r.clusters]
        out(f"  silhouette min/max: {min(sils):.3f} / {max(sils):.3f}")
        if last[2] < first[2] * 0.7 and last[3] > first[3]:
            out(
                "  NOTE: silhouette fell while the closest-centroid distance rose.\n"
                "  Clusters are being stretched — usually a speaker with no\n"
                "  cluster of their own being absorbed into the others."
            )

    out("\n" + "=" * 66)
    out("  SEGMENT COVERAGE")
    out("=" * 66)
    created = set(r.segments)
    finished = set(r.completed)
    missing = sorted(created - finished)
    out(f"  segments created:   {len(created)}")
    out(f"  ASR completions:    {sum(r.completed.values())}")
    out(f"  segments finished:  {len(finished)}")
    if missing:
        out(f"  NEVER COMPLETED:    {missing}")
        out("  ^ this audio certainly did not reach the transcript.")
        problems += 1
    else:
        out("  every segment reached ASR at least once.")
        out(
            "  CAVEAT: this does NOT prove no speech was lost. Audio dropped\n"
            "  inside _split (a turn too short or too quiet to stand alone)\n"
            "  is not logged anywhere and the segment still completes."
        )

    if r.splits:
        out(f"\n  segments split at a speaker change: {len(r.splits)}")
        out(f"  pieces produced: {sum(r.splits.values())}")
        out(f"  cuts refused as too short: {r.refused}")
        out("  (a refused cut costs one speaker label, never any speech)")

    out("\n" + "=" * 66)
    out("  SEGMENT LENGTHS  (each one is a separate Gemini call)")
    out("=" * 66)
    buckets = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 7), (7, 99)]
    total = len(r.segments)
    for lo, hi in buckets:
        n = sum(1 for s in r.segments.values() if lo <= s["dur"] < hi)
        label = f"{lo}-{hi}s" if hi < 99 else f"{lo}s+"
        out(f"  {label:>7} {n:>4}  {bar(n, total)}")
    tiny = [i for i, s in r.segments.items() if s["dur"] < 1.0]
    if total:
        out(f"\n  under 1s: {len(tiny)} of {total} ({len(tiny) / total:.0%})")
        if len(tiny) / total > 0.15:
            out(
                "  High fragmentation. Each of these is a full API round trip,\n"
                "  and they are what pushes the session into the rate limiter.\n"
                "  Raising SILENCE_MS lengthens segments — MEASURE it, do not\n"
                "  guess: re-run this tool after each change."
            )
    by_reason = Counter(s["reason"] for s in r.segments.values())
    out("  cut reasons: " + ", ".join(f"{k}={v}" for k, v in by_reason.most_common()))

    out("\n" + "=" * 66)
    out("  GUARDS AND FAILURES")
    out("=" * 66)
    labels = {
        "queuefull": "ASR queue full — SPEECH LOST",
        "backlog": "segment backlog overflow — SPEECH LOST",
        "gated": "segment gated below MIN_SEGMENT_RMS — SPEECH LOST",
        "langdrop": "dropped: language not in ALLOWED_LANGUAGES — SPEECH LOST",
        "repeat": "repeat guard — SPEECH LOST if the repetition was real",
        "rateguard": "rate guard rejected ASR output — SPEECH LOST",
        "contentguard": "content guard rejected ASR output — SPEECH LOST",
        "timeout": "ASR timeout (retried; latency, not loss)",
        "ratelimit": "ASR rate limited (retried; latency, not loss)",
    }
    lossy = {
        "queuefull",
        "backlog",
        "gated",
        "langdrop",
        "repeat",
        "rateguard",
        "contentguard",
    }
    any_event = False
    for key, label in labels.items():
        hits = r.events.get(key, [])
        if not hits:
            continue
        any_event = True
        out(f"\n  {label}: {len(hits)}")
        for h in hits[:6]:
            out(f"      {h}")
        if len(hits) > 6:
            out(f"      ... and {len(hits) - 6} more")
        if key in lossy:
            problems += 1
    if not any_event:
        out("  none fired.")

    if r.done:
        out("\n" + "=" * 66)
        out("  FINAL")
        out("=" * 66)
        out(f"  {r.done}")

    out("")
    return 1 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("logfile", help="backend log, e.g. podcast_test.log")
    args = ap.parse_args()

    r = Report()
    with open(args.logfile, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            r.feed(line)
    return emit(r)


if __name__ == "__main__":
    sys.exit(main())
