#!/usr/bin/env python3
"""probe_vertex_regions.py — which locations actually serve YOUR model?

    python tools/probe_vertex_regions.py
    python tools/probe_vertex_regions.py --model gemini-3.6-flash --regions global,asia-south1,asia-southeast1

WHY NOT JUST PIN A REGION
-------------------------
The startup warning says `GOOGLE_CLOUD_LOCATION=global` adds routing latency
and tells you to pin a region. That advice is right about latency and can
still be wrong for your project, because newer Gemini models are frequently
released on the GLOBAL endpoint FIRST and reach regional endpoints later, per
project, behind an allowlist. Pinning a region the model is not served in
does not degrade — every ASR call fails outright with "Publisher Model ...
was not found".

Region availability also depends on the project, so no documentation page can
answer this for you. This sends one tiny generateContent call per candidate
region and reports what came back. It is the only reliable answer.

WHAT TO DO WITH THE RESULT
--------------------------
  * a nearby region works  -> set GOOGLE_CLOUD_LOCATION to it. Lower latency,
                              and — the part that matters more for SLT — you
                              can state where the audio was processed, which
                              `global` explicitly cannot tell you. Under the
                              PDPA that is a data-residency question, not a
                              performance one.
  * only global works      -> leave it, and record the residency exposure as
                              a known risk rather than an accident. Ask your
                              Google Cloud contact for regional allowlisting.

COST: one ~10-token request per region. Cents at most.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

DEFAULT_REGIONS = [
    "global",
    "asia-south1",  # Mumbai — nearest to Colombo
    "asia-southeast1",  # Singapore — usually the best-stocked APAC region
    "asia-northeast1",  # Tokyo
    "us-central1",  # Iowa — widest model coverage, worst latency from LK
]


def probe(
    project: str, location: str, model: str, timeout: float
) -> tuple[str, float, str]:
    """Returns (verdict, seconds, detail). Verdict is OK / UNAVAILABLE / ERROR."""
    from google import genai
    from google.genai import types

    t0 = time.perf_counter()
    try:
        client = genai.Client(vertexai=True, project=project, location=location)
        resp = client.models.generate_content(
            model=model,
            contents="ping",
            config=types.GenerateContentConfig(
                max_output_tokens=8,
                # Matches the ASR path: no thinking budget, so the timing here
                # is comparable to a real transcription round trip.
                temperature=0.0,
            ),
        )
        dt = time.perf_counter() - t0
        _ = getattr(resp, "text", "")
        return "OK", dt, "served"
    except Exception as exc:  # noqa: BLE001
        dt = time.perf_counter() - t0
        msg = str(exc).replace("\n", " ")
        low = msg.lower()
        if "was not found" in low or "not_found" in low or "404" in low:
            return "UNAVAILABLE", dt, "model not published in this location"
        if "permission" in low or "403" in low:
            return "ERROR", dt, "permission denied — check the service account"
        if "429" in low or "resource_exhausted" in low:
            return "OK", dt, "served, but rate limited right now"
        return "ERROR", dt, msg[:110]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--project", default=os.getenv("GOOGLE_CLOUD_PROJECT"))
    ap.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"))
    ap.add_argument("--regions", default=",".join(DEFAULT_REGIONS))
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args()

    if not args.project:
        print(
            "No project. Pass --project or set GOOGLE_CLOUD_PROJECT.\n"
            "GOOGLE_APPLICATION_CREDENTIALS must also point at the service "
            "account JSON.",
            file=sys.stderr,
        )
        return 2

    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    print(f"\nproject : {args.project}")
    print(f"model   : {args.model}")
    print(f"{'region':<20} {'result':<13} {'latency':>9}   detail")
    print("-" * 78)

    served: list[tuple[str, float]] = []
    for region in regions:
        verdict, dt, detail = probe(args.project, region, args.model, args.timeout)
        print(f"{region:<20} {verdict:<13} {dt * 1000:>7.0f}ms   {detail}")
        if verdict == "OK":
            served.append((region, dt))

    print()
    regional = [(r, d) for r, d in served if r != "global"]
    if regional:
        best, dt = min(regional, key=lambda x: x[1])
        print(f"Fastest regional endpoint: {best} ({dt * 1000:.0f}ms)")
        print(f"  GOOGLE_CLOUD_LOCATION={best}")
        print(
            "  Pinning this also makes the processing location STATEABLE, "
            "which\n  `global` cannot be. That is the PDPA-relevant difference."
        )
    elif served:
        print("Only the global endpoint serves this model for this project.")
        print("  Leave GOOGLE_CLOUD_LOCATION=global. Pinning a region would")
        print("  fail every ASR call, not merely slow it down.")
        print("  Record the residency exposure and request regional access.")
    else:
        print("No endpoint served this model. Check the model name and the")
        print("service account's Vertex AI User role before anything else.")

    return 0 if served else 1


if __name__ == "__main__":
    sys.exit(main())
