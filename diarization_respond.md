# Diarization review — response

Against the review of `fature/optimize_diarization`. One section per point, and
where a point was **not** implemented, that is stated rather than omitted.

---

## The headline

The review's central instruction was: **do not keep tuning
`SORTFORMER_WINDOW_SEC`; run the raw-vs-rolling experiment first.** So the
window is back to its documented default and there is now a tool that runs that
experiment and prints a verdict.

```bash
cd backend
python diag_diarization.py news.wav --truth news.txt
```

Nothing else in this changeset should be evaluated before that has been run on a
real recording, because until then nobody knows whether the bottleneck is the
model, our stitching, or neither.

---

## Point by point

| # | Review point | Status |
|---|---|---|
| 1, 2 | Window length is not the fix; stitching is suspect | Acknowledged — drives everything below |
| 3 | 4-speaker ceiling is architectural | Documented, surfaced in the verdict |
| 4 | Streaming model used offline; separate the regimes | **Done** — live vs final are now different code paths |
| 5, 6, 7 | Gemini can diarize, but not as the live diarizer | **Not implemented** — see below |
| 8 | Word-level speaker info | **Not implemented** |
| 9 | Two independent diarization signals | Partly — the machinery exists, the second signal does not |
| 10 | Deterministic reconciliation, not "Gemini wins" | **Done** — `reconcile.py` |
| 11 | `finalize()` should be a full-session pass | **Done** |
| 12, 13 | Benchmark pyannote Community-1 | **Adapter written, never executed** |
| 14 | `WINDOW_SEC` is the wrong optimisation target | **Done** — reverted to 90, documented |
| 15 | Split `_stitch()` into five stages | **Done**, with an equivalence test |
| 16 | Run raw vs rolling before changing anything else | **Done** — `diag_diarization.py` |
| 17 | Dump raw / stitched / final for a debugging chain | **Done** |
| 18, 19 | Target architecture | Partly built; the offline half exists, the cross-check half does not |

---

## What changed

### `app/diarization/stitching.py` (new) — point 15

`_stitch()` became five objects:

```
LocalSpeakerTimeline  -> SpeakerAlignment -> SessionIdentityManager -> GlobalTimeline
```

The point of the split is attribution: `SpeakerAlignment` is pure and only
reports what the overlap evidence supports, while `SessionIdentityManager` owns
the guessing — minting, the cap, and the `_nearest_prior_speaker` fallback that
can turn a true `A B C D` into `A B A B`. Every fold is now counted
(`cap_folds` in `stats()`), so that failure mode is visible instead of being a
debug log nobody reads.

**The algorithm is unchanged.** `tests/test_stitching.py` keeps a verbatim copy
of the original function and asserts the two agree across 200 randomised
sessions. That was deliberate — a refactor that also changed behaviour would
invalidate any measurement taken before it.

### `app/diarization/sortformer.py` — points 4, 11, 17

`finalize()` no longer calls `_pass()`. It runs one inference over the whole
recording with the non-streaming checkpoint and replaces the timeline outright.
The rolling result is kept as `rolling_timeline()` so the two can be compared.

If the offline pass cannot run — no checkpoint, no GPU, session past the
duration ceiling, recording disabled — it is **skipped and the rolling timeline
kept**, with the reason in `stats()["final_pass"]`. It is never half-applied.

Note that `session/state.py` already described this call as *"plain offline
diarization — the best labelling the system can produce"*. That comment was
false for the Sortformer backend. It is now true.

### `app/diarization/offline.py` (new) — points 4, 13

Whole-file diarizers behind one interface: offline Sortformer and pyannote
Community-1. Adding the Gemini candidate is one function and one line.

### `app/diarization/reconcile.py` (new) — point 10

Overlap matrix → Hungarian → agreement score → located disagreement regions.
Pure arithmetic, no model, no I/O. Works on any two timelines.

It answers "where do these two disagree", never "which is right" — two systems
can be wrong together. Its real use is aiming ten minutes of annotation at the
regions that matter instead of the whole file.

### `diag_diarization.py` (new) — points 16, 17

Runs the same audio through the raw whole-file model and the production path,
scores both, and prints one of four verdicts:

- raw good, shipped bad → **the stitching**; changing models will not help
- both bad → **the model or the audio**; stitching work will not help
- both good → **not diarization**; look at `TranscriptStore.label_for()`
- shipped beat raw → odd, check audio length and GPU memory

Writes `raw_sortformer.json`, `stitched_timeline.json`, `summary.json`.

---

## Settings

| Setting | Was | Now |
|---|---|---|
| `SORTFORMER_WINDOW_SEC` | `20` in `.env.example` | `90` (the documented default) |
| `SORTFORMER_FINAL_PASS` | — | `offline` |
| `SORTFORMER_OFFLINE_MODEL_ID` | — | blank → `nvidia/diar_sortformer_4spk-v1` |
| `SORTFORMER_FINAL_MAX_SEC` | — | `720` |
| `DIARIZE_DIAGNOSTICS_DIR` | — | blank (off) |

`config/base.py` always defaulted to 90, so the `20` was only live for anyone
copying `.env.example`.

---

## Two things that need a decision, not more code

**1. Licence.** The offline checkpoint `nvidia/diar_sortformer_4spk-v1` is
**CC-BY-NC-4.0** — non-commercial. The streaming model already in use is
CC-BY-4.0. Enabling the offline final pass on an SLT deployment is therefore a
licensing question. Flagged in three places in the code; not resolvable in code.

**2. Memory.** `SORTFORMER_FINAL_PASS=offline` holds the whole session's audio,
about 1.9 MB per minute per session. On a multi-tenant box that is a real
number. `SORTFORMER_FINAL_MAX_SEC=720` bounds it; past the ceiling the offline
pass is skipped rather than run over a truncated recording, because a final pass
over "the last 12 minutes" silently answers a different question.

---

## What was deliberately NOT done

**Gemini diarization (points 5–9) is not implemented.** The review's own
sequencing is the reason: point 16 says run the raw-vs-rolling comparison
*before* modifying the code again, and points 6 and 10 say Gemini must not be
the live diarizer nor decide final labels alone. The prerequisite — a
deterministic reconciliation layer — now exists. Wiring the second signal is the
next commit, after the diagnostic has been run.

**Nothing in `offline.py` has been executed.** No GPU, no checkpoints, no HF
token were available. Both APIs were checked against the current model cards
rather than written from memory, but the first run of `diag_diarization.py` is
also the first test of that file. **Suspect this code before suspecting the
model.**

**No claim is made that diarization is better.** No recording has been scored.
This changeset makes the system measurable and moves the final pass onto the
whole recording; whether either helps is what the diagnostic is for.

---

## Tests

```bash
cd backend && python -m pytest tests/ -q
```

New: `test_stitching.py` (equivalence vs the original), `test_reconcile.py`,
`test_sortformer_final_pass.py`, `test_diag_smoke.py` (the diagnostic runs end
to end with the model stubbed). `test_sortformer.py` was updated to the new
stitcher API — the assertions are unchanged.

Three failures in `test_diarization_scoring.py` are a missing `webrtcvad`
dependency in the dev environment and pre-date this work; confirmed by running
them on a clean checkout of the branch.

---

## Next

1. Save one real news recording, annotate it in Audacity, run the diagnostic.
2. Act on the verdict — and only then decide whether to touch a model or a
   setting.
3. If the model is the bottleneck, `--also pyannote_community1` is the first
   alternative the review named.