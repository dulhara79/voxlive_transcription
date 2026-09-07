"""
diarizer_factory.py — choose a diarization backend.

Both backends expose the same surface, so main.py never branches on which one
is running:

    feed / start / aclose / finalize / wait_for_coverage
    label_for / split_points / timeline / engine.speaker_count / stats / reset

DIARIZATION_BACKEND=embedding   (default) pyannote WeSpeaker embeddings +
                                session-global clustering. CPU-friendly, no
                                speaker ceiling, cannot represent overlapping
                                speech.

DIARIZATION_BACKEND=sortformer  NVIDIA Streaming Sortformer. End-to-end,
                                handles overlap, better DER — but wants a GPU,
                                caps at 4 speakers, and was trained primarily
                                on English. A/B it on real Sinhala/Tamil audio
                                before trusting it.

DIARIZATION_BACKEND=auto        sortformer if NeMo imports AND a CUDA device
                                is visible; embedding otherwise. Never silently
                                puts Sortformer on a CPU, because that turns a
                                200 ms pass into a multi-second one and the
                                symptom looks like a hang, not a config choice.
"""

from __future__ import annotations

import logging

log = logging.getLogger("voxlive.diar")


def _sortformer_viable() -> tuple[bool, str]:
    try:
        import torch
    except ImportError:
        return False, "torch not installed"
    try:
        import nemo.collections.asr  # noqa: F401
    except ImportError:
        return False, "nemo_toolkit[asr] not installed"
    if not torch.cuda.is_available():
        return False, "no CUDA device"
    return True, "ok"


def resolve_backend(requested: str) -> str:
    req = (requested or "embedding").strip().lower()
    if req == "sortformer":
        ok, why = _sortformer_viable()
        if not ok:
            # Requested explicitly, so warn loudly rather than quietly
            # substituting something else.
            log.warning(
                "DIARIZATION_BACKEND=sortformer requested but %s — "
                "falling back to the embedding backend.",
                why,
            )
            return "embedding"
        return "sortformer"
    if req == "auto":
        ok, why = _sortformer_viable()
        log.info(
            "diarization backend auto: %s", "sortformer" if ok else f"embedding ({why})"
        )
        return "sortformer" if ok else "embedding"
    return "embedding"


def build_diarizer(settings, expected_speakers: int, speaker_mode: str = ""):
    """Construct the configured backend for one session.

    `speaker_mode` is per-session ("auto" | "fixed"), sent by the client and
    falling back to SPEAKER_MODE from .env. Only the embedding backend honours
    it — Sortformer decides its own speaker count end-to-end and has no K to
    force — so a fixed request is logged and ignored there rather than being
    silently accepted.
    """
    backend = resolve_backend(getattr(settings, "diarization_backend", "embedding"))
    mode = (speaker_mode or getattr(settings, "speaker_mode", "auto") or "auto").lower()

    if backend == "sortformer":
        if mode == "fixed":
            log.warning(
                "speaker_mode=fixed ignored: the Sortformer backend derives its "
                "own speaker count and exposes no K to force. Use "
                "DIARIZATION_BACKEND=embedding for a known speaker count."
            )
        from .sortformer import SortformerDiarizer

        return SortformerDiarizer(
            sample_rate=settings.sample_rate,
            interval_sec=max(2.0, settings.diarize_interval_sec),
            window_sec=float(getattr(settings, "sortformer_window_sec", 90.0)),
            max_speakers=min(4, settings.max_speakers),
            enabled=settings.diarization_enabled,
        )

    from .stable_service import StableDiarizationService

    # §14: SORTFORMER_WINDOW_SEC is inert here. It is only ever read by the
    # Sortformer branch above, so tuning it while DIARIZATION_BACKEND=embedding
    # changes nothing — a false lead that already cost one debugging session.
    # Say so instead of letting the value sit in .env looking effective.
    sortformer_window = float(getattr(settings, "sortformer_window_sec", 90.0))
    if abs(sortformer_window - 90.0) > 1e-6:
        log.warning(
            "SORTFORMER_WINDOW_SEC=%.0f has NO EFFECT: the active backend is "
            "'embedding', which does not read it. Changing this value will not "
            "alter speaker-count estimation. Set DIARIZATION_BACKEND=sortformer "
            "if you meant to use that backend.",
            sortformer_window,
        )

    # §5 of the review: the EFFECTIVE configuration, at the point it is
    # decided, in one line. The 2-speaker investigation cost days because the
    # runtime values were only ever visible in a .env file that is not in the
    # repository, and `-> 2 speaker(s)` looks identical whether the engine
    # estimated 2 or was ordered to produce 2.
    log.info(
        "diarization runtime config: backend=%s speaker_mode=%s "
        "expected_speakers=%s max_speakers=%d establish=%.1fs%s",
        backend,
        mode,
        expected_speakers or "0 (auto)",
        settings.max_speakers,
        float(getattr(settings, "speaker_establish_sec", 8.0)),
        (
            f" -> K IS FORCED TO {expected_speakers}; Speaker "
            f"{expected_speakers + 1} cannot be created"
            if mode == "fixed" and expected_speakers > 0
            else " -> K estimated from the audio"
        ),
    )

    return StableDiarizationService(
        hf_token=settings.huggingface_token,
        sample_rate=settings.sample_rate,
        expected_speakers=expected_speakers,
        max_speakers=settings.max_speakers,
        speaker_mode=mode,
        establish_sec=float(getattr(settings, "speaker_establish_sec", 8.0)),
        interval_sec=settings.diarize_interval_sec,
        vad_aggressiveness=settings.vad_aggressiveness,
        enabled=settings.diarization_enabled,
        # §9/§10/§12 tunables. Passed explicitly rather than read from the
        # global settings inside the engine, so tests can construct an engine
        # with different values without touching the environment.
        same_speaker_max=float(getattr(settings, "same_speaker_max", 0.50)),
        same_speaker_floor=float(getattr(settings, "speaker_separation_floor", 0.35)),
        separation_ratio=float(getattr(settings, "speaker_separation_ratio", 1.10)),
        new_identity_min_sec=float(getattr(settings, "new_identity_min_sec", 6.0)),
        new_identity_min_windows=int(getattr(settings, "new_identity_min_windows", 4)),
        new_identity_short_sec=float(getattr(settings, "new_identity_short_sec", 3.2)),
        new_identity_short_windows=int(
            getattr(settings, "new_identity_short_windows", 3)
        ),
        new_identity_strong_dist=float(
            getattr(settings, "new_identity_strong_dist", 0.68)
        ),
        growth_check_sec=float(getattr(settings, "speaker_growth_check_sec", 20.0)),
    )


async def warmup_backend(settings) -> None:
    """Load whichever model the configured backend needs, at startup."""
    import asyncio

    if not settings.diarization_enabled:
        log.info("Diarization: OFF — everything labelled Speaker 1")
        return

    backend = resolve_backend(getattr(settings, "diarization_backend", "embedding"))
    if backend == "sortformer":
        from .sortformer import warmup

        await asyncio.to_thread(warmup)
        log.info("Diarization: Sortformer streaming (warm)")
    else:
        from .embedder import warmup

        await asyncio.to_thread(warmup, settings.huggingface_token)
        log.info("Diarization: session-global clustering v10 (embedder warm)")
