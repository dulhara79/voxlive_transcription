"""Embedding diarization service wired to StableSpeakerEngine."""

from __future__ import annotations

from typing import Optional

from .service import DiarizationService
from .stable_speaker_engine import StableSpeakerEngine


class StableDiarizationService(DiarizationService):
    """DiarizationService using the stability-fixed speaker engine."""

    def __init__(
        self,
        hf_token: str,
        sample_rate: int = 16000,
        expected_speakers: int = 0,
        max_speakers: int = 6,
        speaker_mode: str = "auto",
        establish_sec: float = 8.0,
        interval_sec: float = 1.5,
        vad_aggressiveness: int = 2,
        device: Optional[str] = None,
        enabled: bool = True,
        calibrate: bool = True,
        separation_margin: float = 0.0,
        detect_turns: bool = True,
        **engine_kw,
    ):
        # Let the base service create its scheduler/embedder/buffering state.
        super().__init__(
            hf_token=hf_token,
            sample_rate=sample_rate,
            expected_speakers=expected_speakers,
            max_speakers=max_speakers,
            speaker_mode=speaker_mode,
            establish_sec=establish_sec,
            interval_sec=interval_sec,
            vad_aggressiveness=vad_aggressiveness,
            device=device,
            enabled=enabled,
            calibrate=calibrate,
            separation_margin=separation_margin,
            detect_turns=detect_turns,
            **engine_kw,
        )

        # SpeakerEngine itself is lightweight; the expensive model is the
        # Embedder above, which is retained. Replace only the state machine.
        self.engine = StableSpeakerEngine(
            expected_speakers=expected_speakers,
            max_speakers=max_speakers,
            speaker_mode=speaker_mode,
            establish_sec=establish_sec,
            calibrate=calibrate,
            separation_margin=separation_margin,
            detect_turns=detect_turns,
            **engine_kw,
        )
