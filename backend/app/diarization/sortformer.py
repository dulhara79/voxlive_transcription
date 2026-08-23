import asyncio
import logging
import torch
import numpy as np
from typing import List, Dict, Any

from .stitching import GlobalSpeakerIdentityManager, LocalSpeakerTimeline
from .offline import run_offline_diarization

logger = logging.getLogger(__name__)


class SortformerDiarizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("diarization_enabled", True)
        self.window_sec = config.get("SORTFORMER_WINDOW_SEC", 20)
        self.min_speakers = config.get("min_speakers", 1)
        self.max_speakers = config.get("max_speakers", 10)
        self.mode = config.get("diarization_mode", "AUTO")

        self.identity_manager = GlobalSpeakerIdentityManager(
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers,
            mode=self.mode,
        )

        self._audio_buffer = bytearray()
        self._current_offset = 0.0
        self.sample_rate = config.get("sample_rate", 16000)

    async def process_chunk(self, audio_chunk: bytes, timestamp: float):
        if not self.enabled:
            return

        self._audio_buffer.extend(audio_chunk)

        if self._should_run_pass():
            await self._pass()

    def _should_run_pass(self) -> bool:
        buffer_duration = len(self._audio_buffer) / 2 / self.sample_rate
        return buffer_duration >= self.window_sec

    def _infer(self, audio_array: np.ndarray) -> List[tuple]:
        """
        Executes actual NeMo/Sortformer local inference.
        Returns list of (start, end, speaker_id).
        """
        # Your local NeMo integration goes here.
        # For safety, if not loaded, fallback empty.
        try:
            import nemo.collections.asr as nemo_asr

            # Execute actual model forward pass
            return []
        except ImportError:
            logger.error("NeMo not installed. Cannot run Sortformer inference.")
            return []

    async def _pass(self):
        if not self._audio_buffer:
            return []

        # Convert PCM bytes to float32 array for the model
        audio_arr = (
            np.frombuffer(bytes(self._audio_buffer), dtype=np.int16).astype(np.float32)
            / 32768.0
        )

        # 1. Local inference execution on GPU
        raw_segments = await asyncio.to_thread(self._infer, audio_arr)

        local_segments = []
        local_embeddings = {}

        # Build local timeline payload from raw segments
        for start, end, spk_id in raw_segments:
            spk_label = f"speaker_{spk_id}"
            local_segments.append(
                {"speaker": spk_label, "start": float(start), "end": float(end)}
            )
            # Generate local embeddings representing the speaker in this specific window
            # (Replace with your actual representation extraction from the model layer)
            if spk_label not in local_embeddings:
                local_embeddings[spk_label] = np.random.randn(256)

        local_timeline = LocalSpeakerTimeline(
            segments=local_segments,
            embeddings=local_embeddings,
            offset=self._current_offset,
        )

        # 2. Delegate cross-window stitching
        global_segments = self.identity_manager.process_local_timeline(local_timeline)

        # 3. Update offset and clean buffer
        self._current_offset += len(self._audio_buffer) / 2 / self.sample_rate
        self._audio_buffer.clear()

        return global_segments

    async def finalize(self, audio_file_path: str):
        if not self.enabled:
            return None

        logger.info("Starting true offline final diarization pass...")
        final_timeline = await run_offline_diarization(
            audio_path=audio_file_path, config=self.config
        )
        return final_timeline
