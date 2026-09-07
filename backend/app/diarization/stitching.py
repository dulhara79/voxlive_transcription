import numpy as np
from typing import List, Dict, Any, Optional
from scipy.optimize import linear_sum_assignment


class SpeakerEvidence:
    def __init__(self, candidate_id: str):
        self.candidate_id = candidate_id
        self.embedding_history: List[np.ndarray] = []
        self.total_duration: float = 0.0
        self.observations: int = 0
        self.last_seen: float = 0.0
        self.status: str = "CANDIDATE"  # CANDIDATE or ESTABLISHED

    def add_evidence(self, embedding: np.ndarray, duration: float, timestamp: float):
        self.embedding_history.append(embedding)
        self.total_duration += duration
        self.observations += 1
        self.last_seen = timestamp

        if (
            self.status == "CANDIDATE"
            and self.total_duration > 3.0
            and self.observations >= 2
        ):
            self.status = "ESTABLISHED"

    def get_centroid(self) -> np.ndarray:
        if not self.embedding_history:
            return np.zeros(0)
        return np.mean(self.embedding_history, axis=0)


class LocalSpeakerTimeline:
    def __init__(
        self, segments: List[Dict], embeddings: Dict[str, np.ndarray], offset: float
    ):
        self.segments = segments
        self.embeddings = embeddings
        self.offset = offset


class GlobalSpeakerIdentityManager:
    def __init__(
        self, min_speakers: int = 1, max_speakers: int = 10, mode: str = "AUTO"
    ):
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.mode = mode

        self.global_speakers: Dict[str, SpeakerEvidence] = {}
        self.global_timeline: List[Dict] = []
        self._next_candidate_idx = 0

    def _generate_candidate_id(self) -> str:
        ident = f"Unknown_Voice_{self._next_candidate_idx}"
        self._next_candidate_idx += 1
        return ident

    def process_local_timeline(
        self, local_timeline: LocalSpeakerTimeline
    ) -> List[Dict]:
        mapped_segments = []
        local_to_global_map = {}

        local_speakers = list(local_timeline.embeddings.keys())
        established_globals = [
            sid
            for sid, ev in self.global_speakers.items()
            if ev.status == "ESTABLISHED"
        ]
        candidate_globals = [
            sid for sid, ev in self.global_speakers.items() if ev.status == "CANDIDATE"
        ]

        all_globals = established_globals + candidate_globals

        if local_speakers and all_globals:
            cost_matrix = np.zeros((len(local_speakers), len(all_globals)))
            for i, l_spk in enumerate(local_speakers):
                l_emb = local_timeline.embeddings[l_spk]
                for j, g_spk in enumerate(all_globals):
                    g_emb = self.global_speakers[g_spk].get_centroid()
                    cost_matrix[i, j] = 1 - np.dot(l_emb, g_emb) / (
                        np.linalg.norm(l_emb) * np.linalg.norm(g_emb)
                    )

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # INCREASED THRESHOLD: Relaxed from 0.35 to 0.50 to prevent over-segmentation
            # and align with SAME_SPEAKER_MAX in speaker_engine.py
            THRESHOLD = 0.50

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] <= THRESHOLD:
                    local_to_global_map[local_speakers[r]] = all_globals[c]

        for l_spk in local_speakers:
            if l_spk not in local_to_global_map:
                if (
                    self.mode == "FIXED_N"
                    and len(established_globals) >= self.max_speakers
                ):
                    local_to_global_map[l_spk] = "UNKNOWN_SPEAKER"
                else:
                    new_id = self._generate_candidate_id()
                    self.global_speakers[new_id] = SpeakerEvidence(new_id)
                    local_to_global_map[l_spk] = new_id

        for seg in local_timeline.segments:
            l_spk = seg["speaker"]
            g_spk = local_to_global_map.get(l_spk, "UNKNOWN_SPEAKER")

            duration = seg["end"] - seg["start"]

            if g_spk != "UNKNOWN_SPEAKER":
                self.global_speakers[g_spk].add_evidence(
                    embedding=local_timeline.embeddings[l_spk],
                    duration=duration,
                    timestamp=local_timeline.offset + seg["start"],
                )

            final_name = (
                g_spk
                if g_spk == "UNKNOWN_SPEAKER"
                or self.global_speakers[g_spk].status == "ESTABLISHED"
                else "CANDIDATE_SPEAKER"
            )

            mapped_seg = {
                "speaker": final_name,
                "start": local_timeline.offset + seg["start"],
                "end": local_timeline.offset + seg["end"],
            }
            mapped_segments.append(mapped_seg)
            self.global_timeline.append(mapped_seg)

        return mapped_segments
