"""
Provider abstraction. The rest of the app only knows about SpeechProvider,
never about Gemini specifically. Swapping engines = config change.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class TranscriptResult:
    text: str
    language: str  # ISO code: "si" | "en" | "ta"
    confidence: Optional[float] = None  # 0..1 if available
    avg_logprob: Optional[float] = None


class SpeechProvider(ABC):
    """Transcribe one finalized audio segment (already VAD-trimmed).

    `context` is an optional string of RECENT transcript from the same
    conversation. LLM-based engines (Gemini) use it to disambiguate short,
    code-switched segments - the same reason a full 1-minute upload
    transcribes better than an isolated 3-second chunk. Acoustic engines may
    ignore it.
    """

    @abstractmethod
    async def transcribe_segment(
        self, pcm_bytes: bytes, sample_rate: int, context: Optional[str] = None
    ) -> TranscriptResult: ...
