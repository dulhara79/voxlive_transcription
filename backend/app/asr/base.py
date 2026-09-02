"""
Provider abstraction. The rest of the app only knows about SpeechProvider,
never about Gemini specifically. Swapping engines = config change.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from .language_spans import LanguageSpan


@dataclass
class TranscriptResult:
    text: str
    language: str  # ISO code: "si" | "en" | "ta" — the DOMINANT language
    confidence: Optional[float] = None  # 0..1 if available
    avg_logprob: Optional[float] = None
    # Word/phrase-level language structure (Phase 4). `language` above stays
    # for the paragraph colour and the allowed-languages gate; this is where
    # code-switching actually lives. Empty when the text carries no script
    # (pure digits or punctuation), which is a real and meaningful state.
    language_spans: list[LanguageSpan] = field(default_factory=list)


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
