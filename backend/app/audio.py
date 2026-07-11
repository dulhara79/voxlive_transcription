"""
Streaming VAD segmenter.

Consumes raw 16kHz mono int16 PCM (as bytes) and yields finalized speech
segments. This is what makes a batch model usable in "real time": we cut the
audio at natural pauses and transcribe one utterance at a time.

webrtcvad works on 10/20/30 ms frames of 16-bit mono PCM. We use 30 ms.

SEGMENT-END RULES (first one to fire wins):
  1. silence   -> `silence_ms` of trailing silence (normal end of a turn).
  2. soft gap  -> once a segment passes `soft_max_segment_ms`, end it at the
                  NEXT micro-gap between words. This keeps a non-stop speaker
                  (online meeting, lecture) flowing in short, clean lines
                  instead of one slab, WITHOUT slicing through a word.
  3. hard max  -> `max_segment_ms` absolute ceiling. Only hit if someone truly
                  never pauses; this is the one cut that may land mid-word, so
                  it's a last resort, set generously.

ACCURACY - pre-roll padding:
  The VAD only TRIGGERS on a frame loud enough to count as speech, so the soft
  onset of a word gets dropped. We keep a small ring buffer of recent pre-speech
  frames and prepend them on trigger so word onsets aren't clipped.

NOTE on sample rate: webrtcvad ONLY accepts 8000/16000/32000/48000 Hz. The
frontend resamples to exactly 16000 in the AudioWorklet, so this is 16000.
"""
import logging
from collections import deque

import webrtcvad

log = logging.getLogger("voxlive.vad")

_VALID_RATES = (8000, 16000, 32000, 48000)


class VADSegmenter:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        vad_aggressiveness: int = 2,
        silence_ms: int = 500,
        max_segment_ms: int = 8000,
        min_segment_ms: int = 300,
        pre_roll_ms: int = 250,
        soft_max_segment_ms: int = 4000,
    ):
        if frame_ms not in (10, 20, 30):
            raise ValueError("frame_ms must be 10, 20 or 30 (webrtcvad limitation)")
        if sample_rate not in _VALID_RATES:
            raise ValueError(
                f"webrtcvad only supports {_VALID_RATES}, got {sample_rate}. "
                "The frontend must resample to 16000."
            )

        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_bytes = int(sample_rate * frame_ms / 1000) * 2  # int16 = 2 bytes
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.silence_frames = max(1, silence_ms // frame_ms)
        self.max_frames = max(1, max_segment_ms // frame_ms)
        # soft cap must sit below the hard ceiling to be useful
        self.soft_max_frames = min(max(1, soft_max_segment_ms // frame_ms), self.max_frames)
        self.min_bytes = int(sample_rate * min_segment_ms / 1000) * 2

        self.pre_roll_frames = max(0, pre_roll_ms // frame_ms)
        self._preroll = (
            deque(maxlen=self.pre_roll_frames) if self.pre_roll_frames else None
        )

        self._inbuf = bytearray()
        self._seg = bytearray()
        self._triggered = False
        self._silence_run = 0
        self._frames_in_seg = 0

        log.info(
            "VADSegmenter ready: rate=%d frame=%dms aggr=%d silence=%dms "
            "soft_max=%dms max=%dms min=%dms preroll=%dms",
            sample_rate, frame_ms, vad_aggressiveness, silence_ms,
            soft_max_segment_ms, max_segment_ms, min_segment_ms, pre_roll_ms,
        )

    def add_audio(self, pcm_bytes: bytes) -> list[bytes]:
        """Feed raw PCM. Returns a list of finalized segments (may be empty)."""
        finalized: list[bytes] = []
        self._inbuf.extend(pcm_bytes)
        while len(self._inbuf) >= self.frame_bytes:
            frame = bytes(self._inbuf[: self.frame_bytes])
            del self._inbuf[: self.frame_bytes]
            seg = self._process_frame(frame)
            if seg is not None:
                finalized.append(seg)
        return finalized

    def _process_frame(self, frame: bytes):
        is_speech = self.vad.is_speech(frame, self.sample_rate)

        if not self._triggered:
            if is_speech:
                self._triggered = True
                self._seg = bytearray()
                self._frames_in_seg = 0
                if self._preroll:
                    for f in self._preroll:
                        self._seg.extend(f)
                    self._frames_in_seg = len(self._preroll)
                    self._preroll.clear()
                self._seg.extend(frame)
                self._frames_in_seg += 1
                self._silence_run = 0
            elif self._preroll is not None:
                self._preroll.append(frame)
            return None

        # inside a speech run
        self._seg.extend(frame)
        self._frames_in_seg += 1
        self._silence_run = 0 if is_speech else self._silence_run + 1

        hit_silence = self._silence_run >= self.silence_frames
        # past the soft cap, end at the first micro-gap (a single non-speech
        # frame) so long monologues are chopped into clean near-real-time lines.
        soft_cut = self._frames_in_seg >= self.soft_max_frames and not is_speech
        hit_max = self._frames_in_seg >= self.max_frames  # may cut mid-word

        if hit_silence or soft_cut or hit_max:
            seg = bytes(self._seg)
            reason = "silence" if hit_silence else ("soft_gap" if soft_cut else "max_len")
            self._reset_segment()
            if len(seg) >= self.min_bytes:
                log.debug("segment finalized (%s): %d bytes", reason, len(seg))
                return seg
            log.debug("segment dropped (<min, %s): %d bytes", reason, len(seg))
            return None
        return None

    def _reset_segment(self):
        self._seg = bytearray()
        self._triggered = False
        self._silence_run = 0
        self._frames_in_seg = 0

    def flush(self):
        """Force-finalize whatever speech is buffered (call on stop)."""
        seg = bytes(self._seg) if self._triggered else b""
        self._reset_segment()
        return seg if len(seg) >= self.min_bytes else None
