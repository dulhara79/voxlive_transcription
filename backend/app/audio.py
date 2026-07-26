"""
Streaming VAD segmenter.

Consumes raw 16 kHz mono int16 PCM and yields finalized speech segments with
EXACT absolute timing.

WHAT CHANGED IN v10
-------------------
v7 returned bare `bytes` and main.py reconstructed the timing:

    seg_start = max(0.0, end_time - duration)      # end_time = bytes_seen/2/sr

That is wrong by however much audio is sitting in the segmenter's internal
frame buffer, and it drifts further because pre-roll padding prepends frames
from *before* the segment. A few tens of milliseconds does not matter for
display, but it matters a lot now that speaker labels are joined to text by
timestamp overlap: a segment nudged 200 ms late can borrow the label of the
turn after it.

The segmenter knows exactly how many frames it has consumed, so it now reports
(pcm, start, end) itself and there is only one clock in the system.

SEGMENT-END RULES (first to fire wins):
  1. silence   -> `silence_ms` of trailing silence (normal end of a turn).
  2. soft gap  -> past `soft_max_segment_ms`, end at the NEXT micro-gap. Keeps
                  a non-stop speaker flowing in short lines without slicing a
                  word.
  3. hard max  -> `max_segment_ms` ceiling; may land mid-word, so set it
                  generously.

webrtcvad only accepts 8000/16000/32000/48000 Hz; the AudioWorklet resamples
to exactly 16000.
"""

import logging
from collections import deque
from dataclasses import dataclass

import webrtcvad

log = logging.getLogger("voxlive.vad")

_VALID_RATES = (8000, 16000, 32000, 48000)


@dataclass
class Segment:
    pcm: bytes
    start: float  # absolute session seconds
    end: float
    reason: str  # silence | soft_gap | max_len | flush

    @property
    def duration(self) -> float:
        return self.end - self.start


class VADSegmenter:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        vad_aggressiveness: int = 2,
        silence_ms: int = 320,
        max_segment_ms: int = 9000,
        min_segment_ms: int = 300,
        pre_roll_ms: int = 250,
        soft_max_segment_ms: int = 3500,
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
        self.frame_sec = frame_ms / 1000.0
        self.frame_bytes = int(sample_rate * frame_ms / 1000) * 2
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.silence_frames = max(1, silence_ms // frame_ms)
        self.max_frames = max(1, max_segment_ms // frame_ms)
        self.soft_max_frames = min(
            max(1, soft_max_segment_ms // frame_ms), self.max_frames
        )
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
        self._frames_consumed = 0  # THE clock: frames read off the wire
        self._seg_start_frame = 0

        log.info(
            "VADSegmenter ready: rate=%d frame=%dms aggr=%d silence=%dms "
            "soft_max=%dms max=%dms min=%dms preroll=%dms",
            sample_rate,
            frame_ms,
            vad_aggressiveness,
            silence_ms,
            soft_max_segment_ms,
            max_segment_ms,
            min_segment_ms,
            pre_roll_ms,
        )

    @property
    def now(self) -> float:
        """Absolute session seconds consumed so far."""
        return self._frames_consumed * self.frame_sec

    def add_audio(self, pcm_bytes: bytes) -> list[Segment]:
        finalized: list[Segment] = []
        self._inbuf.extend(pcm_bytes)
        while len(self._inbuf) >= self.frame_bytes:
            frame = bytes(self._inbuf[: self.frame_bytes])
            del self._inbuf[: self.frame_bytes]
            self._frames_consumed += 1
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
                # The segment starts at the FIRST pre-roll frame, not at the
                # frame that tripped the VAD — otherwise every start timestamp
                # is late by the pre-roll length.
                pre = len(self._preroll) if self._preroll else 0
                self._seg_start_frame = self._frames_consumed - 1 - pre
                if self._preroll:
                    for f in self._preroll:
                        self._seg.extend(f)
                    self._frames_in_seg = pre
                    self._preroll.clear()
                self._seg.extend(frame)
                self._frames_in_seg += 1
                self._silence_run = 0
            elif self._preroll is not None:
                self._preroll.append(frame)
            return None

        self._seg.extend(frame)
        self._frames_in_seg += 1
        self._silence_run = 0 if is_speech else self._silence_run + 1

        hit_silence = self._silence_run >= self.silence_frames
        soft_cut = self._frames_in_seg >= self.soft_max_frames and not is_speech
        hit_max = self._frames_in_seg >= self.max_frames

        if hit_silence or soft_cut or hit_max:
            pcm = bytes(self._seg)
            reason = (
                "silence" if hit_silence else ("soft_gap" if soft_cut else "max_len")
            )
            start = self._seg_start_frame * self.frame_sec
            end = self._frames_consumed * self.frame_sec
            self._reset_segment()
            if len(pcm) >= self.min_bytes:
                log.debug(
                    "segment %.2f-%.2fs (%s, %d bytes)", start, end, reason, len(pcm)
                )
                return Segment(pcm, max(0.0, start), end, reason)
            log.debug("segment dropped (<min, %s): %d bytes", reason, len(pcm))
            return None
        return None

    def _reset_segment(self):
        self._seg = bytearray()
        self._triggered = False
        self._silence_run = 0
        self._frames_in_seg = 0

    def flush(self):
        """Force-finalize whatever speech is buffered (call on stop)."""
        if not self._triggered:
            return None
        pcm = bytes(self._seg)
        start = self._seg_start_frame * self.frame_sec
        end = self._frames_consumed * self.frame_sec
        self._reset_segment()
        if len(pcm) < self.min_bytes:
            return None
        return Segment(pcm, max(0.0, start), end, "flush")
