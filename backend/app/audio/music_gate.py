"""
music_gate.py — a speech/music discriminator in front of the ASR call.

WHY THIS EXISTS (supervisor review, Fix #4)
-------------------------------------------
The pipeline had nothing that could reject a song:

    Sinhala song -> vocals -> WebRTC VAD says "speech" -> segment created
                 -> Gemini receives singing -> Gemini attempts transcription

WebRTC VAD answers "does this look like speech?", and sung vocals are speech
by every measure it uses. MIN_SEGMENT_RMS is an ENERGY gate, and music is
loud — often louder than speech. So the only thing actually deciding "this is
music, return empty" was a sentence in the Gemini prompt, which asks one
generative request to solve two different problems at once:

    1. Is this speech?
    2. If yes, what exactly was said?

This module answers question 1 separately, so the ASR call only ever has to
answer question 2.

WHAT IT IS AND IS NOT
---------------------
This is a classical signal-features discriminator in the Scheirer & Slaney
(1997) tradition, not a neural classifier. It was chosen because it needs no
model download, no GPU, no extra dependency beyond numpy, and runs in about a
millisecond on a three-second segment — so it can sit on the hot path without
reintroducing the latency problem that DiarizationService was written to fix.
It is also completely transparent: every feature it used is in the log line,
so when it gets something wrong you can see WHICH feature was wrong.

It is NOT a solved problem, and the review is explicit that the classifier
must be evaluated against real recordings rather than assumed to work. Hence
MUSIC_GATE_MODE defaults to `log`:

    off    no gate at all
    log    classify, log the score and the features, DROP NOTHING   <- default
    drop   classify and discard segments scoring above the threshold

Ship it in `log`. Record real Sinhala conversation and real Sinhala songs,
read the `music-gate:` lines, pick MUSIC_GATE_THRESHOLD from the actual
distribution you see, and only then set `drop`. A gate tuned by guesswork that
silently eats quiet Sinhala speech is worse than no gate at all — the failure
is invisible, whereas a song getting transcribed is obvious.

THE FEATURES
------------
Each is normalised to a 0..1 "sounds like music" opinion and then averaged.
All four are about STRUCTURE OVER TIME, which is what actually separates the
two classes — instantaneous spectra do not.

  low_energy_ratio   Fraction of frames well below the segment's mean energy.
                     Speech is full of gaps: stops, breaths, the pause between
                     words. Music, especially with instrumental backing, runs
                     continuously. FEW quiet frames -> music.

  mod4hz             Share of the energy envelope's spectrum in the 3-6 Hz
                     band. That is the syllabic rate — the defining rhythm of
                     speech in every language. Music's envelope energy sits
                     lower (beat, 1-3 Hz) and is spread more evenly. WEAK 4 Hz
                     peak -> music.

  zcr_std            Variability of the zero-crossing rate. Speech alternates
                     constantly between voiced (low ZCR) and unvoiced
                     fricatives like /s/ and /ʃ/ (high ZCR). Sustained sung
                     vowels and instruments do not. STEADY ZCR -> music.

  flux_cv            Coefficient of variation of spectral flux. Speech is a
                     sequence of onsets and decays; music holds a much steadier
                     spectral shape between beats. STEADY SPECTRUM -> music.

WHERE IT IS WEAKEST — read this before trusting a number
  * A cappella singing has speech-like ZCR and can score low (i.e. "speech").
  * Rap is rhythmically speech-like almost by definition.
  * Very short segments (< ~1.5 s) do not contain enough envelope periods to
    measure a 4 Hz modulation at all, so `analyse()` reports low confidence
    and `is_music()` refuses to judge them.
  * Speech over background music is genuinely ambiguous and this returns a
    middling score, which is the honest answer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("voxlive.musicgate")

FRAME_SEC = 0.025
HOP_SEC = 0.010

# A segment shorter than this cannot show a 4 Hz envelope modulation — there
# are simply not enough syllable periods in it — so the gate abstains rather
# than guessing from one feature.
MIN_ANALYSABLE_SEC = 1.5

# Feature normalisation points, expressed as (speech_like, music_like). A raw
# value at `speech_like` scores 0.0, at `music_like` scores 1.0, linear in
# between and clipped outside. These are STARTING POINTS from the literature
# and from the geometry of the features, not measurements of your recordings.
# Re-derive them from your own `music-gate:` logs.
NORM = {
    "low_energy_ratio": (0.22, 0.04),  # speech has many quiet frames
    "mod4hz": (0.34, 0.10),  # speech peaks at the syllabic rate
    "zcr_std": (0.055, 0.012),  # speech alternates voiced/unvoiced
    "flux_cv": (0.95, 0.35),  # speech onsets are bursty
}

# Equal weights: with no labelled Sinhala data to fit on, inventing weights
# would be pretending to a precision this does not have. Fit them once you
# have logged real songs and real speech.
WEIGHTS = {k: 0.25 for k in NORM}


@dataclass
class GateResult:
    """One verdict, with everything needed to argue with it."""

    score: float  # 0 = clearly speech, 1 = clearly music
    confident: bool  # False when the segment is too short to judge
    features: dict  # raw feature values, for tuning
    duration: float

    @property
    def label(self) -> str:
        if not self.confident:
            return "unknown"
        return "music" if self.score >= 0.5 else "speech"

    def as_log(self) -> str:
        feats = " ".join(f"{k}={v:.3f}" for k, v in sorted(self.features.items()))
        return (
            f"score={self.score:.3f} label={self.label} "
            f"dur={self.duration:.2f}s {feats}"
        )


def _frames(x: np.ndarray, sr: int) -> np.ndarray:
    n = max(1, int(sr * FRAME_SEC))
    h = max(1, int(sr * HOP_SEC))
    if len(x) < n:
        return np.empty((0, n), dtype=np.float32)
    count = 1 + (len(x) - n) // h
    idx = np.arange(n)[None, :] + h * np.arange(count)[:, None]
    return x[idx]


def _norm(name: str, value: float) -> float:
    """Map a raw feature onto 0 (speech-like) .. 1 (music-like)."""
    speechy, musicy = NORM[name]
    if speechy == musicy:
        return 0.5
    t = (value - speechy) / (musicy - speechy)
    return float(np.clip(t, 0.0, 1.0))


def analyse(pcm: bytes, sample_rate: int) -> GateResult:
    """Score one segment of int16 PCM. Never raises; never blocks."""
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    duration = len(x) / float(sample_rate) if sample_rate else 0.0

    if len(x) < int(sample_rate * 0.2):
        return GateResult(0.0, False, {}, duration)

    fr = _frames(x, sample_rate)
    if len(fr) < 8:
        return GateResult(0.0, False, {}, duration)

    win = np.hanning(fr.shape[1]).astype(np.float32)
    fw = fr * win

    # ---- energy envelope -------------------------------------------------
    energy = np.sqrt(np.mean(fr * fr, axis=1) + 1e-12)
    mean_e = float(energy.mean())
    low_energy_ratio = float(np.mean(energy < 0.5 * mean_e)) if mean_e > 1e-9 else 0.0

    # ---- 4 Hz modulation -------------------------------------------------
    # Spectrum OF THE ENVELOPE. Envelope sample rate is 1/HOP_SEC = 100 Hz.
    env = energy - energy.mean()
    env_sr = 1.0 / HOP_SEC
    spec = np.abs(np.fft.rfft(env * np.hanning(len(env))))
    freqs = np.fft.rfftfreq(len(env), d=1.0 / env_sr)
    band = (freqs >= 3.0) & (freqs <= 6.0)
    ref = (freqs >= 0.5) & (freqs <= 20.0)
    total = float(spec[ref].sum())
    mod4hz = float(spec[band].sum() / total) if total > 1e-9 else 0.0

    # ---- zero-crossing rate ---------------------------------------------
    zcr = np.mean(np.abs(np.diff(np.sign(fr), axis=1)) > 0, axis=1)
    zcr_std = float(zcr.std())

    # ---- spectral flux ---------------------------------------------------
    mag = np.abs(np.fft.rfft(fw, axis=1))
    mag /= np.maximum(mag.sum(axis=1, keepdims=True), 1e-9)
    flux = np.sqrt(np.sum(np.diff(mag, axis=0) ** 2, axis=1))
    flux_mean = float(flux.mean())
    flux_cv = float(flux.std() / flux_mean) if flux_mean > 1e-9 else 0.0

    features = {
        "low_energy_ratio": low_energy_ratio,
        "mod4hz": mod4hz,
        "zcr_std": zcr_std,
        "flux_cv": flux_cv,
    }
    score = float(sum(WEIGHTS[k] * _norm(k, v) for k, v in features.items()))

    return GateResult(
        score=score,
        confident=duration >= MIN_ANALYSABLE_SEC,
        features=features,
        duration=duration,
    )


def is_music(pcm: bytes, sample_rate: int, threshold: float) -> tuple[bool, GateResult]:
    """`(should_discard, result)`.

    Abstains — returns False — on segments too short to analyse. Discarding
    something the gate cannot actually judge would silently delete short
    utterances ("aeh", "naeh", a one-word answer), and losing real words is a
    far worse failure than transcribing a few seconds of a song.
    """
    result = analyse(pcm, sample_rate)
    if not result.confident:
        return False, result
    return result.score >= threshold, result
