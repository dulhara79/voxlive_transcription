"""End-to-end plumbing test: DiarizationService + TranscriptStore.

webrtcvad and torch aren't needed — both are stubbed. What is under test is
the part that has no model in it and is easy to get subtly wrong: buffer seam
handling, the absolute clock, coverage waiting, and the join between text and
speaker timeline.
"""

import sys, os, asyncio, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

import app.speaker_engine as se
import app.embedder as emb_mod
from app.transcript import TranscriptStore
from test_speaker_engine import voice, utterance

SR = 16000
RNG = np.random.default_rng(3)

# Ground truth: who speaks when (absolute seconds).
SCRIPT = [(0.0, 4.0, 0), (4.6, 9.0, 1), (9.5, 13.0, 0), (13.4, 17.0, 1)]
VOICES = [voice(31), voice(32)]


def who(t: float):
    for a, b, s in SCRIPT:
        if a <= t < b:
            return s
    return None


# ---- stub webrtcvad-backed speech detection: speech == inside the script ----
def fake_speech_regions(pcm, sample_rate, offset=0.0, aggressiveness=2, **kw):
    dur = len(pcm) / sample_rate
    out, cur = [], None
    step = 0.03
    t = 0.0
    while t < dur:
        s = who(offset + t)
        if s is not None:
            if cur is None:
                cur = [offset + t, offset + t + step]
            else:
                cur[1] = offset + t + step
        elif cur is not None:
            out.append(tuple(cur)); cur = None
        t += step
    if cur is not None:
        out.append(tuple(cur))
    return [r for r in out if r[1] - r[0] >= 0.30]


# ---- stub embedder: a window's vector reflects who was speaking mid-window --
class FakeEmbedder:
    def __init__(self, *a, **k):
        self.calls = 0
        self.windows = 0

    def embed_batch(self, waves):
        self.calls += 1
        self.windows += len(waves)
        out = []
        for w in waves:
            # The stub encodes the speaker in the waveform's DC offset, which
            # slice_windows carried through from the synthesised audio.
            spk = int(round(float(np.mean(w)) * 1000))
            spk = 0 if spk not in (0, 1) else spk
            jitter = 0.75 if len(w) >= 1.4 * SR else 1.15
            out.append(utterance(VOICES[spk], jitter))
        return out

    def reset(self):
        pass


def synth_audio(total_sec=17.5):
    """PCM whose DC offset encodes the speaker — lets the stub embedder know
    who was talking without needing a real model."""
    n = int(total_sec * SR)
    pcm = (RNG.normal(scale=0.01, size=n)).astype(np.float32)
    for a, b, s in SCRIPT:
        i0, i1 = int(a * SR), int(b * SR)
        pcm[i0:i1] += s / 1000.0
    return (np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes()


async def main():
    se.speech_regions = fake_speech_regions
    import app.diarization_service as ds
    ds.speech_regions = fake_speech_regions
    ds.Embedder = FakeEmbedder

    svc = ds.DiarizationService(
        hf_token="x", sample_rate=SR, expected_speakers=2, interval_sec=1.0
    )
    changes = []
    async def on_change():
        changes.append(svc.engine.speaker_count())
    svc.on_change = on_change
    svc.start()

    # Stream the audio in 100 ms chunks, in real order, like the WebSocket does.
    audio = synth_audio()
    step = int(0.1 * SR) * 2
    for i in range(0, len(audio), step):
        svc.feed(audio[i : i + step])
        await asyncio.sleep(0)          # let the background loop breathe
        if (i // step) % 12 == 0:
            await asyncio.sleep(0.02)

    await svc.finalize()
    await svc.aclose()

    tl = svc.engine.timeline()
    print(f"  embedder batches: {svc.embedder.calls}  windows: {svc.embedder.windows}")
    print(f"  timeline runs: {len(tl)}  speakers: {svc.engine.speaker_count()}")
    for a, b, s in tl:
        print(f"    {a:5.2f}-{b:5.2f}s  Speaker {s + 1}")

    assert svc.engine.speaker_count() == 2, svc.engine.speaker_count()

    # --- the real check: does the timeline agree with the script? -----------
    ok = tot = 0
    for t in np.arange(0.25, 17.0, 0.25):
        truth = who(t)
        if truth is None:
            continue
        got = svc.engine.label_for(t, t + 0.25)
        tot += 1
        ok += int(got == truth or (got is not None and _consistent(svc, truth, got)))
    print(f"\n  frame accuracy vs script: {ok}/{tot} = {ok / tot:.1%}")
    assert ok / tot > 0.90, f"only {ok / tot:.1%}"

    # --- join text to speakers, the way main.py does ------------------------
    store = TranscriptStore()
    for i, (a, b, s) in enumerate(SCRIPT, start=1):
        c = store.new_chunk(seg_id=i, start=a, end=b)
        c.text, c.language = f"turn{i}", "si"
    store.relabel(svc.label_for)
    paras = store.paragraphs()
    print(f"  paragraphs after join: {len(paras)}")
    for p in paras:
        print(f"    {p['speaker']:10} [{p['start']:.1f}-{p['end']:.1f}] {p['text']}")
    assert len(paras) == 4, "four alternating turns must stay four paragraphs"
    assert paras[0]["speaker"] == paras[2]["speaker"]
    assert paras[1]["speaker"] == paras[3]["speaker"]
    assert paras[0]["speaker"] != paras[1]["speaker"]
    print("\n  A-B-A-B turn structure preserved through the join  OK")


_MAP = {}
def _consistent(svc, truth, got):
    """Cluster ids are arbitrary; accept any consistent bijection."""
    if truth in _MAP:
        return _MAP[truth] == got
    if got in _MAP.values():
        return False
    _MAP[truth] = got
    return True


if __name__ == "__main__":
    print("pipeline integration test\n")
    asyncio.run(main())
    print("\nall passed")
