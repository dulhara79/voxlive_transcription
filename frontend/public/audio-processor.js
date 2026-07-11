// Runs on the audio render thread.
//
// CRITICAL FIX: never trust that the AudioContext is actually at 16 kHz.
// Browsers frequently ignore `new AudioContext({ sampleRate: 16000 })` and run
// at the device-native rate (usually 48000, sometimes 44100). If we ship those
// samples to a backend that labels them 16000, every utterance is decoded ~3x
// too slow and Whisper/gpt-4o returns garbage in ALL languages.
//
// So this worklet ALWAYS resamples from the real context rate (`sampleRate`,
// a global in AudioWorkletGlobalScope) down to a fixed target (default 16000)
// using a continuous streaming linear resampler, then emits 16-bit PCM.
//
// When the context genuinely is 16 kHz, the ratio is 1.0 and this is an exact
// passthrough (no quality loss). When it isn't, we resample correctly instead
// of silently corrupting the stream.
class PCMProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    this.targetRate = opts.targetSampleRate || 16000;

    // input samples consumed per output sample (>1 when downsampling)
    this.step = sampleRate / this.targetRate;

    // Streaming resampler state (carried across 128-sample render quanta so
    // there are no gaps or duplicated samples at block boundaries).
    this.virtualIndex = 0; // fractional read position in the continuous input stream
    this.inputCount = 0;   // global index of the first sample of the current block
    this.prevLast = 0;     // last sample of the previous block (for interpolation continuity)

    // One-time report so the main thread can log the true rate it ended up with.
    this.port.postMessage({
      type: "meta",
      contextSampleRate: sampleRate,
      targetSampleRate: this.targetRate,
      resampling: Math.abs(this.step - 1) > 1e-6,
    });
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch || ch.length === 0) return true;

    const base = this.inputCount;     // global index of ch[0]
    const end = base + ch.length;     // one past the last global index in this block

    // Collect resampled float samples for this block.
    const out = [];
    // We can produce an output sample at global position `gi` only while its
    // right neighbour ch[localI0 + 1] exists, i.e. gi < end - 1.
    while (this.virtualIndex < end - 1) {
      const gi = this.virtualIndex;
      const i0 = Math.floor(gi);
      const frac = gi - i0;
      const localI0 = i0 - base; // can be -1 → use prevLast for continuity

      const s0 = localI0 < 0 ? this.prevLast : ch[localI0];
      const s1 = ch[localI0 + 1];
      out.push(s0 + (s1 - s0) * frac);

      this.virtualIndex += this.step;
    }

    this.inputCount = end;
    this.prevLast = ch[ch.length - 1];

    if (out.length === 0) return true;

    // Float (-1..1) → signed 16-bit little-endian PCM.
    const pcm = new Int16Array(out.length);
    for (let i = 0; i < out.length; i++) {
      let s = out[i];
      s = s < -1 ? -1 : s > 1 ? 1 : s;
      pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }

    // zero-copy transfer to the main thread
    this.port.postMessage(pcm.buffer, [pcm.buffer]);
    return true;
  }
}

registerProcessor("pcm-processor", PCMProcessor);
