import { useCallback, useRef, useState } from "react";

// Streams 16 kHz mono Int16 PCM to the backend over a WebSocket, from one of
// two sources:
//
//   "mic"  — microphone via getUserMedia. For live, in-room speech.
//   "tab"  — a browser tab's audio via getDisplayMedia. For transcribing
//            PLAYED audio (YouTube news clips, recordings): the audio is
//            captured DIGITALLY, so speaker volume, room reverb and mic
//            quality don't exist in the path.
//
// v4: start(source, speakers) — `speakers` is the KNOWN speaker count for
// this session (0 = auto). It's sent as ?speakers=N on the WebSocket URL and
// hard-caps diarization identities on the backend: with speakers=2, phantom
// "Speaker 7" labels are impossible.
//
// An AudioWorklet resamples whatever rate the context actually runs at down
// to exactly 16 kHz (browsers often ignore the requested rate).
//
// STOP HANDSHAKE: the server finishes transcribing every queued segment
// before replying {status:"stopped"}; we keep the socket open until then.
const TARGET_SAMPLE_RATE = 16000;
const STOP_FLUSH_TIMEOUT_MS = 20000;
const LEVEL_UPDATE_MS = 100;

export function useAudioStream(wsUrl, onMessage) {
  const [recording, setRecording] = useState(false);
  const [status, setStatus] = useState("idle");
  const [level, setLevel] = useState(0); // input RMS 0..1

  const wsRef = useRef(null);
  const ctxRef = useRef(null);
  const streamRef = useRef(null);
  const nodeRef = useRef(null);
  const lastLevelAt = useRef(0);
  const stopRef = useRef(null); // so async callbacks can trigger stop()

  const start = useCallback(
    async (source = "mic", speakers = 0) => {
      setStatus("connecting");

      try {
        // 1) open the socket first. speakers>0 = the user KNOWS the count;
        // the backend caps diarization identities to exactly that many.
        const url =
          speakers > 0
            ? `${wsUrl}?speakers=${encodeURIComponent(speakers)}`
            : wsUrl;
        const ws = new WebSocket(url);
        ws.binaryType = "arraybuffer";
        wsRef.current = ws;
        ws.onmessage = (e) => {
          const data = JSON.parse(e.data);
          if (data.type === "status") {
            setStatus(data.state);
            // Server confirms every queued segment has been delivered — only
            // NOW is it safe to close without losing the final transcript.
            if (data.state === "stopped") {
              try {
                ws.close();
              } catch {}
            }
          }
          onMessage(data);
        };
        await new Promise((resolve, reject) => {
          ws.onopen = resolve;
          ws.onerror = () =>
            reject(
              new Error(
                "WebSocket failed to connect — is the backend running?",
              ),
            );
        });

        // 2) audio source -> AudioWorklet -> 16 kHz PCM
        let stream;
        if (source === "tab") {
          // Tab-audio capture. The browser will show a picker: the user must
          // choose a "Chrome Tab" (or "Edge tab") and tick "Also share tab
          // audio". Sharing a window/screen usually yields NO audio.
          const disp = await navigator.mediaDevices.getDisplayMedia({
            video: true, // required by the API even though we only want audio
            audio: {
              echoCancellation: false,
              noiseSuppression: false,
              autoGainControl: false,
            },
          });
          if (disp.getAudioTracks().length === 0) {
            disp.getTracks().forEach((t) => t.stop());
            throw new Error(
              "No tab audio was shared. In the picker, choose a browser TAB " +
                '(not a window or screen) and enable "Also share tab audio".',
            );
          }
          // We only need the audio; drop the video track immediately.
          disp.getVideoTracks().forEach((t) => t.stop());
          // If the user clicks the browser's "Stop sharing" bar, end cleanly.
          disp.getAudioTracks()[0].onended = () => stopRef.current?.();
          stream = disp;
        } else {
          // Microphone. echoCancellation/noiseSuppression/autoGainControl are
          // tuned for phone-call cleanup and HURT transcription fidelity.
          stream = await navigator.mediaDevices.getUserMedia({
            audio: {
              channelCount: 1,
              echoCancellation: false,
              noiseSuppression: false,
              autoGainControl: false,
            },
          });
        }
        streamRef.current = stream;

        // Request 16 kHz. The worklet resamples if the browser ignores us, so
        // correctness no longer hinges on this being honored.
        const ctx = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
        ctxRef.current = ctx;

        console.log(
          `[VoxLive] source=${source} speakers=${speakers || "auto"} ` +
            `AudioContext sampleRate = ${ctx.sampleRate}` +
            (ctx.sampleRate === TARGET_SAMPLE_RATE
              ? " (honored, passthrough)"
              : ` (NOT honored → worklet resampling ${ctx.sampleRate}→${TARGET_SAMPLE_RATE})`),
        );

        await ctx.audioWorklet.addModule("/audio-processor.js");

        const sourceNode = ctx.createMediaStreamSource(stream);
        const node = new AudioWorkletNode(ctx, "pcm-processor", {
          processorOptions: { targetSampleRate: TARGET_SAMPLE_RATE },
        });
        nodeRef.current = node;

        node.port.onmessage = (e) => {
          // The worklet sends one meta object on startup, then PCM buffers.
          if (e.data && e.data.type === "meta") {
            console.log("[VoxLive] worklet meta:", e.data);
            return;
          }

          // Input level (RMS) — cheap, throttled to avoid re-render spam.
          const now = performance.now();
          if (now - lastLevelAt.current > LEVEL_UPDATE_MS) {
            lastLevelAt.current = now;
            const pcm = new Int16Array(e.data);
            let sum = 0;
            for (let i = 0; i < pcm.length; i++) sum += pcm[i] * pcm[i];
            setLevel(Math.sqrt(sum / pcm.length) / 32768);
          }

          if (ws.readyState === WebSocket.OPEN) ws.send(e.data);
        };

        // route through a muted gain node so the worklet keeps pulling audio
        // without playing it back through the speakers.
        const mute = ctx.createGain();
        mute.gain.value = 0;
        sourceNode.connect(node);
        node.connect(mute);
        mute.connect(ctx.destination);

        setRecording(true);
      } catch (err) {
        // Never fail silently: permission denied, backend down, no tab audio —
        // all of it surfaces to the transcript UI as an error row.
        console.error("[VoxLive] start failed:", err);
        try {
          nodeRef.current?.disconnect();
        } catch {}
        try {
          streamRef.current?.getTracks().forEach((t) => t.stop());
        } catch {}
        try {
          ctxRef.current?.close();
        } catch {}
        try {
          wsRef.current?.close();
        } catch {}
        setRecording(false);
        setStatus("error");
        onMessage({
          type: "error",
          message: err?.message || "failed to start recording",
        });
      }
    },
    [wsUrl, onMessage],
  );

  const stop = useCallback(() => {
    // Stop capturing immediately…
    try {
      nodeRef.current?.disconnect();
    } catch {}
    try {
      streamRef.current?.getTracks().forEach((t) => t.stop());
    } catch {}
    try {
      ctxRef.current?.close();
    } catch {}
    setLevel(0);

    // …but keep the SOCKET open until the server says "stopped" (handled in
    // onmessage above). Fallback: force-close if the flush never completes.
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      try {
        ws.send("stop");
      } catch {}
      setTimeout(() => {
        if (ws.readyState === WebSocket.OPEN) {
          console.warn("[VoxLive] flush timed out — closing socket");
          ws.close();
        }
      }, STOP_FLUSH_TIMEOUT_MS);
    } else {
      try {
        ws?.close();
      } catch {}
      setStatus("idle");
    }

    // The button flips back to "Start" right away; the transcript keeps
    // receiving the flushed tail segments until "stopped" arrives.
    setRecording(false);
  }, []);

  stopRef.current = stop;

  return { start, stop, recording, status, level };
}
