# 🎙️ VoxLive — Real-time Live Transcription & Speaker Diarization

VoxLive is a full-stack web application designed for real-time audio capture, live transcription using Gemini, and low-latency speaker diarization powered by PyAnnote. It streamingly transcribes incoming microphone input, automatically segments audio using WebRTC VAD, maps speech to speakers, and updates transcripts dynamically on screen.

---

## 🌟 Key Features

*   **Real-time Audio Capture:** Captured directly from the browser using standard Web APIs, resampled at 16kHz via an `AudioWorklet`.
*   **Gemini-Powered Transcription:** Integrates with Gemini (supporting AI Studio API key and Vertex AI Service Account modes) to transcribe voice segments with low latency.
*   **Parallel Execution Pipeline:** Diarization (speaker identification) and Gemini ASR run in parallel to minimize latency.
*   **Anti-Hallucination Guards:** Built-in mechanisms to block model hallucinations on quiet segments and noise (RMS energy gate, word density gate, context-echo guard, VAD aggressiveness tuning).
*   **Smart Live Upserting:** A custom React transcript manager that updates live text chunks in-place, eliminating duplicate transcript entries.
*   **Centroid-Drift Mitigation:** Stores individual speaker embeddings to prevent speaker identification drift and ensure accurate diarization over long sessions.

---

## 📁 Repository Structure

```
voxlive-gemini/
├── backend/
│   ├── app/
│   │   ├── main.py             # FastAPI WebSocket router & streaming server
│   │   ├── config.py           # Application configurations (loads from .env)
│   │   ├── diarization.py      # Speaker identification and tracking
│   │   ├── audio.py            # Audio utility functions
│   │   ├── schemas.py          # Pydantic data schemas
│   │   ├── postprocess.py      # Output formatting and cleansing
│   │   └── providers/
│   │       └── gemini_provider.py # Gemini translation & ASR provider
│   ├── requirements.txt        # Backend dependencies
│   └── .env.example            # Backend env configurations reference
│
└── frontend/
    ├── src/
    │   ├── App.jsx             # React application entry with transcript hooks
    │   ├── main.jsx            # Main React bootstrapper
    │   ├── index.css           # Global stylesheet with Tailwind imports
    │   ├── components/
    │   │   └── TranscriptView.jsx # Renders live-growing chat/transcript UI
    │   └── hooks/
    │       └── useAudioStream.js # Hook to capture mic data & send via WS
    ├── public/
    │   └── audio-processor.js  # AudioWorklet processor for resampling
    ├── package.json            # Frontend package configurations
    ├── vite.config.js          # Vite config
    └── .env.example            # Frontend env configurations reference
```

---

## 🚀 Getting Started

### Prerequisites
*   **Python:** Version 3.10 or newer.
*   **Node.js:** Version 18.0 or newer.
*   **Google Cloud Platform / Gemini API Account:** GCP Project with Vertex AI enabled, or a Google AI Studio API Key.

---

### 🐍 Backend Setup

1.  **Navigate to backend directory**:
    ```bash
    cd backend
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python -m venv .venv
    # Windows Command Prompt:
    .venv\Scripts\activate.bat
    # PowerShell:
    .venv\Scripts\Activate.ps1
    # macOS/Linux:
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    > [!IMPORTANT]
    > If using `DIARIZATION_MODE=pyannote` or `identify`, make sure to accept PyAnnote's model conditions once at [hf.co/pyannote/embedding](https://hf.co/pyannote/embedding) (requires logging into Hugging Face) and obtain an API token from your Hugging Face settings.

4.  **Configure environment variables**:
    Copy `backend/.env.example` to `backend/.env` and update the settings.
    ```bash
    cp .env.example .env
    ```
    Configure either:
    *   **API Key Mode**: Fill in `GEMINI_API_KEY`.
    *   **Vertex AI Mode**: Set `GEMINI_USE_VERTEX=true`, and fill out `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, and `GOOGLE_APPLICATION_CREDENTIALS` (absolute path to your GCP service account JSON key).

5.  **Run the server**:
    ```bash
    uvicorn app.main:app --reload
    ```
    The backend will start on `http://localhost:8000`.

---

### 💻 Frontend Setup

1.  **Navigate to frontend directory**:
    ```bash
    cd frontend
    ```

2.  **Install dependencies**:
    ```bash
    npm install
    ```

3.  **Configure environment variables** *(optional)*:
    Copy `frontend/.env.example` to `frontend/.env`.
    ```bash
    cp .env.example .env
    ```
    By default, it targets `ws://localhost:8000`. Adjust `VITE_WS_URL` if your backend is hosted elsewhere.

4.  **Start development server**:
    ```bash
    npm run dev
    ```
    Open `http://localhost:5173` in your browser.

---

## ⚙️ Backend Environment Variables Reference

Key parameters in `backend/.env` to configure:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `GEMINI_API_KEY` | *(None)* | API key from Google AI Studio. |
| `GEMINI_USE_VERTEX` | `false` | Set to `true` to authenticate via a Google Cloud Service Account instead of an API key. |
| `GEMINI_MODEL` | `gemini-2.5-flash` | The Gemini model to use (`gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-3.5-flash`). |
| `DIARIZATION_MODE` | `pyannote` | Mode for speaker recognition: `off`, `pyannote` (standard clustering), or `identify` (voiceprint match). |
| `DIARIZATION_THRESHOLD` | `0.45` | Cosine distance threshold for speaker clustering. Lower is more restrictive. |
| `HUGGINGFACE_TOKEN` | *(None)* | Hugging Face token required to download the gating models for speaker diarization. |
| `MIN_SEGMENT_RMS` | `120` | RMS gate value (0 to 32767). Drops segments quieter than this to prevent hallucination. |
| `VAD_AGGRESSIVENESS` | `2` | WebRTC VAD sensitivity (0=least aggressive, 3=most aggressive). |
| `SOFT_MAX_SEGMENT_MS` | `6000` | Target maximum audio chunk length for streaming segments. |

---

## 🎙️ Tuning Speaker Diarization

Diarization accuracy depends heavily on your microphone quality, room acoustics, and speaking volume. The default threshold (`0.45`) is a starting point, and you **must** tune it to avoid issue profiles like:
*   *Everyone is Speaker 1* (threshold is too loose)
*   *One person splits into multiple speakers* (threshold is too strict)

### How to Tune:
1.  Set up your mic and perform a test recording involving 2 people taking turns with brief pauses.
2.  Watch the backend console logs for `diarize:` diagnostic lines which report cosine distance:
    ```text
    diarize: Speaker Speaker-1 match! dist=0.312 (threshold=0.45)
    diarize: New speaker Speaker-2 created (dist=0.612 > threshold=0.45)
    ```
3.  Examine the logs to extract:
    *   **Same-speaker distance band**: The typical distance value when the *same* speaker continues talking (e.g., `0.20 - 0.38`).
    *   **Cross-speaker distance band**: The typical distance value when a *different* speaker begins talking (e.g., `0.52 - 0.65`).
4.  Set your `DIARIZATION_THRESHOLD` in `.env` to the midpoint of the gap between the two bands (e.g., `0.45`).
5.  Restart your backend and run another test to confirm.

---

## 🛡️ Hallucination Prevention Guards

Large Language Models (LLMs) like Gemini can sometimes invent text ("hallucinate") when fed audio chunks that contain only quiet breathing, background hiss, or echo. VoxLive implements 4 layers of defense:

1.  **WebRTC VAD Aggressiveness (`VAD_AGGRESSIVENESS`)**: Filters out non-speech elements before sending to the transcription queue.
2.  **Energy Gate (`MIN_SEGMENT_RMS`)**: Drops audio frames under a minimum root-mean-square amplitude threshold (default `120`). Quiet frames are never passed to Gemini.
3.  **Density Gate (`MAX_WORDS_PER_SEC`)**: Rejects any output transcripts that contain more than `8.0` words per second of audio—a clear sign that the model has invented text.
4.  **Context-Echo Guard**: Suppresses Gemini from repeating previous sentences when analyzing subsequent short pauses.

---

## 🔗 Technical Document Links

For detailed inspects, see:
*   **Backend Main**: [main.py](file:///c:/Users/dulha/Downloads/GitHub/voxlive-gemini/backend/app/main.py)
*   **Diarization Engine**: [diarization.py](file:///c:/Users/dulha/Downloads/GitHub/voxlive-gemini/backend/app/diarization.py)
*   **Configuration Manager**: [config.py](file:///c:/Users/dulha/Downloads/GitHub/voxlive-gemini/backend/app/config.py)
*   **Frontend Entry**: [App.jsx](file:///c:/Users/dulha/Downloads/GitHub/voxlive-gemini/frontend/src/App.jsx)
*   **Transcript UI Component**: [TranscriptView.jsx](file:///c:/Users/dulha/Downloads/GitHub/voxlive-gemini/frontend/src/components/TranscriptView.jsx)