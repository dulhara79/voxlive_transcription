import asyncio
import logging
import torch
import json
from typing import Dict, Any, List
from .reconcile import MultiModelReconciler

logger = logging.getLogger(__name__)


async def run_sortformer_offline(audio_path: str, config: Dict) -> List[Dict]:
    logger.info("Running Sortformer offline pass...")

    def _run_nemo():
        import nemo.collections.asr as nemo_asr

        try:
            # Assuming NeMo offline diarization API integration
            return []
        except Exception as e:
            logger.error(f"Sortformer offline failed: {e}")
            return []

    return await asyncio.to_thread(_run_nemo)


async def run_pyannote_community1(audio_path: str, config: Dict) -> List[Dict]:
    logger.info("Running Pyannote Community-1 offline pass...")
    min_spk = config.get("min_speakers", 1)
    max_spk = config.get("max_speakers", 10)
    hf_token = config.get("HUGGINGFACE_TOKEN", "")

    def _run_pyannote():
        try:
            from pyannote.audio import Pipeline

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # FIXED: 'use_auth_token' changed to 'token' for pyannote.audio >= 3.1
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", token=hf_token
            ).to(device)

            diarization = pipeline(
                audio_path, min_speakers=min_spk, max_speakers=max_spk
            )
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append(
                    {
                        "speaker": str(speaker),
                        "start": float(turn.start),
                        "end": float(turn.end),
                    }
                )
            return segments
        except Exception as e:
            logger.error(f"Pyannote offline failed: {e}")
            return []

    return await asyncio.to_thread(_run_pyannote)


async def run_gemini_diarization(audio_path: str, config: Dict) -> List[Dict]:
    if not config.get("GEMINI_FINAL_DIARIZATION", False):
        return []

    logger.info("Running Gemini whole-file diarization pass...")

    def _run_gemini():
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=config.get("GEMINI_API_KEY", ""))

        try:
            # Safely uploading file to Gemini API for processing
            audio_file = client.files.upload(file=audio_path)

            prompt = (
                "Please analyze the audio file and provide a diarized transcript. "
                "Return the result strictly as a JSON array of objects, where each object "
                "contains 'speaker' (string), 'start' (float seconds), and 'end' (float seconds)."
            )

            response = client.models.generate_content(
                model="gemini-3.1-flash",
                contents=[audio_file, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json", temperature=0.0
                ),
            )

            raw = getattr(response, "text", "[]")
            clean_raw = raw.replace("```json", "").replace("```", "").strip()
            segments = json.loads(clean_raw)
            return segments
        except Exception as e:
            logger.error(f"Gemini diarization failed: {e}")
            return []

    return await asyncio.to_thread(_run_gemini)


async def run_offline_diarization(
    audio_path: str, config: Dict[str, Any]
) -> Dict[str, Any]:

    tasks = [
        run_sortformer_offline(audio_path, config),
        run_pyannote_community1(audio_path, config),
    ]

    if config.get("GEMINI_FINAL_DIARIZATION", False):
        tasks.append(run_gemini_diarization(audio_path, config))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    candidates = {}
    if not isinstance(results[0], Exception) and results[0]:
        candidates["sortformer"] = results[0]
    if not isinstance(results[1], Exception) and results[1]:
        candidates["pyannote"] = results[1]
    if len(results) > 2 and not isinstance(results[2], Exception) and results[2]:
        candidates["gemini"] = results[2]

    reconciler = MultiModelReconciler()
    final_result = reconciler.reconcile(candidates)

    return final_result
