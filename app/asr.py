import asyncio
import gc
from app.config import MOCK_MODE, DEVICE, WHISPER_MODEL_PATH
from app.schemas import MeetingType
from app.prompts import *

from pathlib import Path


def _clean_gpu():
    try:
        import torch

        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
    except ImportError:
        pass


# async def transcribe_audio(audio_path: str, meeting_type: str) -> dict:

def _sync_transcribe_audio(audio_path: str, meeting_type: str) -> dict:
    """Runs ASR and unloads model immediately to save VRAM."""
    if MOCK_MODE:
        # Simulate Whisper processing time
        # await asyncio.sleep(3.5)
        return {
            "transcript": (
                "[00:00:10] Dr. Cebotari: Bună dimineața colegi. Discutăm astăzi revizuirea "
                "protocolului de antibioterapie perioperatorie și necesarul de kituri chirurgicale. "
                "[00:00:35] Elena Morari: По остаткам антибиотиков резерва: лимит на этот месяц превышен, "
                "нужно утвердить дополнительный заказ до пятницы. "
                "[00:01:05] Dr. Rusu: We must update the emergency triage SOP by next week."
            ),
            "language_detected": "multilingual (ro/ru/en)",
        }

    # Real Inference (faster-whisper)
    from faster_whisper import WhisperModel

    # 1. Load Whisper into GPU
    model = WhisperModel(
        WHISPER_MODEL_PATH,
        device=DEVICE,
        compute_type="int8_float16" if DEVICE == "cuda" else "int8",
    )

    if meeting_type == MeetingType.MEDICAL:
        initial_prompt = initial_medical_prompt
    elif meeting_type == MeetingType.EXECUTIVE:
        initial_prompt = initial_executive_prompt
    else:
        initial_prompt = initial_administrative_prompt



    a = Path(audio_path)
    if not a.exists():
        print("PALUNDRA")
    segments, info = model.transcribe(
        audio_path,
        language="ro",
        # beam_size=3,
        # vad_filter=True,
        initial_prompt=initial_prompt,
    )

    transcript = " ".join([seg.text for seg in segments])
    # print(transcript)


    # 2. Release VRAM for the LLM
    del model
    _clean_gpu()

    return {"transcript": transcript, "language_detected": info.language}

async def transcribe_audio(audio_path: str, meeting_type: str) -> dict:
    return await asyncio.to_thread(_sync_transcribe_audio, audio_path, meeting_type)