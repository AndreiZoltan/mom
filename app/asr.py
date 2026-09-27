# app/asr.py
import asyncio
import gc
import os
from pathlib import Path
from typing import Optional

from app.config import DEVICE, MOCK_MODE, WHISPER_MODEL_PATH
from app.prompts import *
from app.schemas import MeetingType

# Optional config-based fallback for LoRA path
try:
    from app.config import WHISPER_LORA_PATH
except ImportError:
    WHISPER_LORA_PATH = os.getenv("WHISPER_LORA_PATH", None)


def _clean_gpu():
    try:
        import torch

        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _transcribe_with_peft(audio_path: str, lora_path: str, initial_prompt: str) -> dict:
    """Fallback runner for HuggingFace PEFT LoRA adapters."""
    import torch
    import torchaudio
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    print(f"[ASR] Loading Hugging Face Whisper with PEFT adapter: {lora_path}")
    base_model_name = "openai/whisper-medium"

    processor = WhisperProcessor.from_pretrained(base_model_name, language="Romanian", task="transcribe")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map=DEVICE,
    )
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()

    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    # Process mono audio in chunks
    waveform_np = waveform.squeeze().numpy()
    inputs = processor(waveform_np, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(DEVICE)
    if DEVICE == "cuda":
        input_features = input_features.half()

    # Prompt conditioning
    prompt_ids = processor.get_prompt_ids(initial_prompt) if initial_prompt else None

    with torch.no_grad():
        predicted_ids = model.generate(input_features, prompt_ids=prompt_ids)

    transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

    del model, base_model, processor
    _clean_gpu()

    return {"transcript": transcript, "language_detected": "ro"}


def _sync_transcribe_audio(
    audio_path: str, meeting_type: str, lora_path: Optional[str] = None
) -> dict:
    """Runs ASR and unloads model immediately to save VRAM."""
    if MOCK_MODE:
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

    # Prompt selection according to meeting type
    if meeting_type == MeetingType.MEDICAL or meeting_type == MeetingType.MEDICAL.value:
        initial_prompt = initial_medical_prompt
    elif meeting_type == MeetingType.EXECUTIVE or meeting_type == MeetingType.EXECUTIVE.value:
        initial_prompt = initial_executive_prompt
    else:
        initial_prompt = initial_administrative_prompt

    audio_file = Path(audio_path)
    if not audio_file.exists():
        print(f"[ASR] Error: Audio file does not exist: {audio_path}")
        return {"transcript": "", "language_detected": "unknown"}

    active_lora = lora_path or WHISPER_LORA_PATH

    # Case A: PEFT HuggingFace LoRA directory detected
    if active_lora and (Path(active_lora) / "adapter_config.json").exists():
        return _transcribe_with_peft(audio_path, active_lora, initial_prompt)

    # Case B: Standard faster-whisper (CTranslate2 format)
    from faster_whisper import WhisperModel

    model_target = active_lora if (active_lora and Path(active_lora).exists()) else WHISPER_MODEL_PATH
    if active_lora and Path(active_lora).exists():
        print(f"[ASR] Loading converted CTranslate2 model from: {model_target}")
    else:
        print("[ASR] Running standard faster-whisper model without LoRA.")

    model = WhisperModel(
        model_target,
        device=DEVICE,
        compute_type="int8_float16" if DEVICE == "cuda" else "int8",
    )

    segments, info = model.transcribe(
        audio_path,
        language="ro",
        initial_prompt=initial_prompt,
    )

    transcript = " ".join([seg.text for seg in segments]).strip()

    del model
    _clean_gpu()

    return {"transcript": transcript, "language_detected": info.language}


async def transcribe_audio(
    audio_path: str, meeting_type: str, lora_path: Optional[str] = None
) -> dict:
    return await asyncio.to_thread(
        _sync_transcribe_audio, audio_path, meeting_type, lora_path
    )