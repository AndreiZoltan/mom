import asyncio
from pathlib import Path
import app_ollama.config as cfg


def _sync_transcribe_audio(audio_path: str, meeting_type: str, device: str = None) -> dict:
    """Runs ASR using faster-whisper with proper device and compute_type resolution."""
    if cfg.MOCK_MODE:
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

    from faster_whisper import WhisperModel

    # 1. Resolve target device
    active_device = device or cfg.DEVICE

    # 2. Pick appropriate compute type
    if "cuda" in active_device.lower():
        # "auto" lets CTranslate2 pick float16 or int8_float16 depending on GPU capability
        compute_type = "auto"
    else:
        compute_type = "int8"

    print(f"--> [ASR] Loading WhisperModel(device='{active_device}', compute_type='{compute_type}')")

    model = WhisperModel(
        cfg.WHISPER_MODEL_PATH,
        device=active_device,
        compute_type=compute_type,
    )

    initial_prompt = (
        "Ședință de consiliu medical și comitet director la spital. "
        "Discutăm despre KPIs, budget, audit și SOP-uri pentru triaj conform noilor clinical guidelines. "
        "Dr. Cebotari: Colegi, avem pe ordinea de zi situația din blocul chirurgical și secția ATI. "
        "Елена: По антибиотикам резерва нужно срочно закрыть отчёт за прошлый квартал. "
        "Revizuim graficul de gărzi, workflow-ul pacienților și achizițiile de consumabile."
    )

    segments, info = model.transcribe(
        audio_path,
        language="ro",
        initial_prompt=initial_prompt,
    )

    transcript = " ".join([seg.text for seg in segments])
    del model

    return {"transcript": transcript, "language_detected": info.language}


async def transcribe_audio(audio_path: str, meeting_type: str, device: str = None) -> dict:
    return await asyncio.to_thread(_sync_transcribe_audio, audio_path, meeting_type, device)