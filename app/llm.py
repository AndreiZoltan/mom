# app/llm.py
import asyncio
import gc
import json
from app.config import LLM_MODEL_PATH, MOCK_MODE
from app.prompts import (
    administrative_system_prompt,
    executive_system_prompt,
    medical_system_prompt,
)
from app.schemas import MeetingType


def _clean_gpu():
    try:
        import torch

        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
    except ImportError:
        pass


def MOCK_DICT(meeting_type: str) -> dict:
    if meeting_type == MeetingType.MEDICAL.value:
        return {
            "summary": "Raport de gardă terapie intensivă (vizită la pat). Analiza pacienților 1, 2 și 3.",
            "patients": [
                {
                    "patient_id": 1,
                    "patient_summary": "Pacientul este în stare generală lucidă. Valorile de laborator indică creatinina 240 și ureea 19. Pentru monitorizare a fost montată o linie arterială. Pacientul continuă tratamentul cu noradrenalină.",
                    "patient_decision": "Se decide continuarea infuziilor cu noradrenalină, continuarea umplerii vasculare și monitorizarea în continuare a pacientului.",
                },
                {
                    "patient_id": 2,
                    "patient_summary": "Pacientul este afebril, prezentând insuficiență respiratorie. Oxigenarea este de 90%, iar pacientul este hipercapnic, cu valoarea CO₂ de 69. În prezent urmează tratament cu Forxiga și Diacarb.",
                    "patient_decision": "S-a decis efectuarea unui consult cardiologic și a unui consult terapeut. De asemenea, s-a decis utilizarea ventilației non-invazive BiPAP.",
                },
            ],
        }
    elif meeting_type == MeetingType.ADMINISTRATIVE.value:
        return {
            "title": "Ședință Operațională: Logistică, Mentenanță Tehnică și Aprovizionare",
            "discussion_points": [
                {
                    "point": "Defecțiune la sistemul de climatizare (HVAC) din Blocul Operator sala 3 și stocul de biocide.",
                    "decisions": [
                        "Echipa de mentenanță tehnică va schimba filtrele HEPA din sala 3 astăzi până la ora 18:00.",
                        "Departamentul Achiziții va plasa comanda de urgență pentru dezinfectant până mâine la prânz.",
                    ],
                }
            ],
        }
    else:  # EXECUTIVE
        return {
            "title": "Ședință Comitet Director: Revizuire Buget și Plan Investiții",
            "discussion_points": [
                {
                    "point": "Analiza execuției bugetare pentru secția Chirurgie Cardiovasculară și achiziția noului angiograf.",
                    "decisions": [
                        "Se aprobă plafonul de Capex în valoare de 450.000 EUR pentru inițierea licitației.",
                        "Directorul Financiar va prezenta analiza de amortizare până pe 15 a lunii viitoare.",
                    ],
                }
            ],
        }


def _sync_extract_decisions_and_actions(transcript: str, meeting_type: str) -> dict:
    if MOCK_MODE:
        return MOCK_DICT(meeting_type)

    from llama_cpp import Llama

    llm = Llama(
        model_path=LLM_MODEL_PATH,
        n_gpu_layers=28,
        n_ctx=8192,
        verbose=False,
    )

    if meeting_type == MeetingType.MEDICAL.value:
        system_prompt = medical_system_prompt
    elif meeting_type == MeetingType.ADMINISTRATIVE.value:
        system_prompt = administrative_system_prompt
    else:
        system_prompt = executive_system_prompt

    user_prompt = f"Transcript:\n{transcript}"

    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=3500,
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    raw_content = response["choices"][0]["message"]["content"].strip()

    if raw_content.startswith("```json"):
        raw_content = raw_content[7:]
    elif raw_content.startswith("```"):
        raw_content = raw_content[3:]
    if raw_content.endswith("```"):
        raw_content = raw_content[:-3]
    raw_content = raw_content.strip()

    # try:
        # result = json.loads(raw_content)
    # except json.JSONDecodeError:
        # result = MOCK_DICT(meeting_type)
    result = json.loads(raw_content)

    # Set appropriate default keys depending on meeting type
    if meeting_type == MeetingType.MEDICAL.value:
        result.setdefault("summary", "")
        result.setdefault("patients", [])
    else:
        result.setdefault("title", "")
        result.setdefault("discussion_points", [])

    del llm
    _clean_gpu()
    return result


async def extract_decisions_and_actions(transcript: str, meeting_type: str) -> dict:
    return await asyncio.to_thread(
        _sync_extract_decisions_and_actions, transcript, meeting_type
    )