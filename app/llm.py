import asyncio
import gc
import json
from app.config import MOCK_MODE, LLM_MODEL_PATH, HOSPITAL_STAFF_DIRECTORY
from app.schemas import ActionItem


def _clean_gpu():
    try:
        import torch

        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
    except ImportError:
        pass


def MOCK_DICT(meeting_type):
    return {
        "summary": (
            f"Ședința ({meeting_type}) a analizat consumul de antibiotice de rezervă, "
            "optimizarea triajului de urgență și achiziția consumabilelor pentru blocul operator."
        ),
        "decisions_made": [
            "Se aproba revizuirea protocolului de profilaxie antibacteriană perioperatorie.",
            "Consiliul a aprobat suplimentarea stocului pentru antibiotice de rezervă.",
            "Standard Operating Procedure (SOP) pentru triaj va fi aliniat cu noile cerințe.",
        ],
        "raw_actions": [
            {
                "task": "Pregătirea raportului de consum pe antibiotice și transmiterea către farmacie",
                "owner": "Elena Morari",
                "deadline": "2026-09-30",
                "department": "Farmacie Clinică",
                "priority": "Urgent",
            },
            {
                "task": "Aprobarea comenzii pentru kiturile de chirurgie laparoscopică",
                "owner": "Vasile Cebotari",
                "deadline": "2026-10-05",
                "department": "Chirurgie",
                "priority": "High",
            },
            {
                "task": "Actualizarea ghidului de triaj în urgență conform normelor noi",
                "owner": "Angela Rusu",
                "deadline": "2026-10-15",
                "department": "Departament Urgență",
                "priority": "Medium",
            },
        ],
    }


# async def extract_decisions_and_actions(transcript: str, meeting_type: str) -> dict:
def _sync_extract_decisions_and_actions(transcript: str, meeting_type: str) -> dict:
    """Takes transcript and extracts structured decisions and action items."""
    if MOCK_MODE:
        # await asyncio.sleep(3.0)
        return MOCK_DICT(meeting_type)
    # Real Inference (llama-cpp-python)
    from llama_cpp import Llama

    llm = Llama(model_path=LLM_MODEL_PATH, n_gpu_layers=28, n_ctx=8192, 
                # temperature=0.1, 
                verbose=False,)

    system_prompt = f"""
You are a medical secretary at Medpark Hospital. Extract decisions and action items from this transcript.
Return ONLY valid JSON matching: {MOCK_DICT(meeting_type)}
"""
    user_prompt = f"Meeting Type: {meeting_type}\n\nTranscript:\n{transcript}"

    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=2048,
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    raw_content = response["choices"][0]["message"]["content"]

    clean_content = raw_content.strip()
    if clean_content.startswith("```json"):
        clean_content = clean_content[7:]
    elif clean_content.startswith("```"):
        clean_content = clean_content[3:]
    if clean_content.endswith("```"):
        clean_content = clean_content[:-3]
    clean_content = clean_content.strip()

    try:
        result = json.loads(clean_content)
    except json.JSONDecodeError as e:
        # Fallback if the LLM output was truncated or slightly malformed
        result = {
            "summary": "Error parsing JSON from LLM output.",
            "decisions_made": [],
            "raw_actions": [],
            "raw_output": raw_content,
        }

    # Ensure required keys exist
    result.setdefault("summary", "")
    result.setdefault("decisions_made", [])
    result.setdefault("raw_actions", [])

    del llm
    _clean_gpu()
    return result


async def extract_decisions_and_actions(transcript: str, meeting_type: str) -> dict:
    if MOCK_MODE:
        await asyncio.sleep(3.0)
        return MOCK_DICT(meeting_type)

    return await asyncio.to_thread(_sync_extract_decisions_and_actions, transcript, meeting_type)