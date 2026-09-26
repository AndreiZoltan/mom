import asyncio
import json
from ollama import AsyncClient
from app_ollama.config import MOCK_MODE, OLLAMA_HOST, OLLAMA_MODEL


def MOCK_DICT(meeting_type: str) -> dict:
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


async def extract_decisions_and_actions(transcript: str, meeting_type: str) -> dict:
    """Takes transcript and extracts structured decisions and action items via Ollama on GPU."""
    if MOCK_MODE:
        await asyncio.sleep(2.0)
        return MOCK_DICT(meeting_type)

    client = AsyncClient(host=OLLAMA_HOST)

    schema_example = json.dumps(MOCK_DICT(meeting_type), ensure_ascii=False)
    system_prompt = (
        "You are a medical secretary at Medpark Hospital. "
        "Extract the executive summary, key decisions, and action items from the provided transcript.\n"
        f"You MUST return ONLY valid JSON matching this schema:\n{schema_example}"
    )

    user_prompt = f"Meeting Type: {meeting_type}\n\nTranscript:\n{transcript}"

    response = await client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        format="json",
        options={
            "temperature": 0.1,
            "num_ctx": 4096,     # Fits safely within 6GB - 8GB VRAM
            "num_gpu": 99,       # Forces all layers to offload to GPU
        },
    )

    raw_content = response["message"]["content"].strip()

    # Clean markdown enclosures if any
    if raw_content.startswith("```json"):
        raw_content = raw_content[7:]
    elif raw_content.startswith("```"):
        raw_content = raw_content[3:]
    if raw_content.endswith("```"):
        raw_content = raw_content[:-3]
    raw_content = raw_content.strip()

    try:
        result = json.loads(raw_content)
    except json.JSONDecodeError:
        result = {
            "summary": "Error parsing JSON from Ollama output.",
            "decisions_made": [],
            "raw_actions": [],
            "raw_output": raw_content,
        }

    result.setdefault("summary", "")
    result.setdefault("decisions_made", [])
    result.setdefault("raw_actions", [])

    return result