# app/mom.py
import re
from typing import Any, List
from app.config import DISTRIBUTION_LISTS
from app.schemas import JobStage, MeetingType, MoMResult, PatientRecord


def _safe_int(val: Any, default: int = 0) -> int:
    """Extracts integer from number or string (e.g., '1' -> 1)."""
    if isinstance(val, int):
        return val
    found = re.findall(r"\d+", str(val))
    return int(found[0]) if found else default


def _format_decision(val: Any) -> str:
    """Normalizes decision whether returned as a list or a string."""
    if isinstance(val, list):
        return " ".join(str(d).strip() for d in val if d)
    return str(val).strip() if val else ""


def compile_mom_document(
    job_id: str,
    filename: str,
    meeting_type: MeetingType,
    transcript: str,
    llm_output: dict,
    elapsed_sec: float,
) -> MoMResult:
    # 1. Format simplified patient records
    formatted_patients: List[PatientRecord] = []
    for raw_p in llm_output.get("patients", []):
        formatted_patients.append(
            PatientRecord(
                patient_id=_safe_int(raw_p.get("patient_id"), default=0),
                patient_summary=str(raw_p.get("patient_summary", "")).strip(),
                patient_decision=_format_decision(raw_p.get("patient_decision", "")),
            )
        )

    # 2. Get distribution list for meeting type
    distribution_list = DISTRIBUTION_LISTS.get(
        meeting_type.value, ["consiliu.medical@medpark.local"]
    )

    # 3. Create short transcript preview
    preview = transcript[:300] + ("..." if len(transcript) > 300 else "")

    return MoMResult(
        job_id=job_id,
        status=JobStage.COMPLETED,
        meeting_type=meeting_type.value,
        filename=filename,
        total_processing_sec=elapsed_sec,
        transcript_preview=preview,
        distribution_list=distribution_list,
        summary=llm_output.get("summary", ""),
        patients=formatted_patients,
    )