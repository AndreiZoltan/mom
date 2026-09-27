# app/mom.py
import re
from typing import Any, List
from app.config import DISTRIBUTION_LISTS
from app.schemas import DiscussionPoint, JobStage, MeetingType, MoMResult, PatientRecord


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


def _compile_medical_mom(llm_output: dict) -> tuple[str, List[PatientRecord]]:
    patients: List[PatientRecord] = []
    for raw_p in llm_output.get("patients", []):
        patients.append(
            PatientRecord(
                patient_id=_safe_int(raw_p.get("patient_id"), default=0),
                patient_summary=str(raw_p.get("patient_summary", "")).strip(),
                patient_decision=_format_decision(raw_p.get("patient_decision", "")),
            )
        )
    return str(llm_output.get("summary", "")).strip(), patients


def _compile_business_mom(llm_output: dict) -> tuple[str, List[DiscussionPoint]]:
    title = str(llm_output.get("title", "")).strip()
    discussion_points: List[DiscussionPoint] = []

    for raw_dp in llm_output.get("discussion_points", []):
        raw_decisions = raw_dp.get("decisions", [])
        if isinstance(raw_decisions, str):
            decisions = [raw_decisions.strip()] if raw_decisions.strip() else []
        elif isinstance(raw_decisions, list):
            decisions = [str(d).strip() for d in raw_decisions if str(d).strip()]
        else:
            decisions = []

        discussion_points.append(
            DiscussionPoint(
                point=str(raw_dp.get("point", "")).strip(),
                decisions=decisions,
            )
        )
    return title, discussion_points


def compile_mom_document(
    job_id: str,
    filename: str,
    meeting_type: MeetingType,
    transcript: str,
    llm_output: dict,
    elapsed_sec: float,
) -> MoMResult:
    # 1. Get distribution list
    distribution_list = DISTRIBUTION_LISTS.get(
        meeting_type.value, ["consiliu.medical@medpark.local"]
    )

    # 2. Preview
    preview = transcript[:300] + ("..." if len(transcript) > 300 else "")

    # 3. Format according to meeting category
    if meeting_type == MeetingType.MEDICAL:
        summary, patients = _compile_medical_mom(llm_output)
        return MoMResult(
            job_id=job_id,
            status=JobStage.COMPLETED,
            meeting_type=meeting_type.value,
            filename=filename,
            total_processing_sec=elapsed_sec,
            transcript_preview=preview,
            distribution_list=distribution_list,
            summary=summary,
            patients=patients,
        )
    else:
        title, discussion_points = _compile_business_mom(llm_output)
        return MoMResult(
            job_id=job_id,
            status=JobStage.COMPLETED,
            meeting_type=meeting_type.value,
            filename=filename,
            total_processing_sec=elapsed_sec,
            transcript_preview=preview,
            distribution_list=distribution_list,
            title=title,
            discussion_points=discussion_points,
        )