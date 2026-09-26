import re
from typing import List, Tuple
from app_ollama.config import DISTRIBUTION_LISTS, HOSPITAL_STAFF_DIRECTORY
from app_ollama.schemas import ActionItem, Attendee, JobStage, MeetingType, MoMResult, PatientCase


def _resolve_staff_member(name: str) -> Tuple[str, str]:
    """Matches a doctor/staff name against directory with prefix tolerance."""
    if not name or name.strip().lower() in ("unassigned", "none", "tbd"):
        return "Unassigned", "unassigned@medpark.local"

    clean_name = re.sub(r"^(dr\.|dr|doctor|prof\.)\s*", "", name.strip(), flags=re.IGNORECASE)

    # 1. Exact match
    if name in HOSPITAL_STAFF_DIRECTORY:
        return name, HOSPITAL_STAFF_DIRECTORY[name]
    if clean_name in HOSPITAL_STAFF_DIRECTORY:
        return clean_name, HOSPITAL_STAFF_DIRECTORY[clean_name]

    # 2. Substring match
    for staff_name, email in HOSPITAL_STAFF_DIRECTORY.items():
        if clean_name.lower() in staff_name.lower() or staff_name.lower() in clean_name.lower():
            return staff_name, email

    # 3. Fallback corporate email generator
    fallback_handle = clean_name.lower().replace(" ", ".")
    return name, f"{fallback_handle}@medpark.local"


def _build_action_item(raw_act: dict, action_id: int) -> ActionItem:
    raw_owner = raw_act.get("owner", "Unassigned")
    resolved_name, email = _resolve_staff_member(raw_owner)
    return ActionItem(
        id=action_id,
        task=raw_act.get("task", ""),
        owner=resolved_name,
        owner_email=email,
        deadline=raw_act.get("deadline", "TBD"),
        department=raw_act.get("department", "General"),
        priority=raw_act.get("priority", "Medium"),
    )


def compile_mom_document(
    job_id: str,
    filename: str,
    meeting_type: MeetingType,
    transcript: str,
    llm_output: dict,
    elapsed_sec: float,
) -> MoMResult:
    """Formats actions, resolves patient cases, attendees, and produces delivery markdown."""

    # 1. Parse top-level Action Items
    action_items: List[ActionItem] = []
    for idx, act in enumerate(llm_output.get("raw_actions", []), start=1):
        action_items.append(_build_action_item(act, idx))

    # 2. Parse Patient Cases (if present in clinical/multi-patient formats)
    patient_cases: List[PatientCase] = []
    case_counter = 1
    raw_cases = llm_output.get("patient_cases", [])

    for case_data in raw_cases:
        case_actions = [
            _build_action_item(act, len(action_items) + idx)
            for idx, act in enumerate(case_data.get("action_items", []), start=1)
        ]
        patient_cases.append(
            PatientCase(
                patient_id=case_data.get("patient_id", case_counter),
                clinical_summary=case_data.get("clinical_summary", ""),
                decisions=case_data.get("decisions", []),
                action_items=case_actions,
            )
        )
        case_counter += 1

    # 3. Detect Attendees (from assigned tasks or mentioned in transcript)
    detected_attendees_map = {}
    
    # Add owners of tasks
    for item in action_items:
        if item.owner != "Unassigned":
            detected_attendees_map[item.owner] = item.owner_email

    # Also detect staff mentioned in transcript text
    for staff_name, email in HOSPITAL_STAFF_DIRECTORY.items():
        if staff_name.lower() in transcript.lower():
            detected_attendees_map[staff_name] = email

    attendees: List[Attendee] = [
        Attendee(name=name, email=email, role="Participant")
        for name, email in detected_attendees_map.items()
    ]

    # Fallback to standard board members if nobody was specifically recognized
    if not attendees:
        attendees = [
            Attendee(name="Dr. Vasile Cebotari", email="v.cebotari@medpark.local", role="Șef Chirurgie"),
            Attendee(name="Elena Morari", email="e.morari@medpark.local", role="Farmacist Principal"),
        ]

    # 4. Generate Clean Markdown for email / export
    actions_md = (
        "\n".join(
            [
                f"- **[{item.priority}] {item.task}** | Owner: {item.owner} (`{item.owner_email}`) | Deadline: {item.deadline}"
                for item in action_items
            ]
        )
        if action_items
        else "_No general action items recorded._"
    )

    decisions = llm_output.get("decisions_made") or llm_output.get("general_decisions") or []
    decisions_md = (
        "\n".join([f"- {d}" for d in decisions])
        if decisions
        else "_No general decisions recorded._"
    )

    summary_text = llm_output.get("summary") or llm_output.get("general_summary") or ""

    # Optional Patient Cases block in Markdown
    cases_md = ""
    if patient_cases:
        cases_sections = []
        for case in patient_cases:
            case_decisions = "\n".join([f"  - {d}" for d in case.decisions]) or "  - None"
            case_acts = (
                "\n".join(
                    [
                        f"  - **[{act.priority}] {act.task}** (Owner: {act.owner} | {act.deadline})"
                        for act in case.action_items
                    ]
                )
                or "  - None"
            )
            cases_sections.append(
                f"### Patient Case #{case.patient_id}\n"
                f"**Clinical Status:** {case.clinical_summary}\n\n"
                f"**Clinical Decisions:**\n{case_decisions}\n\n"
                f"**Case Action Items:**\n{case_acts}"
            )
        cases_md = "\n\n## Clinical Patient Cases\n" + "\n\n".join(cases_sections)

    email_markdown = f"""# Minutes of Meeting: {meeting_type.value}
**Recording:** `{filename}` | **Processing Time:** {elapsed_sec}s

## Executive Summary
{summary_text}

## Key Decisions Made
{decisions_md}
{cases_md}

## Action Items & Responsibilities
{actions_md}
"""

    return MoMResult(
        job_id=job_id,
        status=JobStage.COMPLETED,
        meeting_type=meeting_type.value,
        filename=filename,
        total_processing_sec=elapsed_sec,
        summary=summary_text,
        decisions_made=decisions,
        patient_cases=patient_cases,
        action_items=action_items,
        attendees=attendees,
        distribution_list=DISTRIBUTION_LISTS.get(meeting_type.value, []),
        transcript_preview=transcript[:350] + ("..." if len(transcript) > 350 else ""),
        email_body_markdown=email_markdown,
    )