from typing import List
from app.config import DISTRIBUTION_LISTS, HOSPITAL_STAFF_DIRECTORY
from app.schemas import ActionItem, Attendee, JobStage, MeetingType, MoMResult


def compile_mom_document(
    job_id: str,
    filename: str,
    meeting_type: MeetingType,
    transcript: str,
    llm_output: dict,
    elapsed_sec: float,
) -> MoMResult:
    """Formats actions, resolves emails, and prepares email markdown."""

    # Map owners to corporate email addresses
    action_items: List[ActionItem] = []
    for idx, act in enumerate(llm_output.get("raw_actions", []), start=1):
        owner_name = act.get("owner", "Unassigned")
        email = HOSPITAL_STAFF_DIRECTORY.get(
            owner_name, f"{owner_name.lower().replace(' ', '.')}@medpark.local"
        )
        action_items.append(
            ActionItem(
                id=idx,
                task=act.get("task", ""),
                owner=owner_name,
                owner_email=email,
                deadline=act.get("deadline", "TBD"),
                department=act.get("department", "General"),
                priority=act.get("priority", "Medium"),
            )
        )

    # Predefined attendees based on participants detected
    attendees = [
        Attendee(
            name="Dr. Vasile Cebotari",
            email="v.cebotari@medpark.local",
            role="Șef Chirurgie",
        ),
        Attendee(
            name="Elena Morari",
            email="e.morari@medpark.local",
            role="Farmacist Principal",
        ),
        Attendee(
            name="Dr. Angela Rusu", email="a.rusu@medpark.local", role="Medic Urgență"
        ),
    ]

    # Generate Markdown for n8n email delivery
    actions_md = "\n".join(
        [
            f"- **[{item.priority}] {item.task}** | Owner: {item.owner} ({item.owner_email}) | Deadline: {item.deadline}"
            for item in action_items
        ]
    )
    decisions_md = "\n".join([f"- {d}" for d in llm_output.get("decisions_made", [])])

    email_markdown = f"""# Minutes of Meeting: {meeting_type.value}
**Recording File:** `{filename}` | **Processing Time:** {elapsed_sec}s

## Executive Summary
{llm_output.get("summary", "")}

## Key Decisions Made
{decisions_md}

## Action Items & Owners
{actions_md}
"""

    return MoMResult(
        job_id=job_id,
        status=JobStage.COMPLETED,
        meeting_type=meeting_type.value,
        filename=filename,
        total_processing_sec=elapsed_sec,
        summary=llm_output.get("summary", ""),
        decisions_made=llm_output.get("decisions_made", []),
        action_items=action_items,
        attendees=attendees,
        distribution_list=DISTRIBUTION_LISTS.get(meeting_type.value, []),
        transcript_preview=transcript[:350] + ("..." if len(transcript) > 350 else ""),
        email_body_markdown=email_markdown,
    )
