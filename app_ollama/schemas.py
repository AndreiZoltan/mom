from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class MeetingType(str, Enum):
    MEDICAL = "Medical"
    EXECUTIVE = "Executive"
    ADMINISTRATIVE = "Administrative"


class JobStage(str, Enum):
    QUEUED = "QUEUED"
    TRANSCRIBING = "TRANSCRIBING"
    EXTRACTING_DECISIONS = "EXTRACTING_DECISIONS"
    FORMATTING_MOM = "FORMATTING_MOM"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ActionItem(BaseModel):
    id: int
    task: str
    owner: str
    owner_email: str
    deadline: str
    department: str
    priority: str


class Attendee(BaseModel):
    name: str
    email: str
    role: str


class PatientCase(BaseModel):
    patient_id: int
    clinical_summary: str
    decisions: List[str]
    action_items: List[ActionItem]


class MoMResult(BaseModel):
    job_id: str
    status: JobStage
    meeting_type: str
    filename: str
    total_processing_sec: float
    summary: str
    decisions_made: List[str]
    patient_cases: List[PatientCase] = Field(default_factory=list)
    action_items: List[ActionItem]
    attendees: List[Attendee]
    distribution_list: List[str]
    transcript_preview: str
    email_body_markdown: str


class JobCreationResponse(BaseModel):
    job_id: str
    status: JobStage
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStage
    progress_percent: int
    elapsed_seconds: float
    message: str