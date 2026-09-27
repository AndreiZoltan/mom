# app/schemas.py
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


# Medical specific record
class PatientRecord(BaseModel):
    patient_id: int
    patient_summary: str
    patient_decision: str


# Executive & Administrative specific record
class DiscussionPoint(BaseModel):
    point: str
    decisions: List[str] = Field(default_factory=list)


class MoMResult(BaseModel):
    job_id: str
    status: JobStage
    meeting_type: str
    filename: str
    total_processing_sec: float
    transcript_preview: str
    distribution_list: List[str]

    # Populated for MEDICAL meetings
    summary: Optional[str] = None
    patients: Optional[List[PatientRecord]] = None

    # Populated for EXECUTIVE and ADMINISTRATIVE meetings
    title: Optional[str] = None
    discussion_points: Optional[List[DiscussionPoint]] = None


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