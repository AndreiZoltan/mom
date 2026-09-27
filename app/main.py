# app/main.py
import os
import shutil
import tempfile
import time
import uuid
from typing import Dict
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)

from app.asr import transcribe_audio
from app.llm import extract_decisions_and_actions
from app.mom import compile_mom_document
from app.schemas import (
    JobCreationResponse,
    JobStage,
    JobStatusResponse,
    MeetingType,
    MoMResult,
)

app = FastAPI(
    title="Secure Medical Documentation Pipeline",
    description="Offline ASR & ICU/Medical Board Patient Cases Synthesis",
    version="0.4.0",
)

# In-memory storage for jobs
JOBS_DB: Dict[str, dict] = {}


async def run_pipeline(
    job_id: str, temp_path: str, filename: str, meeting_type: MeetingType
):
    """Orchestrates Whisper ASR -> Local LLM -> Simplified Patient MoM Compiler."""
    job = JOBS_DB[job_id]
    try:
        # Step 1: Voice2Text
        job["status"] = JobStage.TRANSCRIBING
        job["progress"] = 30
        job["message"] = "Transcrierea audio-ului clinic (Whisper)..."
        asr_output = await transcribe_audio(temp_path, meeting_type.value)

        # Step 2: Text2Decisions (Patient Case Extraction)
        job["status"] = JobStage.EXTRACTING_DECISIONS
        job["progress"] = 75
        job["message"] = "Extragerea cazurilor clinice și deciziilor pe pacienți (Local LLM)..."
        llm_output = await extract_decisions_and_actions(
            asr_output["transcript"], meeting_type.value
        )

        # Step 3: MoM Creation
        job["status"] = JobStage.FORMATTING_MOM
        job["progress"] = 90
        job["message"] = "Sintetizarea raportului pe pacienți..."

        elapsed = round(time.time() - job["start_time"], 2)
        final_mom = compile_mom_document(
            job_id=job_id,
            filename=filename,
            meeting_type=meeting_type,
            transcript=asr_output["transcript"],
            llm_output=llm_output,
            elapsed_sec=elapsed,
        )

        job["status"] = JobStage.COMPLETED
        job["progress"] = 100
        job["message"] = "Raport finalizat cu succes."
        job["result"] = final_mom

    except Exception as e:
        job["status"] = JobStage.FAILED
        job["message"] = f"Pipeline failed: {str(e)}"
    finally:
        # Clean up uploaded audio temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


# --- Endpoints ---


@app.post(
    "/api/v1/meetings",
    response_model=JobCreationResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_meeting(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    meeting_type: MeetingType = Form(MeetingType.MEDICAL),
):
    """1. Upload audio -> returns job ID"""
    job_id = str(uuid.uuid4())

    # Save audio stream to temporary file
    suffix = os.path.splitext(file.filename or "")[-1] or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_audio_path = tmp.name

    JOBS_DB[job_id] = {
        "job_id": job_id,
        "status": JobStage.QUEUED,
        "progress": 5,
        "start_time": time.time(),
        "message": "Înregistrare în așteptare pentru procesare...",
        "result": None,
    }

    background_tasks.add_task(
        run_pipeline, job_id, temp_audio_path, file.filename or "audio.mp3", meeting_type
    )

    return JobCreationResponse(
        job_id=job_id,
        status=JobStage.QUEUED,
        message="Înregistrare recepționată. Procesarea a început.",
    )


@app.get("/api/v1/meetings/{job_id}/status", response_model=JobStatusResponse)
def get_status(job_id: str):
    """2. ID -> returns status and current processing stage"""
    if job_id not in JOBS_DB:
        raise HTTPException(status_code=404, detail="Job not found")

    job = JOBS_DB[job_id]
    elapsed = round(time.time() - job["start_time"], 2)

    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress_percent=job["progress"],
        elapsed_seconds=elapsed,
        message=job["message"],
    )


@app.get("/api/v1/meetings/{job_id}/result", response_model=MoMResult)
def get_result(job_id: str):
    """3. ID -> returns final MoM with patient summaries and decisions"""
    if job_id not in JOBS_DB:
        raise HTTPException(status_code=404, detail="Job not found")

    job = JOBS_DB[job_id]

    if job["status"] == JobStage.FAILED:
        raise HTTPException(status_code=500, detail=job["message"])

    if job["status"] != JobStage.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_425_TOO_EARLY,
            detail=f"Rezultatul nu este încă gata. Status: {job['status']} ({job['progress']}%)",
        )

    return job["result"]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8002, 
                # reload=True,
                reload=False
                )
