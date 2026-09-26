import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import app_ollama.config as config
from app_ollama.asr import transcribe_audio
from app_ollama.config import MOCK_MODE, OLLAMA_HOST, OLLAMA_MODEL
from app_ollama.llm import extract_decisions_and_actions
from app_ollama.mom import compile_mom_document
from app_ollama.schemas import MeetingType


async def check_ollama_status():
    """Verify Ollama is reachable and model exists before starting ASR."""
    if MOCK_MODE:
        return

    try:
        from ollama import AsyncClient

        client = AsyncClient(host=OLLAMA_HOST)
        tags = await client.list()

        # Check if requested model or tag exists
        available_models = [m.get("model", "") for m in tags.get("models", [])]
        model_base = OLLAMA_MODEL.split(":")[0]

        if not any(OLLAMA_MODEL in m or model_base in m for m in available_models):
            print(
                f"[!] Warning: Model '{OLLAMA_MODEL}' was not found in Ollama local cache.\n"
                f"    Run: 'ollama pull {OLLAMA_MODEL}' if inference fails."
            )
    except Exception as e:
        print(f"[!] Warning: Could not connect to Ollama server at {OLLAMA_HOST}: {e}")
        print("    Ensure Ollama is running ('ollama serve') or check OLLAMA_HOST in your environment.\n")


async def run(
    audio_path: str,
    meeting_type_str: str,
    device: str,
    print_json: bool,
    output_file: str | None,
):
    path_obj = Path(audio_path)
    if not path_obj.exists():
        print(f"[!] Error: Audio file not found at: {audio_path}")
        sys.exit(1)

    try:
        meeting_type = MeetingType(meeting_type_str)
    except ValueError:
        valid_types = [t.value for t in MeetingType]
        print(f"[!] Invalid meeting type '{meeting_type_str}'. Must be one of: {valid_types}")
        sys.exit(1)

    # Apply device override to configuration and environment
    config.DEVICE = device
    os.environ["DEVICE"] = device

    filename = path_obj.name
    total_start = time.time()

    print("\n" + "=" * 60)
    print(f"[*] Processing File : {filename}")
    print(f"[*] Meeting Category: {meeting_type.value}")
    print(f"[*] Config Active   : ASR Device='{device}' | Ollama Model='{OLLAMA_MODEL}'")
    print(f"[*] Ollama Endpoint : {OLLAMA_HOST}")
    print(f"[*] Mock Mode       : {MOCK_MODE}")
    print("=" * 60 + "\n")

    # Quick pre-flight check for Ollama
    await check_ollama_status()

    # --- STAGE 1: Whisper ASR ---
    print(f"[1/3] Transcribing audio with faster-whisper on [{device}]...")
    t0 = time.time()
    try:
        asr_res = await transcribe_audio(str(path_obj), meeting_type.value, device)
    except Exception as e:
        print(f"[!] ASR failed: {e}")
        sys.exit(1)

    asr_duration = round(time.time() - t0, 2)
    detected_lang = asr_res.get("language_detected", "unknown")
    print(f"      Done in {asr_duration}s. (Language: {detected_lang})")
    snippet = asr_res["transcript"][:160].replace("\n", " ")
    print(f"      Transcript preview: \"{snippet}...\"\n")

    # --- STAGE 2: Ollama LLM Extraction ---
    print(f"[2/3] Extracting decisions & action items via Ollama ({OLLAMA_MODEL})...")
    t0 = time.time()
    try:
        llm_res = await extract_decisions_and_actions(
            asr_res["transcript"], meeting_type.value
        )
    except Exception as e:
        print(f"[!] LLM extraction failed: {e}")
        print(f"[!] Make sure Ollama is running and accessible at '{OLLAMA_HOST}'")
        sys.exit(1)

    llm_duration = round(time.time() - t0, 2)
    print(f"      Done in {llm_duration}s.\n")

    # --- STAGE 3: MoM Compiler ---
    print("[3/3] Compiling Minutes of Meeting document...")
    elapsed_total = round(time.time() - total_start, 2)
    mom_result = compile_mom_document(
        job_id="local-cli-session",
        filename=filename,
        meeting_type=meeting_type,
        transcript=asr_res["transcript"],
        llm_output=llm_res,
        elapsed_sec=elapsed_total,
    )

    # Prepare outputs
    rendered_json = json.dumps(mom_result.model_dump(), indent=2, ensure_ascii=False)
    rendered_text = (
        f"{'=' * 60}\n"
        f"{mom_result.email_body_markdown}\n"
        f"{'=' * 60}\n"
        f"Recipients: {', '.join(mom_result.distribution_list)}\n"
        f"Total Pipeline Runtime: {elapsed_total}s\n"
    )

    # Print to console
    if print_json:
        print("\n=== RAW JSON OUTPUT ===")
        print(rendered_json)
    else:
        print("\n" + rendered_text)

    # Save to disk if requested
    if output_file:
        out_path = Path(output_file)
        content_to_save = (
            rendered_json
            if print_json or out_path.suffix == ".json"
            else mom_result.email_body_markdown
        )
        out_path.write_text(content_to_save, encoding="utf-8")
        print(f"[✓] Output written to: {out_path.resolve()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Offline MoM Pipeline locally using Whisper & Ollama"
    )
    parser.add_argument(
        "audio_path", help="Path to local audio file (.mp3, .wav, .m4a, etc.)"
    )
    parser.add_argument(
        "--device",
        default=config.DEVICE,
        help=f"Compute device for Whisper ASR (e.g., 'cpu', 'cuda', 'cuda:0'). Default: '{config.DEVICE}'",
    )
    parser.add_argument(
        "--type",
        default="Medical",
        choices=["Medical", "Executive", "Administrative"],
        help="Meeting category (default: Medical)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON output instead of formatted Markdown",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional file path to save the generated MoM (.md or .json)",
    )

    args = parser.parse_args()
    asyncio.run(
        run(
            args.audio_path,
            args.type,
            args.device,
            args.json,
            args.output,
        )
    )