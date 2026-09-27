"""
train_qwen_qlora.py
Continued Domain Adaptation / Fine-Tuning of Qwen with 4-bit QLoRA
directly from a folder containing PDF files (USMF medical textbooks & protocols).

Hardware footprint: ~4.5 GB - 5.5 GB VRAM (fits on 6 GB GTX GPU).
"""

import os
import re
import argparse
from typing import List
from pypdf import PdfReader

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Base model identifier
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_PDF_DIR = "./usmf_books"
OUTPUT_DIR = "./qwen-medical-domain-lora"


def clean_medical_text(text: str) -> str:
    """Cleans extracted PDF text from artifacts, hyphens, and multi-spaces."""
    # Join hyphenated split words at line breaks (e.g., "hiper- \n tensiv" -> "hipertensiv")
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    # Replace multiple newlines and tab spaces with a single space
    text = re.sub(r"[\r\n\t]+", " ", text)
    # Remove repeated whitespace
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def extract_chunks_from_pdf_dir(
    pdf_dir: str, chunk_size: int = 1500, overlap: int = 150
) -> List[str]:
    """
    Reads all PDF files in the target directory, extracts page text,
    and slices the accumulated content into overlapping token/character chunks.
    """
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir, exist_ok=True)
        print(f"[!] Created directory '{pdf_dir}'. Please drop your USMF PDF files here.")
        return []

    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"[!] No PDF files found in '{pdf_dir}'.")
        return []

    print(f"Found {len(pdf_files)} PDF file(s) in '{pdf_dir}'. Extracting text...")
    all_chunks = []

    for pdf_file in pdf_files:
        full_path = os.path.join(pdf_dir, pdf_file)
        try:
            reader = PdfReader(full_path)
            doc_text = ""
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    doc_text += " " + clean_medical_text(page_text)

            # Slice document into overlapping text windows
            step = chunk_size - overlap
            for start in range(0, len(doc_text), step):
                chunk = doc_text[start : start + chunk_size]
                # Keep chunks with meaningful substance
                if len(chunk.strip()) > 300:
                    all_chunks.append(chunk.strip())

            print(f" -> Parsed '{pdf_file}': {len(reader.pages)} pages.")
        except Exception as e:
            print(f"[!] Error reading {pdf_file}: {e}")

    print(f"Total training passages generated: {len(all_chunks)}")
    return all_chunks


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen on medical PDFs with QLoRA")
    parser.add_argument(
        "--pdf_dir",
        type=str,
        default=DEFAULT_PDF_DIR,
        help="Path to folder containing PDF files (e.g., ./usmf_books)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=300,
        help="Number of training steps",
    )
    args = parser.parse_args()

    # 1. Extract training text from the PDF directory
    chunks = extract_chunks_from_pdf_dir(args.pdf_dir)
    if not chunks:
        print("[!] Dataset is empty. Insert at least one PDF file into the directory and rerun.")
        return

    dataset = Dataset.from_dict({"text": chunks})

    # 2. Tokenizer Setup
    print(f"Loading Tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. 4-bit NormalFloat (NF4) Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading Model {MODEL_NAME} in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # 4. Prepare for Low-VRAM QLoRA
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    # Target all linear projection layers of Qwen's attention and MLP
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 5. Training Configuration (Tuned for low memory / 6 GB GTX)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,        # 1 sample at a time
        gradient_accumulation_steps=8,       # Effective batch size = 8
        learning_rate=2e-4,
        warmup_ratio=0.03,
        max_steps=args.max_steps,
        logging_steps=10,
        fp16=True,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        optim="paged_adamw_8bit",             # Offloads optimizer states dynamically
        report_to="none",
    )

    # 6. SFTTrainer Execution
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args,
    )

    print(f"Starting QLoRA domain adaptation on PDF corpus from '{args.pdf_dir}'...")
    trainer.train()

    print(f"Saving fine-tuned adapter weights to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training finished successfully. Adapter is ready for local offline inference.")


if __name__ == "__main__":
    main()