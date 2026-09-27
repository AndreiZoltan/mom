"""
train_whisper_lora.py
Fine-tune Whisper-medium with LoRA for Romanian/Russian/English medical code-switching.
Optimized for 6 GB - 8 GB VRAM GPUs.
"""

import os
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import Dataset, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_NAME = "openai/whisper-medium"
OUTPUT_DIR = "./whisper-medium-medical-lora"
LANGUAGE = "Romanian"  # Base language setting
TASK = "transcribe"


# 1. Data Collator with dynamic padding
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Split inputs and labels
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # Mask padding tokens in labels with -100 so CrossEntropyLoss ignores them
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # If bos token is prepended in previous steps, cut it off
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

import soundfile as sf
import librosa

def prepare_dataset(batch, feature_extractor, tokenizer):
    # batch["audio"] is simply the string file path
    audio_path = batch["audio"]
    
    # Read audio with soundfile (rock-solid, no torchcodec needed)
    speech_array, sampling_rate = sf.read(audio_path, dtype="float32")
    
    # Convert stereo to mono if needed
    if speech_array.ndim > 1:
        speech_array = speech_array.mean(axis=1)
        
    # Resample to 16 kHz if not already 16 kHz
    if sampling_rate != 16000:
        speech_array = librosa.resample(speech_array, orig_sr=sampling_rate, target_sr=16000)

    # Compute log-Mel spectrogram
    batch["input_features"] = feature_extractor(
        speech_array, sampling_rate=16000
    ).input_features[0]

    # Tokenize target sentence
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

def main():
    print(f"Loading processor for {MODEL_NAME}...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
    tokenizer = WhisperTokenizer.from_pretrained(
        MODEL_NAME, language=LANGUAGE, task=TASK
    )
    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME, language=LANGUAGE, task=TASK
    )

    # 2. Example / Mock Dataset
    # Replace this structure with your parsed Medplatform.md audio slices & transcripts
    sample_data = {
        "audio": [
            "data/sample1.wav",  # Place local 16kHz mono WAV paths here
            "data/sample2.wav",
        ],
        "sentence": [
            "La boxa doi pacientul este afebril, oxigenarea este de 90%, facem BiPAP.",
            "S-a decis administrarea de Forxiga și Diacarb, consult cardiologic urgent.",
        ],
    }

    if not os.path.exists("data/sample1.wav"):
        print("[!] Generating mock wav files for syntax demonstration...")
        import numpy as np
        import soundfile as sf
        os.makedirs("data", exist_ok=True)
        dummy_audio = np.zeros(16000 * 5, dtype=np.float32)  # 5 sec silence
        sf.write("data/sample1.wav", dummy_audio, 16000)
        sf.write("data/sample2.wav", dummy_audio, 16000)

    dataset = Dataset.from_dict(sample_data).cast_column("audio", Audio(sampling_rate=16000))
    train_dataset = dataset.map(
        lambda b: prepare_dataset(b, feature_extractor, tokenizer),
        remove_columns=["audio", "sentence"],
    )

    # 3. Model Loading & LoRA Configuration
    print(f"Loading {MODEL_NAME} in FP16 with gradient checkpointing...")
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.gradient_checkpointing_enable()

    # Target projection layers of attention modules
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,          # Keep at 1-2 for <=8 GB VRAM
        gradient_accumulation_steps=8,         # Effective batch size = 16
        learning_rate=1e-4,
        warmup_steps=50,
        max_steps=500,
        fp16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="no",
        save_total_limit=2,
        dataloader_num_workers=0,
        report_to="none",
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
    )

    print("Starting LoRA fine-tuning...")
    trainer.train()

    print(f"Saving fine-tuned adapter to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("Done!")


if __name__ == "__main__":
    main()