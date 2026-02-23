"""LoRA SFT fine-tuning of Qwen2-VL-2B-Instruct on WhiteboardGym dataset.

Trains the model to predict action token strings given (canvas_image, transcript).

Usage:
    python3 -m src.training.sft_train \
        --dataset  data/processed/dataset.jsonl \
        --output   checkpoints/sft-v1 \
        --epochs   2 \
        --batch-size 1
"""

import argparse
import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_model, LoraConfig


MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

PROMPT_TEMPLATE = "Transcript: {transcript}\nAction:"


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class WhiteboardSFTDataset(Dataset):
    """JSONL dataset for SFT. Each item returns processor-ready inputs with masked labels."""

    def __init__(self, jsonl_path: str, processor, max_length: int = 512):
        self.records = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        image = Image.open(rec["image_path"]).convert("RGB")
        transcript = rec.get("transcript", "")
        action_str = rec["action_str"]

        # Build the conversation in Qwen2-VL chat format
        prompt_text = PROMPT_TEMPLATE.format(transcript=transcript)
        full_text = prompt_text + " " + action_str

        # Process with the Qwen2-VL processor: image + text
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": action_str},
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        # Squeeze batch dim
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # Label masking: only the assistant's action_str tokens contribute to loss.
        # Tokenize just the prompt portion to find where assistant tokens begin.
        prompt_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            },
        ]
        prompt_only_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_inputs = self.processor(
            text=[prompt_only_text],
            images=[image],
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_len = prompt_inputs["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # Pass through pixel_values and image_grid_thw if present
        if "pixel_values" in inputs:
            result["pixel_values"] = inputs["pixel_values"].squeeze(0)
        if "image_grid_thw" in inputs:
            result["image_grid_thw"] = inputs["image_grid_thw"].squeeze(0)

        return result


def main():
    parser = argparse.ArgumentParser(description="LoRA SFT on Qwen2-VL-2B-Instruct")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument("--output", required=True, help="Checkpoint output directory")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    device = _detect_device()
    print(f"Device: {device}")

    # Load model and processor
    print(f"Loading {MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset
    dataset = WhiteboardSFTDataset(args.dataset, processor, max_length=args.max_length)
    print(f"Dataset: {len(dataset)} records")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=1,
        save_strategy="epoch",
        remove_unused_columns=False,
        fp16=(device == "cuda"),
        dataloader_pin_memory=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print("Starting SFT training ...")
    trainer.train()

    # Save LoRA adapter
    model.save_pretrained(args.output)
    processor.save_pretrained(args.output)
    print(f"Checkpoint saved to {args.output}")


if __name__ == "__main__":
    main()
