"""
To train the model:
uv run src/train.py --data_dir data --model_name openai/whisper-small --output_dir whisper-heb-ipa --batch_size 16 --learning_rate 1e-5 --max_steps 1000

To upload the model to the hub:
uv run hf upload --repo-type model whisper-heb-ipa ./whisper-heb-ipa

To use with wandb:
uv run wandb login
"""

import argparse
import pandas as pd
import torch
from pathlib import Path
from datasets import Dataset, Audio
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--model_name", type=str, default="openai/whisper-small")
parser.add_argument("--output_dir", type=str, default="./whisper-heb-ipa")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--max_steps", type=int, default=1000)
parser.add_argument("--report_to", type=str, default="tensorboard", choices=["wandb", "tensorboard"])
args = parser.parse_args()

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

def load_dataset_from_csv(data_dir):
    # Read metadata
    data_path = Path(data_dir)
    df = pd.read_csv(data_path / "metadata.csv", sep="|", header=None, names=["filename", "text"])
    
    # Create audio paths
    audio_paths = [str(data_path / "wav" / f"{filename}.wav") for filename in df["filename"]]
    
    # Create dataset
    dataset = Dataset.from_dict({
        "audio": audio_paths,
        "text": df["text"].tolist()
    })
    
    # Cast audio column
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # Split train/test (80/20)
    dataset = dataset.train_test_split(test_size=0.2)
    return dataset

def prepare_dataset(batch, processor):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

def compute_metrics(pred, tokenizer):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    wer = evaluate.load("wer")
    wer_score = 100 * wer.compute(predictions=pred_str, references=label_str)
    
    return {"wer": wer_score}

def main():
    # Load processor and model
    processor = WhisperProcessor.from_pretrained(args.model_name, language="Hebrew", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    
    # Set generation config
    model.generation_config.language = "hebrew"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None # Deprecated
    
    # Load dataset
    dataset = load_dataset_from_csv(args.data_dir)
    
    # Prepare dataset
    dataset = dataset.map(
        lambda batch: prepare_dataset(batch, processor),
        remove_columns=dataset["train"].column_names,
        num_proc=min(6, os.cpu_count()) # limit to 16 processes
    )
    
    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=8,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        logging_steps=25,
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        report_to=args.report_to,
    )
    
    # Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor.tokenizer),
        tokenizer=processor,
    )
    
    # Train
    trainer.train()
    
    # Save model
    trainer.save_model()

if __name__ == "__main__":
    main()