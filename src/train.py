"""
To train the model:
uv run src/train.py --data_dir data --model_name ivrit-ai/whisper-large-v3-turbo --output_dir whisper-heb-ipa --batch_size 16 --learning_rate 1e-5 --max_steps 1000
OR
export WANDB_PROJECT=whisper-heb-ipa
uv run src/train.py --data_dir data --model_name ivrit-ai/whisper-large-v3-turbo --output_dir whisper-heb-ipa --batch_size 16 --learning_rate 1e-5 --max_steps 90000 --report_to wandb

To upload the model to the hub:
uv run hf upload --repo-type model whisper-heb-ipa ./whisper-heb-ipa

To use with wandb:
uv run wandb login
"""

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from config import get_args
from constants import LANGUAGE, GENERATION_MAX_LENGTH
from eval import compute_metrics
from data import DataCollatorSpeechSeq2SeqWithPadding, load_dataset_from_csv


def main():
    args = get_args()
    # Load processor and model
    processor = WhisperProcessor.from_pretrained(args.model_name, language=LANGUAGE.capitalize(), task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    
    # Set generation config
    model.generation_config.language = LANGUAGE
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None # Deprecated
    
    dataset = load_dataset_from_csv(args.data_dir, args.metadata, args.wav_dir, args.max_eval_samples)

    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        # Gradient checkpointing is disabled to fix a "backward through the graph a second time" RuntimeError.
        # This error occurs when gradient checkpointing is enabled alongside a custom data collator.
        gradient_checkpointing=False,
        fp16=args.fp16,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        predict_with_generate=True,
        generation_max_length=GENERATION_MAX_LENGTH,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=args.report_to,
        dataloader_num_workers=args.dataloader_num_workers,
    )
    
    # Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor.tokenizer),
        processing_class=processor,
    )
    
    # Train
    trainer.train()
    
    # Save model
    trainer.save_model()

if __name__ == "__main__":
    main()