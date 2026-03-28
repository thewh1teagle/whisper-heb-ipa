import pandas as pd
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
from datasets import Dataset, Audio
from constants import SAMPLING_RATE, MAX_LABEL_TOKENS


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_features": self.processor.feature_extractor(
                f["audio"]["array"], sampling_rate=f["audio"]["sampling_rate"]
            ).input_features[0]}
            for f in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": self.processor.tokenizer(f["text"]).input_ids} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def _load_split(data_path, metadata, wav_dir, tokenizer):
    raw = pd.read_csv(data_path / metadata, sep="|", header=None)
    if raw.shape[1] == 3:
        # id|ipa|text — use ipa as training target
        raw.columns = ["filename", "text", "_text_orig"]
        raw = raw[["filename", "text"]]
    else:
        raw.columns = ["filename", "text"]
    df = raw

    keep_rows = []
    skipped = 0
    for _, row in df.iterrows():
        label_len = len(tokenizer(row["text"]).input_ids)
        if label_len <= MAX_LABEL_TOKENS:
            keep_rows.append(row)
        else:
            skipped += 1

    if skipped:
        print(f"Skipped {skipped} overlong examples in {metadata} (> {MAX_LABEL_TOKENS} label tokens)")

    if not keep_rows:
        raise ValueError(f"All examples in {metadata} were filtered out as overlong.")

    df = pd.DataFrame(keep_rows)
    audio_paths = [str(data_path / wav_dir / f"{filename}.wav") for filename in df["filename"]]
    dataset = Dataset.from_dict({"audio": audio_paths, "text": df["text"].tolist()})
    return dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))


def load_dataset_from_csv(data_dir, tokenizer, train_metadata="metadata_train.csv", eval_metadata="metadata_test.csv", wav_dir="wav"):
    data_path = Path(data_dir)
    return {
        "train": _load_split(data_path, train_metadata, wav_dir, tokenizer),
        "eval": _load_split(data_path, eval_metadata, wav_dir, tokenizer),
    }
