import pandas as pd
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union
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


def load_dataset_from_csv(data_dir, tokenizer, metadata="metadata.csv", wav_dir="wav", max_eval_samples=150):
    data_path = Path(data_dir)
    df = pd.read_csv(data_path / metadata, sep="|", header=None, names=["filename", "text"])

    keep_rows = []
    skipped = 0
    for _, row in df.iterrows():
        label_len = len(tokenizer(row["text"]).input_ids)
        if label_len <= MAX_LABEL_TOKENS:
            keep_rows.append(row)
        else:
            skipped += 1

    if skipped:
        print(f"Skipped {skipped} overlong examples (> {MAX_LABEL_TOKENS} label tokens)")

    if not keep_rows:
        raise ValueError("All examples were filtered out as overlong.")

    df = pd.DataFrame(keep_rows)
    audio_paths = [str(data_path / wav_dir / f"{filename}.wav") for filename in df["filename"]]
    dataset = Dataset.from_dict({
        "audio": audio_paths,
        "text": df["text"].tolist()
    })
    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    splits = dataset.train_test_split(test_size=max_eval_samples, shuffle=True)
    return {"train": splits["train"], "eval": splits["test"]}
