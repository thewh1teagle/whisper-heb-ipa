"""
Prepare ivrit-ai/crowd-recital-whisper-training dataset for training.
Extracts audio to wav files, cleans transcripts, and phonemizes to metadata_ipa.csv.

Usage:
uv run scripts/prepare_crowd_recital.py renikud.onnx dataset/crowd-recital-whisper-training dataset/crowd-recital
"""

import argparse
import csv
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_from_disk
from tqdm import tqdm


TIMESTAMP_RE = re.compile(r"<\|[\d.]+\|>")


def clean_transcript(text):
    text = TIMESTAMP_RE.sub("", text).strip()
    text = text.replace("…", ".")
    text = re.sub(r"[\u0590-\u05cf|]", "", text)  # strip nikud and prefix boundary marker
    return text


def main():
    parser = argparse.ArgumentParser(description="Prepare crowd-recital dataset into wav + metadata_ipa.csv")
    parser.add_argument("model", type=str, help="Path to renikud.onnx")
    parser.add_argument("input_dir", type=str, help="Path to downloaded dataset (dataset/crowd-recital-whisper-training)")
    parser.add_argument("output_dir", type=str, help="Output directory (e.g. dataset/crowd-recital)")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Phonemization threads (default: cpu count)")
    args = parser.parse_args()

    out = Path(args.output_dir)
    wav_dir = out / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset, concatenate_datasets
    try:
        ds = load_from_disk(args.input_dir)
        if hasattr(ds, "values"):
            ds = concatenate_datasets(list(ds.values()))
    except FileNotFoundError:
        # Downloaded via huggingface-cli — load from parquet files directly
        parquet_files = list(Path(args.input_dir).rglob("*.parquet"))
        ds = load_dataset("parquet", data_files=[str(f) for f in parquet_files], split="train")

    # Step 1: extract wavs and collect texts
    print("Extracting audio and cleaning transcripts...")
    rows = []  # (file_id, text)
    skipped = 0
    for i, example in enumerate(tqdm(ds, total=len(ds))):
        text = clean_transcript(example["transcript"])
        if not text:
            skipped += 1
            continue

        audio = example["audio"]
        array = np.array(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]

        file_id = str(i)
        wav_path = wav_dir / f"{file_id}.wav"
        if not wav_path.exists():
            sf.write(wav_path, array, sr)
        rows.append((file_id, text))

    if skipped:
        print(f"Skipped {skipped} examples with empty transcript", file=sys.stderr)

    # Step 2: phonemize
    print("Phonemizing...")
    from renikud_onnx import G2P
    thread_local = {}

    def phonemize(row):
        tid = threading.get_ident()
        if tid not in thread_local:
            thread_local[tid] = G2P(args.model)
        file_id, text = row
        return file_id, thread_local[tid].phonemize(text)

    with open(out / "metadata_ipa.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for file_id, phonemes in tqdm(executor.map(phonemize, rows), total=len(rows)):
                writer.writerow([file_id, phonemes])

    print(f"Done. Written to {out}")


if __name__ == "__main__":
    main()
