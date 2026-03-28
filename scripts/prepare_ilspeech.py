"""
Prepare ILSpeech speaker1 + speaker2 for training.
Downloads, extracts, merges, phonemizes, and splits into train/test.

Usage:
uv run scripts/prepare_ilspeech.py renikud.onnx
"""

import argparse
import csv
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import sh
from tqdm import tqdm

SPEAKERS = {
    "s1": "https://huggingface.co/datasets/thewh1teagle/ILSpeech/resolve/main/speaker1/ilspeech_speaker1_v1.7z",
    "s2": "https://huggingface.co/datasets/thewh1teagle/ILSpeech/resolve/main/speaker2/ilspeech_speaker2_v1.7z",
}
TEST_SIZE = 150


def main():
    parser = argparse.ArgumentParser(description="Prepare ILSpeech dataset for training")
    parser.add_argument("model", type=str, help="Path to renikud.onnx")
    parser.add_argument("--output_dir", type=str, default="dataset/ilspeech", help="Output directory")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    wav_dir = out / "wav"
    wav_dir.mkdir(exist_ok=True)
    dataset_dir = Path("dataset")

    # Step 1: download + extract
    rows = []  # (id, text)
    for prefix, url in SPEAKERS.items():
        archive_name = url.split("/")[-1].split("?")[0]
        archive_path = dataset_dir / archive_name
        folder_name = archive_name.replace(".7z", "")
        extract_path = dataset_dir / folder_name

        print(f"Downloading {archive_name}...")
        sh.wget("-nc", url, "-O", str(archive_path), _ok_code=[0, 1])

        if not extract_path.exists():
            print(f"Extracting {archive_name}...")
            sh.Command("7z")("x", str(archive_path), f"-o{dataset_dir}")

        # read metadata
        metadata_path = extract_path / "metadata.csv"
        with open(metadata_path, encoding="utf-8") as f:
            for row in csv.reader(f, delimiter="|"):
                if len(row) < 2:
                    continue
                file_id, phonemes = row[0], row[1]
                new_id = f"{prefix}_{file_id}"
                # copy wav
                src_wav = extract_path / "wav" / f"{file_id}.wav"
                dst_wav = wav_dir / f"{new_id}.wav"
                if not dst_wav.exists() and src_wav.exists():
                    sh.cp(str(src_wav), str(dst_wav))
                rows.append((new_id, phonemes))

    print(f"Total rows: {len(rows)}")

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

    phonemized = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for file_id, phonemes in tqdm(executor.map(phonemize, rows), total=len(rows)):
            phonemized.append((file_id, phonemes))

    # Step 3: deterministic split — shuffle with fixed seed, last TEST_SIZE = test
    import random
    random.seed(42)
    random.shuffle(phonemized)
    test_rows = phonemized[-TEST_SIZE:]
    train_rows = phonemized[:-TEST_SIZE]

    def write_csv(path, data):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="|")
            for row in data:
                writer.writerow(row)

    write_csv(out / "metadata_ipa_train.csv", train_rows)
    write_csv(out / "metadata_ipa_test.csv", test_rows)

    print(f"Train: {len(train_rows)}, Test: {len(test_rows)}")
    print(f"Done. Written to {out}")


if __name__ == "__main__":
    main()
