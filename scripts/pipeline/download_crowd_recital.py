"""
Download ivrit-ai/crowd-recital-whisper-training from HuggingFace to disk.

Usage:
uv run scripts/pipeline/download_crowd_recital.py dataset/crowd-recital-whisper-training
"""

import argparse
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", help="Directory to save the dataset")
    args = parser.parse_args()

    print("Downloading ivrit-ai/crowd-recital-whisper-training...")
    ds = load_dataset("ivrit-ai/crowd-recital-whisper-training")
    ds.save_to_disk(args.output_dir)
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
