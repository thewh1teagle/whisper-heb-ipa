"""
Usage:
uv pip install git+https://github.com/thewh1teagle/renikud.git#subdirectory=renikud-onnx
wget https://huggingface.co/thewh1teagle/renikud/resolve/main/model.onnx -O renikud.onnx
uv run scripts/phonemize.py renikud.onnx metadata.csv
"""

import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from renikud_onnx import G2P
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Phonemize Hebrew text from a metadata CSV")
    parser.add_argument("model", type=str, help="Path to model.onnx")
    parser.add_argument("metadata", type=str, help="Path to metadata CSV (id|text)")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    # Each thread needs its own G2P instance (ONNX session is not thread-safe)
    thread_local_g2p = {}

    def phonemize(row):
        import threading
        tid = threading.get_ident()
        if tid not in thread_local_g2p:
            thread_local_g2p[tid] = G2P(args.model)
        return row[0], thread_local_g2p[tid].phonemize(row[1])

    with open(args.metadata, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f, delimiter="|"))

    writer = csv.writer(sys.stdout, delimiter="|")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for id_, phonemes in tqdm(executor.map(phonemize, rows), total=len(rows)):
            writer.writerow([id_, phonemes])


if __name__ == "__main__":
    main()
