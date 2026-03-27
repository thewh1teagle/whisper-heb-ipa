"""
Usage:
uv run scripts/prepare_saspeech.py saspeech_gold_standard/metadata.csv > saspeech_gold_standard/metadata_clean.csv
"""

import argparse
import csv
import re
import sys


def clean(text):
    text = text.replace("…", ".")
    text = re.sub(r"[\u0590-\u05cf]", "", text)  # strip nikud
    return text


def main():
    parser = argparse.ArgumentParser(description="Prepare SASpeech metadata CSV into id|text format")
    parser.add_argument("metadata", type=str, help="Path to SASpeech metadata.csv (id|text|text)")
    args = parser.parse_args()

    writer = csv.writer(sys.stdout, delimiter="|")
    with open(args.metadata, newline="", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="|"):
            id_, text = row[0], row[1]
            writer.writerow([id_, clean(text)])


if __name__ == "__main__":
    main()
