"""
Usage:
uv run scripts/prepare_saspeech.py saspeech_gold_standard/metadata.csv > saspeech_gold_standard/metadata_clean.csv
uv run scripts/prepare_saspeech.py saspeech_automatic_data/metadata.csv --text_col 2 > saspeech_automatic_data/metadata_clean.csv
"""

import argparse
import csv
import re
import sys


def clean(text):
    text = text.replace("…", ".")
    text = re.sub(r"[\u0590-\u05cf|]", "", text)  # strip nikud and prefix boundary marker
    return text


def main():
    parser = argparse.ArgumentParser(description="Prepare SASpeech metadata CSV into id|text format")
    parser.add_argument("metadata", type=str, help="Path to SASpeech metadata.csv")
    parser.add_argument("--text_col", type=int, default=1, help="Column index of transcript (default: 1; use 2 for automatic data)")
    parser.add_argument("--id_col", type=int, default=0, help="Column index of ID (default: 0; use 1 for automatic data where file_id matches wav filename)")
    parser.add_argument("--delimiter", type=str, default="|", help="Input delimiter (default: |; use tab for TSV)")
    args = parser.parse_args()

    delim = "\t" if args.delimiter == "tab" else args.delimiter
    writer = csv.writer(sys.stdout, delimiter="|")
    with open(args.metadata, newline="", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter=delim):
            if len(row) <= args.text_col:
                continue
            id_, text = row[args.id_col], row[args.text_col]
            writer.writerow([id_, clean(text)])


if __name__ == "__main__":
    main()
