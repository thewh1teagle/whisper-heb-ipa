"""
Merge saspeech automatic + manual IPA metadata and wavs into a single dataset dir.
Prefixes IDs with 'automatic_' / 'manual_' to avoid collisions.

Usage:
uv run scripts/pipeline/combine_saspeech.py dataset dataset/saspeech
"""

import argparse
import shutil
import pathlib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()

    base = pathlib.Path(args.dataset_dir)
    out = pathlib.Path(args.output_dir)
    out_wav = out / "wav"
    out_wav.mkdir(parents=True, exist_ok=True)

    rows = []
    for prefix, src, wav_src in [
        ("automatic", base / "saspeech_automatic", base / "saspeech_automatic"),
        ("manual",    base / "saspeech_manual",     base / "saspeech_manual" / "saspeech_manual"),
    ]:
        for line in (src / "metadata_ipa.csv").read_text().splitlines():
            if not line.strip():
                continue
            idx, ipa = line.split("|", 1)
            new_id = f"{prefix}_{idx}"
            rows.append(f"{new_id}|{ipa}")
            shutil.copy2(wav_src / "wav" / f"{idx}.wav", out_wav / f"{new_id}.wav")

    (out / "metadata_train.csv").write_text("\n".join(rows) + "\n")
    print(f"Combined {len(rows)} samples")


if __name__ == "__main__":
    main()
