"""
Copy ILSpeech test set (IPA) into a target dataset dir as metadata_test.csv + wavs.

Usage:
uv run scripts/pipeline/copy_ilspeech_eval.py dataset/ilspeech-v2 dataset/saspeech
"""

import argparse
import shutil
import pathlib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ilspeech_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()

    ilspeech = pathlib.Path(args.ilspeech_dir)
    out = pathlib.Path(args.output_dir)
    out_wav = out / "wav"
    out_wav.mkdir(parents=True, exist_ok=True)

    rows = []
    for line in (ilspeech / "metadata_test.csv").read_text().splitlines():
        if not line.strip():
            continue
        idx, ipa, text = line.split("|", 2)
        rows.append(f"{idx}|{ipa}")
        shutil.copy2(ilspeech / "wav" / f"{idx}.wav", out_wav / f"{idx}.wav")

    (out / "metadata_test.csv").write_text("\n".join(rows) + "\n")
    print(f"Copied {len(rows)} eval samples")


if __name__ == "__main__":
    main()
