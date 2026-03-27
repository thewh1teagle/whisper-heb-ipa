"""
Benchmark a Whisper IPA model against ILSpeech ground truth phonemes.

Setup:
    ./scripts/train_bench.sh

Usage:
    uv run scripts/benchmark.py --checkpoint whisper-heb-ipa/checkpoint-1000
"""

import argparse
import csv
import sys
from pathlib import Path

import jiwer
import torch
from tqdm import tqdm
from transformers import pipeline

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from constants import LANGUAGE

# Hebrew IPA word chars (consonants + vowels + stress)
HEBREW_IPA_CHARS = set("abdefhijklmnopstuvwzɡʁʃʒʔˈχ")
VOWELS = set("aeiou")
STRESS = "ˈ"


def load_gt(metadata_ipa: str):
    data = {}
    with open(metadata_ipa, newline="", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="|"):
            data[row[0]] = row[1]
    return data


def stress_error_rate(refs, hyps):
    """Fraction of words where stress position differs."""
    errors, total = 0, 0
    for ref, hyp in zip(refs, hyps):
        ref_words = ref.split()
        hyp_words = hyp.split()
        for rw, hw in zip(ref_words, hyp_words):
            total += 1
            ref_stress = rw.index(STRESS) if STRESS in rw else -1
            hyp_stress = hw.index(STRESS) if STRESS in hw else -1
            if ref_stress != hyp_stress:
                errors += 1
    return errors / total if total else 0.0


def vowel_error_rate(refs, hyps):
    """CER computed only over vowel characters."""
    ref_vowels = [" ".join(c for c in ref if c in VOWELS) for ref in refs]
    hyp_vowels = [" ".join(c for c in hyp if c in VOWELS) for hyp in hyps]
    # filter out empty
    pairs = [(r, h) for r, h in zip(ref_vowels, hyp_vowels) if r.strip()]
    if not pairs:
        return 0.0
    return jiwer.cer([p[0] for p in pairs], [p[1] for p in pairs])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Whisper checkpoint")
    parser.add_argument("--data_dir", type=str, default="ilspeech_speaker2_v1")
    parser.add_argument("--wav_dir", type=str, default="wav")
    parser.add_argument("--metadata_ipa", type=str, default="metadata.csv")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--save", type=str, default=None, help="Save report to file")
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    gt = dict(list(load_gt(data_path / args.metadata_ipa).items())[:args.max_samples])

    device = 0 if torch.cuda.is_available() else "cpu"
    pipe = pipeline(
        task="automatic-speech-recognition",
        model=args.checkpoint,
        chunk_length_s=30,
        device=device,
        generate_kwargs={"language": LANGUAGE, "task": "transcribe"},
    )

    refs, hyps, results = [], [], []
    for id_, ref in tqdm(gt.items()):
        audio_path = str(data_path / args.wav_dir / f"{id_}.wav")
        hyp = pipe(audio_path)["text"].strip()
        refs.append(ref)
        hyps.append(hyp)
        results.append({"id": id_, "ref": ref, "hyp": hyp, "correct": ref == hyp})

    cer = jiwer.cer(refs, hyps)
    wer = jiwer.wer(refs, hyps)
    ser = stress_error_rate(refs, hyps)
    ver = vowel_error_rate(refs, hyps)

    print("\nSample Predictions (first 5):")
    for r in results[:5]:
        print(f"\n  ID:   {r['id']}")
        print(f"  GT:   {r['ref']}")
        print(f"  Pred: {r['hyp']}")

    print(f"\nResults ({len(results)} samples):")
    print(f"  CER: {cer:.4f}")
    print(f"  WER: {wer:.4f}")
    print(f"  SER: {ser:.4f}  (stress error rate)")
    print(f"  VER: {ver:.4f}  (vowel error rate)")

    if args.save:
        wrong = [r for r in results if not r["correct"]]
        correct = [r for r in results if r["correct"]]
        with open(args.save, "w", encoding="utf-8") as f:
            f.write(f"Results: {len(results)} samples | CER: {cer:.4f} | WER: {wer:.4f} | SER: {ser:.4f} | VER: {ver:.4f}\n")
            f.write(f"Wrong: {len(wrong)} | Correct: {len(correct)}\n")
            f.write("=" * 80 + "\n\n")
            for r in wrong:
                f.write(f"[WRONG] {r['id']}\n")
                f.write(f"  GT:   {r['ref']}\n")
                f.write(f"  PRED: {r['hyp']}\n\n")
            f.write("=" * 80 + "\n\n")
            for r in correct:
                f.write(f"[OK] {r['id']}\n")
                f.write(f"  {r['ref']}\n\n")
        print(f"Report saved to {args.save}")


if __name__ == "__main__":
    main()
