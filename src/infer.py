"""
Usage:
wget https://github.com/thewh1teagle/phonikud-chatterbox/releases/download/asset-files-v1/female1.wav -O audio.wav
uv run src/infer.py audio.wav
uv run src/infer.py audio.wav --model thewh1teagle/whisper-heb-ipa
uv run src/infer.py audio.wav --model ./whisper-heb-ipa/checkpoint-600
"""

import argparse
import torch
from transformers import pipeline


def main():
    parser = argparse.ArgumentParser(description="Transcribe a WAV file using Whisper")
    parser.add_argument("audio", type=str, help="Path to the audio file")
    parser.add_argument(
        "--model",
        type=str,
        default="thewh1teagle/whisper-heb-ipa",
        help="Model name or path (default: thewh1teagle/whisper-heb-ipa)",
    )
    args = parser.parse_args()

    device = 0 if torch.cuda.is_available() else "cpu"

    pipe = pipeline(
        task="automatic-speech-recognition",
        model=args.model,
        chunk_length_s=30,
        device=device,
    )

    result = pipe(args.audio, batch_size=8)
    print(result["text"])


if __name__ == "__main__":
    main()
