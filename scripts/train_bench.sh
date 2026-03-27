#!/bin/bash
# Usage: ./scripts/train_bench.sh <checkpoint>
# Example: ./scripts/train_bench.sh whisper-heb-ipa/checkpoint-1000

# Download and extract ILSpeech speaker2 benchmark data (first time only)
wget -nc https://huggingface.co/datasets/thewh1teagle/ILSpeech/resolve/main/speaker2/ilspeech_speaker2_v1.7z
if [ ! -d ilspeech_speaker2_v1 ]; then
    7z x ilspeech_speaker2_v1.7z
fi

uv run scripts/benchmark.py --checkpoint "${1:?Usage: $0 <checkpoint>}" --data_dir ilspeech_speaker2_v1
