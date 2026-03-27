#!/bin/bash
# Usage: ./scripts/train_findtune.sh [extra args]
# Example: ./scripts/train_findtune.sh --model_name ivrit-ai/whisper-large-v3-turbo --max_steps 5000
uv run src/train.py \
    --data_dir saspeech_gold_standard \
    --metadata metadata_ipa.csv \
    --wav_dir wavs \
    "$@"
