#!/bin/bash
# Usage: ./scripts/train_findtune.sh [extra args]
# Example: ./scripts/train_findtune.sh --model_name ivrit-ai/whisper-large-v3-turbo --num_train_epochs 6
uv run src/train.py \
    --data_dir dataset/saspeech_gold_standard \
    --metadata metadata_ipa.csv \
    --wav_dir wav \
    "$@"
