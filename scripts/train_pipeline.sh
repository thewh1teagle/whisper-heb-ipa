#!/bin/bash
set -e

DATASET_DIR="dataset"
ILSPEECH_DIR="$DATASET_DIR/ilspeech-v2"
SASPEECH_DIR="$DATASET_DIR/saspeech"
CROWD_RECITAL_RAW="$DATASET_DIR/crowd-recital-whisper-training"
CROWD_RECITAL_DIR="$DATASET_DIR/crowd-recital"
STAGE1_DIR="checkpoints/stage1"
STAGE2_DIR="checkpoints/stage2"
STAGE3_DIR="checkpoints/stage3"
RENIKUD_MODEL="renikud.onnx"

# ── ILSpeech ──────────────────────────────────────────────────────────────────

echo "==> Downloading ILSpeech v2..."
[ -f "$DATASET_DIR/ilspeech-v2.7z" ] || wget "https://huggingface.co/datasets/thewh1teagle/ILSpeech/resolve/main/ilspeech-v2.7z?download=true" \
    -O "$DATASET_DIR/ilspeech-v2.7z"

echo "==> Extracting ILSpeech..."
[ -d "$ILSPEECH_DIR" ] || 7z x "$DATASET_DIR/ilspeech-v2.7z" -o"$DATASET_DIR" -y

# ── SASpeech ──────────────────────────────────────────────────────────────────

echo "==> Downloading SASpeech automatic..."
mkdir -p "$DATASET_DIR/saspeech_automatic"
[ -f "$DATASET_DIR/saspeech_automatic.7z" ] || wget "https://huggingface.co/datasets/thewh1teagle/saspeech/resolve/main/saspeech_automatic/saspeech_automatic.7z?download=true" \
    -O "$DATASET_DIR/saspeech_automatic.7z"

echo "==> Downloading SASpeech manual..."
mkdir -p "$DATASET_DIR/saspeech_manual"
[ -f "$DATASET_DIR/saspeech_manual.7z" ] || wget "https://huggingface.co/datasets/thewh1teagle/saspeech/resolve/main/saspeech_manual/saspeech_manual_v2.7z?download=true" \
    -O "$DATASET_DIR/saspeech_manual.7z"

echo "==> Extracting SASpeech..."
[ -d "$DATASET_DIR/saspeech_automatic/wav" ] || 7z x "$DATASET_DIR/saspeech_automatic.7z" -o"$DATASET_DIR/saspeech_automatic" -y
[ -d "$DATASET_DIR/saspeech_manual/saspeech_manual" ] || 7z x "$DATASET_DIR/saspeech_manual.7z" -o"$DATASET_DIR/saspeech_manual" -y

# clean tab-delimited metadata into id|text format
echo "==> Preparing SASpeech metadata..."
uv run scripts/prepare_saspeech.py "$DATASET_DIR/saspeech_automatic/metadata.csv" --text_col 1 --delimiter tab \
    > "$DATASET_DIR/saspeech_automatic/metadata_clean.csv"
uv run scripts/prepare_saspeech.py "$DATASET_DIR/saspeech_manual/saspeech_manual/metadata.csv" --text_col 1 --delimiter tab \
    > "$DATASET_DIR/saspeech_manual/metadata_clean.csv"

echo "==> Downloading renikud model..."
[ -f "$RENIKUD_MODEL" ] || wget "https://huggingface.co/thewh1teagle/renikud/resolve/main/model.onnx" -O "$RENIKUD_MODEL"

# install g2p lib for phonemize.py
echo "==> Installing renikud..."
uv pip install "git+https://github.com/thewh1teagle/renikud.git#subdirectory=renikud-onnx"

# convert hebrew text to IPA phonemes
echo "==> Phonemizing SASpeech automatic..."
[ -f "$DATASET_DIR/saspeech_automatic/metadata_ipa.csv" ] || \
    uv run scripts/phonemize.py "$RENIKUD_MODEL" "$DATASET_DIR/saspeech_automatic/metadata_clean.csv" \
    > "$DATASET_DIR/saspeech_automatic/metadata_ipa.csv"

echo "==> Phonemizing SASpeech manual..."
[ -f "$DATASET_DIR/saspeech_manual/metadata_ipa.csv" ] || \
    uv run scripts/phonemize.py "$RENIKUD_MODEL" "$DATASET_DIR/saspeech_manual/metadata_clean.csv" \
    > "$DATASET_DIR/saspeech_manual/metadata_ipa.csv"

# merge both saspeech IPA metadata + wavs, prefix ids to avoid collisions
echo "==> Combining SASpeech..."
[ -f "$SASPEECH_DIR/metadata_train.csv" ] || \
    uv run scripts/pipeline/combine_saspeech.py "$DATASET_DIR" "$SASPEECH_DIR"

# use ilspeech test set (already IPA) as stage 1 eval
echo "==> Copying ILSpeech test set as eval..."
[ -f "$SASPEECH_DIR/metadata_test.csv" ] || \
    uv run scripts/pipeline/copy_ilspeech_eval.py "$ILSPEECH_DIR" "$SASPEECH_DIR"

# ── Crowd Recital ─────────────────────────────────────────────────────────────

echo "==> Downloading crowd-recital dataset..."
[ -d "$CROWD_RECITAL_RAW" ] || \
    uv run scripts/pipeline/download_crowd_recital.py "$CROWD_RECITAL_RAW"

# extract wavs, clean transcripts, phonemize to metadata_ipa.csv
echo "==> Preparing crowd-recital..."
[ -f "$CROWD_RECITAL_DIR/metadata_ipa.csv" ] || \
    uv run scripts/prepare_crowd_recital.py "$RENIKUD_MODEL" "$CROWD_RECITAL_RAW" "$CROWD_RECITAL_DIR"

# use ilspeech test set as eval
echo "==> Copying ILSpeech test set as crowd-recital eval..."
[ -f "$CROWD_RECITAL_DIR/metadata_test.csv" ] || \
    uv run scripts/pipeline/copy_ilspeech_eval.py "$ILSPEECH_DIR" "$CROWD_RECITAL_DIR"

# ── Stage 1: train on Crowd Recital ──────────────────────────────────────────

echo "==> Stage 1: training on crowd-recital..."
uv run src/train.py \
    --data_dir "$CROWD_RECITAL_DIR" \
    --train_metadata metadata_ipa.csv \
    --eval_metadata metadata_test.csv \
    --model_name ivrit-ai/whisper-large-v3-turbo \
    --output_dir "$STAGE1_DIR" \
    --num_train_epochs 10 \
    --early_stopping_patience 2 \
    "$@"

echo "==> Stage 1 done. Model at $STAGE1_DIR"

# ── Stage 2: fine-tune on SASpeech ───────────────────────────────────────────

echo "==> Stage 2: fine-tuning on SASpeech..."
uv run src/train.py \
    --data_dir "$SASPEECH_DIR" \
    --train_metadata metadata_train.csv \
    --eval_metadata metadata_test.csv \
    --model_name "$STAGE1_DIR" \
    --output_dir "$STAGE2_DIR" \
    --num_train_epochs 10 \
    --early_stopping_patience 2 \
    "$@"

echo "==> Stage 2 done. Model at $STAGE2_DIR"

# ── Stage 3: fine-tune on ILSpeech ───────────────────────────────────────────

echo "==> Stage 3: fine-tuning on ILSpeech..."
uv run src/train.py \
    --data_dir "$ILSPEECH_DIR" \
    --train_metadata metadata_train.csv \
    --eval_metadata metadata_test.csv \
    --model_name "$STAGE2_DIR" \
    --output_dir "$STAGE3_DIR" \
    --num_train_epochs 10 \
    --early_stopping_patience 2 \
    "$@"

echo "==> Stage 3 done. Model at $STAGE3_DIR"
