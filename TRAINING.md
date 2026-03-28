# Training

## Data preparation

See `src/prepare.py` for data preparation.

## Training on SASpeech

Check total audio duration:

```console
soxi -DT ./dataset/saspeech_automatic/wav/*.wav | awk '{print $1/60 " minutes"}'
```

Download and extract into `dataset/`:

```console
mkdir -p dataset
wget https://openslr.trmal.net/resources/134/saspeech_gold_standard_v1.0.tar.gz -P dataset
tar xf dataset/saspeech_gold_standard_v1.0.tar.gz -C dataset/
```

Clean and prepare the transcripts (strips nikud, normalizes punctuation):

```console
uv run scripts/prepare_saspeech.py dataset/saspeech_gold_standard/metadata.csv > dataset/saspeech_gold_standard/metadata_clean.csv
```

Phonemize (produces `metadata_ipa.csv` with `id|phonemes`):

```console
wget https://huggingface.co/thewh1teagle/renikud/resolve/main/model.onnx -O renikud.onnx
uv run scripts/phonemize.py renikud.onnx dataset/saspeech_gold_standard/metadata_clean.csv > dataset/saspeech_gold_standard/metadata_ipa.csv
```

Train:

```console
./scripts/train_findtune.sh
# pass extra args as needed, e.g.:
./scripts/train_findtune.sh --model_name ivrit-ai/whisper-large-v3-turbo --num_train_epochs 6
```

## Training

See `src/train.py` for training.

### Resume from checkpoint

```console
./scripts/train_findtune.sh --resume_from_checkpoint ./checkpoints/whisper-heb-ipa/checkpoint-200
```

### Flash Attention 2

Enables faster training and lower VRAM usage. Enable with `--flash_attn`:

```console
./scripts/train_findtune.sh --flash_attn
```

Install the wheel first:

- **x86_64**: find prebuilt wheels at https://github.com/mjun0812/flash-attention-prebuild-wheels
- **aarch64 (ARM)**: find prebuilt wheels at https://pypi.jetson-ai-lab.io/sbsa/cu130

Validate installation:
```console
uv run python -c "import flash_attn; print(flash_attn.__version__)"
```

### Data loading

Audio is processed on-the-fly (no preprocessing cache). Benchmarked at ~400ms per batch of 16 on the training machine, vs ~4000ms per GPU step — dataloader is not a bottleneck. Uses `dataloader_num_workers=0` (the HuggingFace Audio column is not fork-safe with multiple workers).

## Monitor GPU

```console
uvx nvitop
```

## Monitor training progress

Either use wandb or tensorboard.

with tensorboard:

```console
uvx tensorboard --logdir checkpoints/whisper-heb-ipa
```

with wandb:

```console
uvx wandb login
uv run src/train.py --report_to wandb # it will print the URL to the wandb dashboard
```

## Sync tensorboard to wandb

```console
uvx wandb sync ./checkpoints/whisper-heb-ipa
```

## Upload model to HuggingFace

See `scripts/upload_model.sh`.

## Convert to CTranslate2

See `scripts/export_ct2.sh`.

## Gotchas

- https://huggingface.co/openai/whisper-large-v3/discussions/201
- To infer on macOS:

```console
uv pip uninstall torchcodec
uv run --no-sync src/infer.py
```
