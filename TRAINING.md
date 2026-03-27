# Training

## Data preparation

See `src/prepare.py` for data preparation.

## Training on SASpeech

Download and extract:

```console
wget https://openslr.trmal.net/resources/134/saspeech_gold_standard_v1.0.tar.gz
tar xf saspeech_gold_standard_v1.0.tar.gz
```

Clean and prepare the transcripts (strips nikud, normalizes punctuation):

```console
uv run scripts/prepare_saspeech.py saspeech_gold_standard/metadata.csv > saspeech_gold_standard/metadata_clean.csv
```

Phonemize (produces `metadata_ipa.csv` with `id|phonemes`):

```console
wget https://huggingface.co/thewh1teagle/renikud/resolve/main/model.onnx -O renikud.onnx
uv run scripts/phonemize.py renikud.onnx saspeech_gold_standard/metadata_clean.csv > saspeech_gold_standard/metadata_ipa.csv
```

Train:

```console
./scripts/train_findtune.sh
# pass extra args as needed, e.g.:
./scripts/train_findtune.sh --model_name ivrit-ai/whisper-large-v3-turbo --max_steps 5000
```

## Training

See `src/train.py` for training.

### Resume from checkpoint

```console
./scripts/train_findtune.sh --resume_from_checkpoint ./whisper-heb-ipa/checkpoint-200
```

### Data loading

Audio is processed on-the-fly (no preprocessing cache). Benchmarked at ~400ms per batch of 16 on the training machine, vs ~4000ms per GPU step — dataloader is not a bottleneck. Uses `dataloader_num_workers=4` by default so batches are prefetched in parallel with GPU training.

## Monitor GPU

```console
uvx nvitop
```

## Monitor training progress

Either use wandb or tensorboard.

with tensorboard:

```console
uvx tensorboard --logdir whisper-heb-ipa
```

with wandb:

```console
uvx wandb login
uv run src/train.py --report_to wandb # it will print the URL to the wandb dashboard
```

## Sync tensorboard to wandb

```console
uvx wandb sync ./whisper-heb-ipa
```

## Upload model to HuggingFace

See `scripts/upload_model.sh`.

## Convert to CTranslate2

See `scripts/export_ct2.sh`.
