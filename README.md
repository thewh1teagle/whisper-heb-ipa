# whisper-heb-ipa

Transcribe Hebrew speech into IPA using a fine-tuned Whisper model.

## Data preparation (SASpeech manual v2) ⭐

```console
# Prepare repository on saspeech1 branch
git clone https://github.com/thewh1teagle/whisper-heb-ipa -b saspeech1

# Download data
wget https://huggingface.co/datasets/thewh1teagle/saspeech/resolve/main/saspeech_manual/saspeech_manual_v2.7z
wget https://huggingface.co/datasets/thewh1teagle/saspeech/resolve/main/saspeech_automatic/saspeech_automatic.7z

# Extract data
sudo apt install p7zip-full -y
7z x saspeech_manual_v2.7z
7z x saspeech_automatic.7z

# Convert to LJSpeech format
uv run src/prepare_ljspeech.py --input_path saspeech_automatic/metadata.csv --output_path saspeech_automatic/metadata1.csv
uv run src/prepare_ljspeech.py --input_path saspeech_manual/metadata.csv --output_path saspeech_manual/metadata1.csv

# Rename metadata
mv saspeech_automatic/metadata.csv saspeech_automatic/metadata.old.csv
mv saspeech_manual/metadata.csv saspeech_manual/metadata.old.csv
mv saspeech_automatic/metadata1.csv saspeech_automatic/metadata.csv
mv saspeech_manual/metadata1.csv saspeech_manual/metadata.csv

# Optional: combine datasets to one folder
uv run src/prepare.py --input_folder saspeech_manual saspeech_automatic --output_folder saspeech_data/
```

## Training 

See `src/train.py` for training.

Example training ⭐

```console
uv run src/train.py ... --data_dir saspeech_data/ --dataset_cache_path ./saspeech_data_cache
uv run src/train.py ... --data_dir ilspeech_data/ --dataset_cache_path ./ilspeech_data_cache
```

## Inference

See `src/infer.py` for inference.


The model is fine-tuned on the ILSpeech dataset.

## Monitor GPU

```console
uv pip install nvitop
uv run nvitop
```

## Monitor training progress

Either use wandb or tensorboard.

with tensorboard:

```console
uv run tensorboard --logdir whisper-heb-ipa
```

with wandb:

```console
uv run wandb login
uv run src/train.py --report_to wandb # it will print the URL to the wandb dashboard
```

## Sync tensorboard to wandb

```console
uv run wandb sync ./whisper-heb-ipa
```

## Upload/Download dadtaset cache

```console
uv run hf upload --repo-type dataset thewh1teagle/whisper-heb-ipa-dataset ./dataset_cache
uv run hf download --repo-type dataset thewh1teagle/whisper-heb-ipa-dataset --local-dir ./dataset_cache
```

## Upload model to HuggingFace

```console
uv run hf upload --repo-type model thewh1teagle/whisper-heb-ipa ./whisper-heb-ipa/checkpoint-9000
```

## Convert to CTransalte2

```console
git clone https://huggingface.co/thewh1teagle/whisper-heb-ipa
uv pip install 'ctranslate2>=4.6.0'
uv run ct2-transformers-converter \
    --model ./whisper-heb-ipa \
    --output_dir ./whisper-heb-ipa-ct2 \
    --quantization int8_float16
uv run hf upload --repo-type model thewh1teagle/whisper-heb-ipa-ct2 ./whisper-heb-ipa-ct2
```

## References

- ivrit.ai whisper turbo https://huggingface.co/ivrit-ai/whisper-large-v3-turbo/tree/main
- huggingface how to fine tune whisper https://huggingface.co/blog/fine-tune-whisper
- https://medium.com/@balaragavesh/fine-tuning-whisper-to-predict-phonemes-from-audio-using-hugging-face-transformers-babbb46a9f05

## Gotchas

- https://huggingface.co/openai/whisper-large-v3/discussions/201
- To infer on macOS:

```console
uv pip uninstall torchcodec
uv run --no-sync src/infer.py
```