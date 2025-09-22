# whisper-heb-ipa

Transcribe Hebrew speech into IPA using a fine-tuned Whisper model.

This project is part of the [phonikud.github.io](https://phonikud.github.io) project.

## Data preparation

See `src/prepare.py` for data preparation.

## Training

See `src/train.py` for training.

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