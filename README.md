# whisper-heb-ipa

Transcribe Hebrew speech into IPA using a fine-tuned Whisper model.

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

## References

- ivrit.ai whisper turbo https://huggingface.co/ivrit-ai/whisper-large-v3-turbo/tree/main
- huggingface how to fine tune whisper https://huggingface.co/blog/fine-tune-whisper