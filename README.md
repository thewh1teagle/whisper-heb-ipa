# whisper-heb-ipa

Transcribe Hebrew speech into IPA using a fine-tuned Whisper model.

This project is part of the [phonikud.github.io](https://phonikud.github.io) project.

Note: for the most accurate model use this link [whisper-heb-ipa-large-v3-turbo-ct2](https://huggingface.co/thewh1teagle/whisper-heb-ipa-large-v3-turbo-ct2)

## Inference

See `src/infer.py` for inference.

## Training

See [TRAINING.md](TRAINING.md).

## Gotchas

- https://huggingface.co/openai/whisper-large-v3/discussions/201
- To infer on macOS:

```console
uv pip uninstall torchcodec
uv run --no-sync src/infer.py
```
