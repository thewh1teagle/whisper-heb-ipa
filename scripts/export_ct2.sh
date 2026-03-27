#!/bin/bash
git clone https://huggingface.co/thewh1teagle/whisper-heb-ipa
uv pip install 'ctranslate2>=4.6.0'
uv run ct2-transformers-converter \
    --model ./whisper-heb-ipa \
    --output_dir ./whisper-heb-ipa-ct2 \
    --quantization int8_float16
