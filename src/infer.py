"""
Usage:
wget https://github.com/thewh1teagle/phonikud-chatterbox/releases/download/asset-files-v1/female1.wav

# Run with default HF model
uv run src/infer.py

# Or run with local checkpoint
uv run src/infer.py --model_name ./whisper-heb-ipa/checkpoint-200
"""

import argparse
import numpy as np
from transformers import pipeline
import gradio as gr
import librosa

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name", 
    type=str, 
    default="thewh1teagle/whisper-heb-ipa",
    help="Model name on Hugging Face hub or path to local checkpoint"
)
args = parser.parse_args()

# Force task to avoid HFValidationError on local checkpoints
pipe = pipeline(
    task="automatic-speech-recognition",
    model=args.model_name
)

def transcribe(audio):
    if audio is None:
        return ""
    sr, data = audio
    # Ensure float32 for librosa
    data = np.array(data).astype(np.float32)

    # Resample to 16k for Whisper
    if sr != 16000:
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        sr = 16000

    # Limit to 30s
    max_len = sr * 30
    if len(data) > max_len:
        data = data[:max_len]

    result = pipe({"array": data, "sampling_rate": sr})
    return result["text"]

iface = gr.Interface(
    fn=transcribe, 
    inputs=gr.Audio(sources=["microphone", "upload"], type="numpy"),
    outputs="text",
    title="Whisper Hebrew IPA",
    description="Realtime demo for Hebrew speech recognition using a fine-tuned Whisper Hebrew IPA model.",
)

iface.launch()
