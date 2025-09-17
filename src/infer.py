"""
wget https://github.com/thewh1teagle/phonikud-chatterbox/releases/download/asset-files-v1/female1.wav
uv run src/infer.py
"""

from transformers import pipeline
import gradio as gr

pipe = pipeline(model="thewh1teagle/whisper-heb-ipa")

def transcribe(audio):
    text = pipe(audio)["text"]
    return text

iface = gr.Interface(
    fn=transcribe, 
    inputs=gr.Audio(source="microphone", type="filepath"), 
    outputs="text",
    title="Whisper Hebrew IPA",
    description="Realtime demo for Hebrew speech recognition using a fine-tuned Whisper Hebrew IPA model.",
)

iface.launch()
