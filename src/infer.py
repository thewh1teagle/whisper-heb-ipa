
"""
Usage:
wget https://github.com/thewh1teagle/phonikud-chatterbox/releases/download/asset-files-v1/female1.wav -O example1.wav

# Run with default HF model
uv run src/infer.py

# Or run with local checkpoint
uv run src/infer.py --model ./whisper-heb-ipa/checkpoint-600

# Or with whisper small
uv run src/infer.py --model openai/whisper-small
"""


import torch
from transformers import pipeline
import gradio as gr
import argparse

def main():
    parser = argparse.ArgumentParser(description="Whisper Transcription Demo")
    parser.add_argument(
        "--model", 
        type=str, 
        default="openai/whisper-small",
        help="Model name or path for Whisper (default: openai/whisper-small)"
    )
    args = parser.parse_args()
    
    MODEL_NAME = args.model
    BATCH_SIZE = 8

    device = 0 if torch.cuda.is_available() else "cpu"

    pipe = pipeline(
        task="automatic-speech-recognition",
        model=MODEL_NAME,
        chunk_length_s=30,
        device=device,
    )


    def transcribe(file, task):
        outputs = pipe(file, batch_size=BATCH_SIZE, generate_kwargs={"task": task})
        text = outputs["text"]
        return text

    demo = gr.Blocks()

    mic_transcribe = gr.Interface(
        fn=transcribe,
        inputs=[
            gr.Audio(sources=["microphone", "upload"], type="filepath"),
            gr.Radio(["transcribe", "translate"], label="Task", value="transcribe"),
        ],
        outputs="text",
        theme="huggingface",
        title="Whisper Demo: Transcribe Audio",
        description=(
            "Transcribe long-form microphone or audio inputs with the click of a button! Demo uses the"
            f" checkpoint [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) and 🤗 Transformers to transcribe audio files"
            " of arbitrary length."
        ),
        allow_flagging="never",
    )

    file_transcribe = gr.Interface(
        fn=transcribe,
        inputs=[
            gr.Audio(sources=["upload"], label="Audio file", type="filepath"),
            gr.Radio(["transcribe", "translate"], label="Task", value="transcribe"),
        ],
        outputs="text",
        theme="huggingface",
        title="Whisper Demo: Transcribe Audio",
        description=(
            "Transcribe long-form microphone or audio inputs with the click of a button! Demo uses the"
            f" checkpoint [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) and 🤗 Transformers to transcribe audio files"
            " of arbitrary length."
        ),
        examples=[
            ["./example1.wav", "transcribe"],
        ],
        cache_examples=True,
        allow_flagging="never",
    )

    with demo:
        gr.TabbedInterface([file_transcribe, mic_transcribe], ["Transcribe Audio File", "Transcribe Microphone"])

    demo.launch()


if __name__ == "__main__":
    main()