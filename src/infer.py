
"""
Usage:
wget https://github.com/thewh1teagle/phonikud-chatterbox/releases/download/asset-files-v1/female1.wav -O example1.wav

# Run with default HF model
uv run src/infer.py

# Or run with local checkpoint
uv run src/infer.py --model ./whisper-heb-ipa/checkpoint-600

# Or with whisper small
uv run src/infer.py --model openai/whisper-small

# Or with thewh1teagle/whisper-heb-ipa
uv run src/infer.py --model thewh1teagle/whisper-heb-ipa
"""


import torch
from transformers import pipeline
import gradio as gr
import argparse
from pydub import AudioSegment
from pydub.effects import normalize
import tempfile
import os

def main():
    parser = argparse.ArgumentParser(description="Whisper Transcription Demo")
    parser.add_argument(
        "--model", 
        type=str, 
        default="thewh1teagle/whisper-heb-ipa",
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

    def normalize_audio(file_path):
        """Normalize audio using pydub to improve transcription quality."""
        try:
            # Load audio file
            audio = AudioSegment.from_file(file_path)
            
            # Normalize the audio (adjusts volume to optimal level)
            normalized_audio = normalize(audio)
            
            # Create a temporary file for the normalized audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                normalized_audio.export(temp_file.name, format="wav")
                return temp_file.name
        except Exception as e:
            print(f"Warning: Audio normalization failed: {e}")
            # Return original file if normalization fails
            return file_path

    def transcribe(file, task):
        # Normalize the audio before transcription
        normalized_file = normalize_audio(file)
        
        try:
            outputs = pipe(normalized_file, batch_size=BATCH_SIZE, generate_kwargs={"task": task})
            text = outputs["text"]
            return text
        finally:
            # Clean up temporary normalized file if it was created
            if normalized_file != file and os.path.exists(normalized_file):
                try:
                    os.unlink(normalized_file)
                except Exception as e:
                    print(f"Warning: Could not delete temporary file {normalized_file}: {e}")

    demo = gr.Blocks(
        css="""
        .large-textbox textarea {
            font-size: 20px !important;
            line-height: 1.6 !important;
        }
        """
    )

    mic_transcribe = gr.Interface(
        fn=transcribe,
        inputs=[
            gr.Audio(sources=["microphone", "upload"], type="filepath"),
            gr.Radio(["transcribe", "translate"], label="Task", value="transcribe"),
        ],
        outputs=gr.Textbox(
            label="Transcription", 
            lines=6, 
            max_lines=15, 
            min_width=400,
            show_copy_button=True,
            placeholder="Transcribed text will appear here...",
            elem_classes=["large-textbox"]
        ),
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
        outputs=gr.Textbox(
            label="Transcription", 
            lines=6, 
            max_lines=15, 
            min_width=400,
            show_copy_button=True,
            placeholder="Transcribed text will appear here...",
            elem_classes=["large-textbox"]
        ),
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

    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()