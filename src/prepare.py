"""
wget https://huggingface.co/datasets/thewh1teagle/ILSpeech/resolve/main/speaker1/ilspeech_speaker1_v1.7z
wget https://huggingface.co/datasets/thewh1teagle/ILSpeech/resolve/main/speaker2/ilspeech_speaker2_v1.7z
7z x ilspeech_speaker1_v1.7z
7z x ilspeech_speaker2_v1.7z

uv run src/prepare.py --input_folder ilspeech_speaker1_v1 ilspeech_speaker2_v1 --output_folder data
"""

import argparse
import pandas as pd
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input_folder", type=str, nargs="+", required=True)
parser.add_argument("--output_folder", type=str, required=True)
args = parser.parse_args()

def main():
    output_path = Path(args.output_folder)
    wav_output_path = output_path / "wav"
    wav_output_path.mkdir(parents=True, exist_ok=True)
    
    combined_data = []
    counter = 0
    
    for input_folder in args.input_folder:
        input_path = Path(input_folder)
        df = pd.read_csv(input_path / "metadata.csv", sep='|', header=None, names=['index', 'ipa'])
        
        for _, row in df.iterrows():
            original_wav = input_path / "wav" / f"{row['index']}.wav"
            new_wav = wav_output_path / f"{counter}.wav"
            
            if original_wav.exists():
                os.link(original_wav, new_wav)
                combined_data.append(f"{counter}|{row['ipa']}")
                counter += 1
    
    # Save combined metadata
    with open(output_path / "metadata.csv", "w") as f:
        f.write("\n".join(combined_data))
    
    print(f"Total files processed: {len(combined_data)}")


if __name__ == "__main__":
    main()
