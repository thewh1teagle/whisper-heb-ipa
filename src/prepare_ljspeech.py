"""
uv run src/prepare_ljspeech.py --input_path saspeech_automatic/metadata.csv --output_path saspeech_automatic/metadata1.csv
"""

import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
args = parser.parse_args()

df = pd.read_csv(args.input_path, sep='\t', names=['file_id', 'text', 'phonemes'])
df[['file_id', 'phonemes']].to_csv(args.output_path, sep='|', header=False, index=False)
