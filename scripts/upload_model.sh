#!/bin/bash
REPO=${1:-thewh1teagle/whisper-heb-ipa}
CHECKPOINT=${2:-./whisper-heb-ipa/checkpoint-9000}
uv run hf upload --repo-type model "$REPO" "$CHECKPOINT"
