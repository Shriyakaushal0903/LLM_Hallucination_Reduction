#!/usr/bin/env bash
set -e

# path to the hallulens repo root
HALLULENS_DIR="${PWD}/hallulens"
# path to your local HF-style model
LOCAL_MODEL_PATH="${PWD}/checkpoints/local-model"

# environment (adjust as needed)
export PYTHONPATH="$HALLULENS_DIR:$PYTHONPATH"
export MODEL_LOCAL_PATH="$LOCAL_MODEL_PATH"

# run a sample task script from HalluLens (choose one script from the repo)
# adjusted to pass model path to python environment variables or to the script if it accepts args
cd "$HALLULENS_DIR"

python -m scripts.run_task --task precisewikiqa --model_path "$LOCAL_MODEL_PATH" --device "cuda"
