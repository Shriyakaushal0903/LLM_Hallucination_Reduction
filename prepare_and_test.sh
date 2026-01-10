#!/usr/bin/env bash
set -e
MODEL_PATH="$(pwd)/checkpoints/local-model"

python - <<PY
from utils.lm import custom_api
prompt = "Write a short answer: what is the capital of France?"
print("Using model:", "$MODEL_PATH")
resp = custom_api(prompt, model_name_or_path="$MODEL_PATH", max_tokens=64, temperature=0.0, device=None)
print("=== RESPONSE ===")
print(resp)
PY

