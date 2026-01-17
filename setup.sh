#!/bin/bash
set -e

MODEL_ID="ministral/Ministral-3b-instruct"
MODEL_DIR="./models/Ministral-3b-instruct"
VENV_DIR="./.venv"

echo "Downloading $MODEL_ID to $MODEL_DIR"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python python3 "$VENV_DIR"
source "$VENV_DIR/bin/activate"

uv pip install -U huggingface_hub

if [ -n "${HF_TOKEN:-}" ]; then
  echo "$HF_TOKEN" | hf auth login --token --stdin
fi

mkdir -p "$MODEL_DIR"
hf download "$MODEL_ID" --local-dir "$MODEL_DIR"

echo "Done. Model in $MODEL_DIR"