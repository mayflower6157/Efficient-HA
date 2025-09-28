#!/usr/bin/env bash
set -euo pipefail

# ========= User Config =========
MODEL_ID="google/gemma-3-4b-it"
DATA_PATH="/mnt/disks/extra-disk/datasets/coco2014/val2014"
DEVICE="cuda:0"
MAX_TOKENS=64

# List of generation methods you want to run
METHODS=("dola")

# ========= Run Loop =========
for METHOD in "${METHODS[@]}"; do
    echo ">>> Running ${MODEL_ID} with method: ${METHOD}"

    python generate_response_chair.py \
        --model_id "$MODEL_ID" \
        --method "$METHOD" \
        --datapath "$DATA_PATH" \
        --device "$DEVICE" \
        --max_tokens "$MAX_TOKENS"

    echo ">>> Finished ${MODEL_ID} with method: ${METHOD}"
    echo
done
