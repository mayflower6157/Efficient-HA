#!/usr/bin/env bash
set -euo pipefail

# ========= User Config =========
MODEL_ID="llava-hf/llava-1.5-7b-hf"
DATA_PATH="/mnt/disks/extra-disk/datasets/coco2014/val2014"
DEVICE="cuda:0"
MAX_TOKENS=64

# Base output directory
OUTPUT_DIR="./opera_log/chair_eval_results"

# List of generation methods you want to run
METHODS=("greedy" "beam" )

# ========= Run Loop =========
for METHOD in "${METHODS[@]}"; do
    echo ">>> Running ${MODEL_ID} with method: ${METHOD}"

    python generate_response_chair.py \
        --model_id "$MODEL_ID" \
        --method "$METHOD" \
        --datapath "$DATA_PATH" \
        --device "$DEVICE" \
        --max_tokens "$MAX_TOKENS" \
        --output "${OUTPUT_DIR}/${MODEL_ID}/${METHOD}/responses.json"

    echo ">>> Finished ${MODEL_ID} with method: ${METHOD}"
    echo
done
