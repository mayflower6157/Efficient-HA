#!/usr/bin/env bash
set -euo pipefail

# ========= User Config =========
MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct"
DATA_PATH="/mnt/disks/extra-disk/datasets/coco2014/val2014"
DEVICE="cuda:0"
MAX_TOKENS=10

# Base output directory
OUTPUT_DIR="./opera_log/pope_eval_results"

# List of generation methods you want to run
METHODS=("greedy")
POPE_TYPES=("random")
#METHODS=("greedy" "beam")
#POPE_TYPES=("random" "popular" "adversarial")

# ========= Run Loop =========
for METHOD in "${METHODS[@]}"; do
  for POPE in "${POPE_TYPES[@]}"; do
    RESP_FILE="${OUTPUT_DIR}/${MODEL_ID}/${POPE}/${METHOD}/responses.json"

    if [ ! -f "$RESP_FILE" ]; then
      echo ">>> Generating responses for ${MODEL_ID} with method: ${METHOD} and POPE type: ${POPE}"
      python generate_response_pope.py \
        --model_id "$MODEL_ID" \
        --method "$METHOD" \
        --datapath "$DATA_PATH" \
        --device "$DEVICE" \
        --max_tokens "$MAX_TOKENS" \
        --output "$RESP_FILE"
      echo ">>> Finished generating responses"
    else
      echo ">>> Skipping generation: ${RESP_FILE} already exists"
    fi

    echo
  done
done