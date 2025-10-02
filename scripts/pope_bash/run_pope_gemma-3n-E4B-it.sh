#!/usr/bin/env bash
set -euo pipefail

# ========= User Config =========
MODEL_ID="google/gemma-3n-E4B-it"
DATA_PATH="/mnt/disks/extra-disk/datasets/coco2014/val2014"
DEVICE="cuda:0"
BATCH_SIZE=2
MAX_TOKENS=8

# Base output directory
OUTPUT_DIR="./opera_log/pope_eval_results"
#OUTPUT_DIR="./test"

# List of generation methods you want to run
METHODS=("beam")
#POPE_TYPES=("random")
#METHODS=("greedy" "beam")
POPE_TYPES=("random" "popular" "adversarial")

# ========= Run Loop =========
for METHOD in "${METHODS[@]}"; do
  for POPE in "${POPE_TYPES[@]}"; do
    RESP_DIR="${OUTPUT_DIR}/${MODEL_ID}/${METHOD}/"
    RESP_FILE="${RESP_DIR}/POPE_type_${POPE}_${METHOD}.jsonl"
    if [ ! -f "$RESP_FILE" ]; then
      echo ">>> Generating responses for ${MODEL_ID} with method: ${METHOD} and POPE type: ${POPE}"
      python generate_response_pope.py \
        --model_id "$MODEL_ID" \
        --method "$METHOD" \
        --pope_type "$POPE" \
        --batch_size "$BATCH_SIZE"       \
        --datapath "$DATA_PATH" \
        --device "$DEVICE" \
        --max_tokens "$MAX_TOKENS" \
        --output "$RESP_DIR"
      echo ">>> Finished generating responses"
    else
      echo ">>> Skipping generation: ${RESP_FILE} already exists"
    fi

    echo
  done
done
