#!/usr/bin/env bash
set -euo pipefail

# ======== User Config ========
MODEL_ID="llava-hf/llava-1.5-7b-hf"
DATA_PATH="/mnt/disks/extra-disk/datasets/coco2014/val2014"
DEVICE="cuda:0"
BATCH_SIZE=16
MAX_TOKENS=8
EARLY_EXIT_LAYERS=10

# Base output directory
OUTPUT_DIR="./opera_log/pope_eval_results"

# List of generation methods you want to run
METHODS=("beam" "greedy" "dola" "deco")

# POPE types
POPE_TYPES=("random" "popular" "adversarial")

# ======== Run Loop ========
for METHOD in "${METHODS[@]}"; do
  for POPE in "${POPE_TYPES[@]}"; do

    # Directory structure logic
    if [[ "$METHOD" == "dola" || "$METHOD" == "deco" ]]; then
      RESP_DIR="${OUTPUT_DIR}/${MODEL_ID}/${EARLY_EXIT_LAYERS}_layers/${METHOD}/"
    else
      RESP_DIR="${OUTPUT_DIR}/${MODEL_ID}/${METHOD}/"
    fi

    RESP_FILE="${RESP_DIR}/pope_${POPE}_${METHOD}.jsonl"

    # Make sure directories exist
    mkdir -p "$RESP_DIR"

    # Step 1: Generate responses if not already present
    if [ ! -f "$RESP_FILE" ]; then
      echo ">>> Generating responses for ${MODEL_ID} with method: ${METHOD} and POPE type: ${POPE}"
      python generate_response_pope.py \
        --model_id "$MODEL_ID" \
        --method "$METHOD" \
        --pope_type "$POPE" \
        --batch_size "$BATCH_SIZE" \
        --datapath "$DATA_PATH" \
        --device "$DEVICE" \
        --max_tokens "$MAX_TOKENS" \
        --output "$RESP_FILE"
      echo ">>> Finished generating responses"
    else
      echo ">>> Skipping: ${RESP_FILE} already exists"
    fi

    echo
  done
done