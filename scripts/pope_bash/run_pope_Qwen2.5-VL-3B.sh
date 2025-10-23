#!/usr/bin/env bash
set -euo pipefail

# ======== User Config ========
MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct"
DATA_PATH="/home/li0007xu/EH/Efficient-HA/val2014"
DEVICE="cuda:0"
BATCH_SIZE=1
MAX_TOKENS=8
EARLY_EXIT_LAYERS=10

# Base output directory
OUTPUT_DIR="./opera_log/pope_eval_results"

# List of generation methods you want to run
METHODS=("deco")

# POPE types
POPE_TYPES=("random" "popular" "adversarial")

# ======== Run Loop ========
for METHOD in "${METHODS[@]}"; do
  for POPE in "${POPE_TYPES[@]}"; do

    # Directory structure logic
    if [[ "$METHOD" == "dola" || "$METHOD" == "deco" ]]; then
      RESP_DIR="${OUTPUT_DIR}/${MODEL_ID}/${EARLY_EXIT_LAYERS}_layers/${METHOD}"
    else
      RESP_DIR="${OUTPUT_DIR}/${MODEL_ID}/${METHOD}"
    fi

    RESP_FILE="${RESP_DIR}/POPE_type_${POPE}_${METHOD}.jsonl"

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
        --early_exit_layers "$EARLY_EXIT_LAYERS" \
        --output "$RESP_DIR" \
        # --debug 
        # --silent
      echo ">>> Finished generating responses"
    else
      echo ">>> Skipping: ${RESP_FILE} already exists"
    fi

    echo
  done
done
