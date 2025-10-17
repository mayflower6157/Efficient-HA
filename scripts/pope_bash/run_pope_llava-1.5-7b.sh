#!/usr/bin/env bash
set -euo pipefail

# ======== User Config ========
MODEL_ID="llava-hf/llava-1.5-7b-hf"
DATA_PATH="/home/li0007xu/EH/Efficient-HA/val2014"
DEVICE="cuda:0"
BATCH_SIZE=1
MAX_TOKENS=8
EARLY_EXIT_LAYERS=8

# Output directory
OUTPUT_DIR="./opera_log/pope_eval_results"

# Evaluation methods & POPE types
#METHODS=("beam" "greedy" "dola" "deco")
METHODS=("dola" "deco")
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
    mkdir -p "$RESP_DIR"

    echo
    echo "=============================================================="
    echo ">>> MODEL: ${MODEL_ID}"
    echo ">>> METHOD: ${METHOD}"
    echo ">>> POPE TYPE: ${POPE}"
    echo ">>> OUTPUT DIR: ${RESP_DIR}"
    echo "=============================================================="

    # Run Python evaluation (auto-resume + metrics handled inside)
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
      --resume_if_exists \
      # --silent

    echo "✅ Done: ${POPE} (${METHOD})"
    echo
  done
done