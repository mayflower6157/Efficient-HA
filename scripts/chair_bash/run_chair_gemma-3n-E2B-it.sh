#!/usr/bin/env bash
set -euo pipefail

# ======== User Config ========
MODEL_ID="google/gemma-3n-E2B-it"
DATA_PATH="/mnt/disks/extra-disk/datasets/coco2014/val2014"
COCO_ANN="/mnt/disks/extra-disk/datasets/coco2014/annotations"
DEVICE="cuda:0"
MAX_TOKENS=64
EARLY_EXIT_LAYERS=10

# Base output directory
OUTPUT_DIR="./opera_log/chair_eval_results"

# List of generation methods you want to run
METHODS=("deco")

# ======== Run Loop ========
for METHOD in "${METHODS[@]}"; do
  # Directory structure: add <LAYERS> folder only for dola/deco
  if [[ "$METHOD" == "dola" || "$METHOD" == "deco" ]]; then
    RESP_FILE="${OUTPUT_DIR}/${MODEL_ID}/${EARLY_EXIT_LAYERS}_layers/${METHOD}/responses_fixed.json"
    METRIC_FILE="${OUTPUT_DIR}/${MODEL_ID}/${EARLY_EXIT_LAYERS}_layers/${METHOD}/metric_fixed.json"
  else
    RESP_FILE="${OUTPUT_DIR}/${MODEL_ID}/${METHOD}/responses.json"
    METRIC_FILE="${OUTPUT_DIR}/${MODEL_ID}/${METHOD}/metric.json"
  fi

  # Make sure directories exist
  mkdir -p "$(dirname "$RESP_FILE")"

  # Step 1: Generate responses if not already present
  if [ ! -f "$RESP_FILE" ]; then
    echo ">>> Generating responses for ${MODEL_ID} with method: ${METHOD}"
    python generate_response_chair.py \
      --model_id "$MODEL_ID" \
      --method "$METHOD" \
      --datapath "$DATA_PATH" \
      --device "$DEVICE" \
      --max_tokens "$MAX_TOKENS" \
      --output "$RESP_FILE"
  fi

  # Step 2: Run CHAIR eval if not already present
  if [ ! -f "$METRIC_FILE" ]; then
    echo ">>> Running CHAIR eval for ${MODEL_ID} with method: ${METHOD}"
    python chair.py \
      --cap_file "$RESP_FILE" \
      --image_id_key image_id \
      --caption_key response \
      --coco_path "$COCO_ANN" \
      --save_path "$METRIC_FILE"
    echo ">>> Finished CHAIR eval"
  else
    echo ">>> Skipping CHAIR eval: ${METRIC_FILE} already exists"
  fi

  echo
done