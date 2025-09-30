#!/usr/bin/env bash
set -euo pipefail

# ========= User Config =========
MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct"
DATA_PATH="/mnt/disks/extra-disk/datasets/coco2014/val2014"
COCO_ANN="/mnt/disks/extra-disk/datasets/coco2014/annotations"
DEVICE="cuda:0"
MAX_TOKENS=64

# Base output directory
OUTPUT_DIR="./opera_log/chair_eval_results"

# List of generation methods you want to run
METHODS=("greedy" "beam")

# ========= Run Loop =========
for METHOD in "${METHODS[@]}"; do
    RESP_FILE="${OUTPUT_DIR}/${MODEL_ID}/${METHOD}/responses.json"
    METRIC_FILE="${OUTPUT_DIR}/${MODEL_ID}/${METHOD}/metric.json"

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
        echo ">>> Finished generating responses"
    else
        echo ">>> Skipping generation: ${RESP_FILE} already exists"
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