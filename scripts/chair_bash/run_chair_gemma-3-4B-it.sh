#!/usr/bin/env bash
set -euo pipefail

# ======== User Config ========
MODEL_ID="google/gemma-3-4b-it"
DATA_PATH="/home/li0007xu/EH/Efficient-HA/val2014"
COCO_ANN="/home/li0007xu/Reasoning/Deco/annotations_2014_coco"
DEVICE="cuda:4"
MAX_TOKENS=64
EARLY_EXIT_LAYERS=10

# Base output directory
OUTPUT_DIR="./opera_log/chair_eval_results"

# List of runs.
# Format per entry: method|layers|variant
#   - leave layers empty for methods without early-exit layers
#   - variant is optional (e.g., fixed, improv)
RUN_CONFIGS=(
  "deco|${EARLY_EXIT_LAYERS}|alpha-schedule"
  # "deco|${EARLY_EXIT_LAYERS}|fixed"
  # "deco|${EARLY_EXIT_LAYERS}|improv"
  # "deco|8|improv"
  # "dola|${EARLY_EXIT_LAYERS}|"
  # "greedy||"
  # "beam||"
)

# ======== Run Loop ========
for CONFIG in "${RUN_CONFIGS[@]}"; do
  IFS='|' read -r METHOD LAYER_OVERRIDE VARIANT <<< "$CONFIG"
  [[ -n "$METHOD" ]] || continue

  RUN_LAYERS="${LAYER_OVERRIDE:-$EARLY_EXIT_LAYERS}"
  RUN_DIR="${OUTPUT_DIR}/${MODEL_ID}"
  if [[ -n "$LAYER_OVERRIDE" ]]; then
    RUN_DIR+="/${METHOD}-${LAYER_OVERRIDE}-layers"
  else
    RUN_DIR+="/${METHOD}"
  fi

  if [[ -n "$VARIANT" && "$VARIANT" != "default" ]]; then
    RUN_DIR+="-${VARIANT}"
  fi

  RESP_FILE="${RUN_DIR}/responses.json"
  METRIC_FILE="${RUN_DIR}/metrics.json"

  mkdir -p "$RUN_DIR"

  # Step 1: Generate responses if not already present
  if [ ! -f "$RESP_FILE" ]; then
    echo ">>> Generating responses for ${MODEL_ID} with method: ${METHOD}"
    python generate_response_chair.py \
      --model_id "$MODEL_ID" \
      --method "$METHOD" \
      --datapath "$DATA_PATH" \
      --device "$DEVICE" \
      --max_tokens "$MAX_TOKENS" \
      --early_exit_layers "$RUN_LAYERS" \
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
