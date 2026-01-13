#!/bin/bash

# Load environment variables from .env file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: .env file not found at $ENV_FILE"
    echo "Please copy .env.example to .env and configure your settings:"
    echo "  cp .env.example .env"
    exit 1
fi

echo "Loading environment variables from .env file..."
set -a  # automatically export all variables
source "$ENV_FILE"
set +a  # stop automatically exporting

# Validate critical variables
if [ "$MODEL_PATH" = "/your/model/path" ] || [ -z "$MODEL_PATH" ]; then
    echo "Error: MODEL_PATH not configured in .env file"
    exit 1
fi

echo "==== start infer... ===="

cd "$( dirname -- "${BASH_SOURCE[0]}" )"

# Extract dataset name from DATASET path
DATASET_NAME=$(basename "$DATASET" | sed 's/\.jsonl$//' | sed 's/\.json$//')

# Create log directory: OUTPUT_PATH/LLM_MODEL/DATASET_NAME
LOG_DIR="$OUTPUT_PATH/$LLM_MODEL/$DATASET_NAME"
mkdir -p "$LOG_DIR"

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/${TIMESTAMP}_output.log"

python -u run.py \
--dataset "$DATASET" \
--output "$OUTPUT_PATH" \
--max_workers $MAX_WORKERS \
--model $MODEL_PATH \
--temperature $TEMPERATURE \
--presence_penalty $PRESENCE_PENALTY \
--total_splits ${WORLD_SIZE:-1} \
--worker_split $((${RANK:-0} + 1)) \
--roll_out_count $ROLLOUT_COUNT 2>&1 | tee -a "$LOG_FILE"
