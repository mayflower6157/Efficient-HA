#!/bin/bash
echo ">>> Starting bash1"
bash ./scripts/pope_bash/run_pope_gemma-3n-E4B-it.sh > gemma-3n-E4B.log 2>&1

echo ">>> bash1 finished, starting bash2"
bash ./scripts/pope_bash/run_pope_gemma-3n-E2B-it.sh > gemma-3n-E2B.log 2>&1

echo ">>> All jobs finished"
