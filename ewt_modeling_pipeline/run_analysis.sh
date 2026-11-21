#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-outputs/final_run}"
OUTPUT_DIR="${2:-${RUN_DIR}/analysis}"

python analyze_ewt_results.py \
  --run-dir "$RUN_DIR" \
  --output "$OUTPUT_DIR" \
  --formats all \
  --top-features 15 \
  --dpi 180 \
  --overwrite
