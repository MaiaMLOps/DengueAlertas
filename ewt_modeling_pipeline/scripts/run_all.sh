#!/usr/bin/env bash
set -euo pipefail
INPUT="${1:-data/processed/pilot_municipalities_weekly.parquet}"
OUTPUT="${2:-outputs/final_run}"
python -m pytest -q
python run_pipeline.py --input "$INPUT" --config configs/pipeline.yaml --output "$OUTPUT"
