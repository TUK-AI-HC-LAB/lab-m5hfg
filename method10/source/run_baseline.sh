#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method10/source
OUTPUT=${MUSC_OUTPUT:-/home/test/musc_results/bottle_paper_20261002}
if [[ -f "$OUTPUT/metrics.csv" ]]; then
    echo "Completed output exists. Set MUSC_OUTPUT to a fresh run directory." >&2
    exit 1
fi
mkdir -p "$OUTPUT"
/home/test/miniforge3/envs/patchcore-gpu/bin/python -u "$ROOT/run_musc_local.py" --output "$OUTPUT" 2>&1 | tee "$OUTPUT/run.log"
