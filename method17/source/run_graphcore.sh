#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method17/source
mkdir -p /home/test/graphcore_results
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$ROOT"
exec /home/test/graphcore-env/bin/python -u "$ROOT/run_graphcore.py"
