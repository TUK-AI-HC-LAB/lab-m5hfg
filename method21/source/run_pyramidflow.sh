#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method21/source
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
exec /home/test/miniforge3/envs/patchcore-gpu/bin/python run_pyramidflow.py
