#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method20/source
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PYTHONUNBUFFERED=1
exec /home/test/miniforge3/envs/patchcore-gpu/bin/python run_spade.py
