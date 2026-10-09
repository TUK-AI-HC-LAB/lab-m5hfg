#!/usr/bin/env bash
set -euo pipefail
SOURCE=/mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method12/source
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
/home/test/miniforge3/envs/patchcore-gpu/bin/python -u "$SOURCE/run_draem.py"
