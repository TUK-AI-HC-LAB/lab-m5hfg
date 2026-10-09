#!/usr/bin/env bash
set -euo pipefail
SOURCE=/mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method15/source
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/home/test/acr-env/bin/python -u "$SOURCE/run_acr.py"
