#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/home/test/regad-env/bin/python -u /mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method16/source/run_regad.py
