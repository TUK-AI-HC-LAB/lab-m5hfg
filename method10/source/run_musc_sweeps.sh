#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 MPLBACKEND=Agg
ROOT=/mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method10/source
exec /home/test/miniforge3/envs/patchcore-gpu/bin/python -u "$ROOT/run_musc_sweeps.py"
