#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 MPLBACKEND=Agg
exec /home/test/igd-env/bin/python -u /mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method14/source/run_igd.py
