#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method18/source
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MPLBACKEND=Agg
cd "$ROOT"
exec /home/test/vtadl-env/bin/python -u "$ROOT/run_vtadl.py"
