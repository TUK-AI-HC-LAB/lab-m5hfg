#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method19/source
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MPLBACKEND=Agg
cd "$ROOT"
exec /home/test/psvdd-env/bin/python -u "$ROOT/run_psvdd.py"
