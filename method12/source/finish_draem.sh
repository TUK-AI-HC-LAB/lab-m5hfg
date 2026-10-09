#!/usr/bin/env bash
set -euo pipefail
SOURCE=/mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method12/source
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/home/test/miniforge3/envs/patchcore-gpu/bin/python
"$PY" -u "$SOURCE/evaluate_rscin.py"
"$PY" -u "$SOURCE/verify_draem.py"
"$PY" -u "$SOURCE/build_report.py"
"$PY" -u /mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method10/source/build_musc_reproduction_tables.py
