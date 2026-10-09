#!/usr/bin/env bash
set -euo pipefail
export IGD_RUN_TAG=mvtec_batch8_seed42_20261007 IGD_ACCELERATED=1
export IGD_LOCAL_BATCH=8 IGD_GLOBAL_BATCH=16
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_COMPILE_THREADS=2
export CC=/home/test/igd-compiler/bin/x86_64-conda-linux-gnu-gcc
export CXX=/home/test/igd-compiler/bin/x86_64-conda-linux-gnu-g++
export PATH="/home/test/igd-compiler/bin:$PATH"
exec /home/test/igd-env/bin/python -u /mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method14/source/run_igd.py
