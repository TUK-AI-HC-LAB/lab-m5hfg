#!/usr/bin/env bash
set -euo pipefail
SOURCE=/mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method14/source
bash "$SOURCE/prepare_igd.sh"
if [ ! -x /home/test/igd-compiler/bin/x86_64-conda-linux-gnu-g++ ]; then
  if [ -d /home/test/igd-compiler ]; then
    /home/test/miniforge3/bin/conda install -y -p /home/test/igd-compiler --override-channels -c conda-forge gcc_linux-64 gxx_linux-64
  else
    /home/test/miniforge3/bin/conda create -y -p /home/test/igd-compiler --override-channels -c conda-forge gcc_linux-64 gxx_linux-64
  fi
fi
export CC=/home/test/igd-compiler/bin/x86_64-conda-linux-gnu-gcc
export CXX=/home/test/igd-compiler/bin/x86_64-conda-linux-gnu-g++
export PATH="/home/test/igd-compiler/bin:$PATH"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export IGD_RUN_TAG=mvtec_batch8_seed42_20261007
/home/test/igd-env/bin/python "$SOURCE/check_loss_compile.py"
