#!/usr/bin/env bash
set -euo pipefail
python=/home/test/miniforge3/envs/patchcore-gpu/bin/python
script=/mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method11/source/run_aprilgan_official.py
output=/home/test/aprilgan_results/mvtec_zero_shot_20261006
mkdir -p /home/test/aprilgan_results
"$python" -u "$script" --repo /home/test/VAND-APRIL-GAN --data /home/test/data/mvtec \
    --output "$output" --seed 42 2>&1 | tee "${output}.log"
