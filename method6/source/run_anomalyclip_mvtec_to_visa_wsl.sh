#!/usr/bin/env bash
# Reproduce AnomalyCLIP's MVTec AD source -> VisA target protocol in WSL.
# Usage: bash run_anomalyclip_mvtec_to_visa_wsl.sh [train|test|all]
set -euo pipefail

MODE="${1:-all}"
PYTHON="/home/test/miniforge3/envs/patchcore-gpu/bin/python"
CODE_ROOT="/home/test/anomalyclip"
MVTEC_ROOT="/home/test/data/mvtec"
VISA_ROOT="/home/test/data/VisA_20220922"
CHECKPOINT_ROOT="/home/test/anomalyclip_checkpoints/mvtec_to_visa"
RESULT_ROOT="/mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method6/source/results/anomalyclip_mvtec_to_visa"

train() {
  mkdir -p "${CHECKPOINT_ROOT}"
  cd "${CODE_ROOT}"
  CUDA_VISIBLE_DEVICES=0 "${PYTHON}" train.py \
    --dataset mvtec --train_data_path "${MVTEC_ROOT}" --save_path "${CHECKPOINT_ROOT}" \
    --features_list 24 --image_size 518 --batch_size 8 --epoch 15 --save_freq 1 --print_freq 1 \
    --depth 9 --n_ctx 12 --t_n_ctx 4 --seed 111
}

test() {
  mkdir -p "${RESULT_ROOT}"
  cd "${CODE_ROOT}"
  CUDA_VISIBLE_DEVICES=0 "${PYTHON}" test.py \
    --dataset visa --data_path "${VISA_ROOT}" --save_path "${RESULT_ROOT}" \
    --checkpoint_path "${CHECKPOINT_ROOT}/epoch_15.pth" \
    --features_list 24 --image_size 518 --depth 9 --n_ctx 12 --t_n_ctx 4 --seed 111 \
    --sigma 4 --metrics image-pixel-level
}

case "${MODE}" in
  train) train ;;
  test) test ;;
  all) train; test ;;
  *) echo "Usage: $0 [train|test|all]" >&2; exit 2 ;;
esac
