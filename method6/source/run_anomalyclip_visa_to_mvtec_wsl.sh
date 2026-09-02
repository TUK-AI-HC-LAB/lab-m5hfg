#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
PYTHON="/home/test/miniforge3/envs/patchcore-gpu/bin/python"
CODE_ROOT="/home/test/anomalyclip"
VISA_ROOT="/home/test/data/VisA_20220922"
MVTEC_ROOT="/home/test/data/mvtec"
CHECKPOINT_ROOT="/home/test/anomalyclip_checkpoints/visa_to_mvtec"
RESULT_ROOT="/mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method6/source/results/anomalyclip_visa_to_mvtec"

mkdir -p "$CHECKPOINT_ROOT" "$RESULT_ROOT"
cd "$CODE_ROOT"

prepare() {
  "$PYTHON" /mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method6/source/generate_anomalyclip_metadata.py \
    --visa-root "$VISA_ROOT" \
    --mvtec-root "$MVTEC_ROOT"
}

train() {
  "$PYTHON" train.py \
    --dataset visa \
    --train_data_path "$VISA_ROOT" \
    --save_path "$CHECKPOINT_ROOT" \
    --features_list 24 \
    --image_size 518 \
    --batch_size 8 \
    --epoch 15 \
    --save_freq 1 \
    --print_freq 1 \
    --depth 9 \
    --n_ctx 12 \
    --t_n_ctx 4 \
    --seed 111
}

test() {
  "$PYTHON" test.py \
    --dataset mvtec \
    --data_path "$MVTEC_ROOT" \
    --save_path "$RESULT_ROOT" \
    --checkpoint_path "$CHECKPOINT_ROOT/epoch_15.pth" \
    --features_list 24 \
    --image_size 518 \
    --depth 9 \
    --n_ctx 12 \
    --t_n_ctx 4 \
    --seed 111 \
    --sigma 4 \
    --metrics image-pixel-level
}

case "$MODE" in
  prepare)
    prepare
    ;;
  train)
    train
    ;;
  test)
    test
    ;;
  all)
    prepare
    train
    test
    ;;
  *)
    echo "Usage: $0 {prepare|train|test|all}" >&2
    exit 2
    ;;
esac
