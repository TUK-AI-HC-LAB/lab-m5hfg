#!/usr/bin/env bash
# Reproduce WinCLIP's VisA zero- and few-normal-shot protocols in WSL.
# Usage: bash run_winclip_visa_wsl.sh [zero|one|two|four|all]
set -euo pipefail

MODE="${1:-all}"
PYTHON="/home/test/miniforge3/envs/patchcore-gpu/bin/python"
CODE_ROOT="/home/test/accurate-winclip"
DATA_ROOT="/home/test/data/VisA_20220922"
RESULT_ROOT="/mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method7/source/results/winclip_visa"

run_shot() {
  local shot="$1"
  local name="$2"
  mkdir -p "${RESULT_ROOT}/${name}"
  cd "${CODE_ROOT}"
  PYTHONPATH="${CODE_ROOT}/src:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES=0 "${PYTHON}" reproduce_WinCLIP.py \
    --dataset visa \
    --data_path "${DATA_ROOT}" \
    --save_path "${RESULT_ROOT}/${name}" \
    --model ViT-B-16-plus-240 \
    --pretrained openai \
    --k_shot "${shot}" \
    --image_size 240 \
    --seed 10
}

case "${MODE}" in
  zero) run_shot 0 zero_shot ;;
  one)  run_shot 1 one_shot ;;
  two)  run_shot 2 two_shot ;;
  four) run_shot 4 four_shot ;;
  all)
    run_shot 0 zero_shot
    run_shot 1 one_shot
    run_shot 2 two_shot
    run_shot 4 four_shot
    ;;
  *) echo "Usage: $0 [zero|one|two|four|all]" >&2; exit 2 ;;
esac
