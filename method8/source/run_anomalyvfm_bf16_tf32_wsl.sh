#!/usr/bin/env bash
# Optimized AnomalyVFM run: paper numerical settings plus BF16 autocast and TF32.
# This is intentionally separate from the paper-condition script because the precision path changes.
set -euo pipefail

UPSTREAM=/home/test/AnomalyVFM
ENV_NAME=anomalyvfm
WINDOWS_REPO=/mnt/c/Users/test/Desktop/Codex/lab-m5hfg
RESULT_DIR="$WINDOWS_REPO/method8/source/results/anomalyvfm_bf16_tf32"
SYNTHETIC_DATA=/home/test/data/anomalyvfm_paper_10000_seed12
RUN_DIR=/home/test/anomalyvfm_bf16_tf32
PATCH_FILE="$WINDOWS_REPO/method8/source/anomalyvfm_bf16_tf32.patch"

source /home/test/miniforge3/etc/profile.d/conda.sh
mkdir -p "$RESULT_DIR" "$RUN_DIR" "$UPSTREAM/data"
cd "$UPSTREAM"
ln -sfn /home/test/data/mvtec "$UPSTREAM/data/mvtec"
# The official evaluator expects category/test/defect/*.JPG layout.
ln -sfn /home/test/data/VisA_pytorch/1cls "$UPSTREAM/data/visa"

# Apply once; check the already-applied state first because `git apply --check`
# can report success even when a subsequent apply would reject overlapping hunks.
if git -C "$UPSTREAM" apply --reverse --check "$PATCH_FILE"; then
  :
elif git -C "$UPSTREAM" apply --check "$PATCH_FILE"; then
  git -C "$UPSTREAM" apply "$PATCH_FILE"
else
  echo "The local upstream clone does not match the recorded BF16/TF32 patch." >&2
  exit 1
fi

# Same paper-disclosed model, resolution, effective batch, optimizer, LR, and steps.
# --bf16 and --tf32 are this optimized run's only intentional deviations.
TRAIN_ARGS=(
  --model radio --peft-type lora --peft-rank 64 --image-size 768
  --batch-size 32 --accumulation-steps 4 --optimizer adamw
  --learning-rate 1e-4 --train-steps 500 --seed 12
  --data-path "$SYNTHETIC_DATA" --no-evaluate --bf16 --tf32
  --out-path "$RUN_DIR"
)
conda run -n "$ENV_NAME" python "$UPSTREAM/train.py" "${TRAIN_ARGS[@]}" > "$RESULT_DIR/log.txt" 2>&1

cp "$RUN_DIR/model.pkl" "$RESULT_DIR/model.pkl"

TEST_ARGS=(
  --model radio --peft-type lora --peft-rank 64 --image-size 768
  --model-path "$RESULT_DIR/model.pkl" --datasets mvtec_ad visa
  --out-path "$RESULT_DIR/evaluation"
)
conda run -n "$ENV_NAME" python "$UPSTREAM/test.py" "${TEST_ARGS[@]}" >> "$RESULT_DIR/log.txt" 2>&1
