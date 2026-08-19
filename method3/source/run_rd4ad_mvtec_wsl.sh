#!/usr/bin/env bash
set -euo pipefail

# Official RD4AD code is kept locally and ignored by Git.
RD4AD_ROOT="/mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method3/source/rd4ad"
# MVTec AD is stored in the Linux filesystem so image loading does not use /mnt/c.
MVTEC_ROOT="/home/test/data/mvtec"
RESULT_ROOT="/mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method3/source/results/rd4ad_runs"
CHECKPOINT_ROOT="/home/test/rd4ad_checkpoints"
CATEGORIES=(carpet bottle hazelnut leather cable capsule grid pill transistor metal_nut screw toothbrush zipper tile wood)

source /home/test/miniforge3/bin/activate patchcore-gpu
cd "$RD4AD_ROOT"
mkdir -p "$RESULT_ROOT" "$CHECKPOINT_ROOT"

for category in "${CATEGORIES[@]}"; do
  result_csv="$RESULT_ROOT/${category}.csv"
  # A CSV header is written at start, so require the category result row before skipping.
  if [[ -f "$result_csv" ]] && grep -q "^${category}," "$result_csv"; then
    echo "SKIP_COMPLETED: $category"
    continue
  fi

  python main.py \
    --data_root "$MVTEC_ROOT" \
    --checkpoint_root "$CHECKPOINT_ROOT" \
    --results_csv "$result_csv" \
    --epochs 200 \
    --test_every 10 \
    --seed 111 \
    --categories "$category"
done

echo "RD4AD_MVTEC_FINISHED"
