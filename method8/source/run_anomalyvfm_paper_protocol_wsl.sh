#!/usr/bin/env bash
# Run AnomalyVFM with the numerical settings stated in the CVPR 2026 paper.
# The upstream clone, synthetic training set, models, checkpoints, and raw log stay outside this repository.
set -euo pipefail

UPSTREAM=/home/test/AnomalyVFM
ENV_NAME=anomalyvfm
WINDOWS_REPO=/mnt/c/Users/test/Desktop/Codex/lab-m5hfg
RESULT_DIR="$WINDOWS_REPO/method8/source/results/anomalyvfm_paper_protocol"
RELEASE_DATA=/home/test/data/anomalyvfm_release/extracted/synthetic_dataset_flux_filter_dinov3
SYNTHETIC_DATA=/home/test/data/anomalyvfm_paper_10000_seed12
RUN_DIR=/home/test/anomalyvfm_paper_protocol

source /home/test/miniforge3/etc/profile.d/conda.sh
mkdir -p "$RESULT_DIR" "$RUN_DIR" "$UPSTREAM/data"

# These links use the existing target data only for evaluation. Neither dataset is used in training.
ln -sfn /home/test/data/mvtec "$UPSTREAM/data/mvtec"
ln -sfn /home/test/data/VisA_20220922 "$UPSTREAM/data/visa"

# The released post-paper archive is larger than the paper's 10,000-image set.
# Build the paper-sized 5,000 normal + 5,000 anomaly subset as symlinks, not copies.
if [ ! -f "$SYNTHETIC_DATA/manifest.txt" ]; then
  conda run -n "$ENV_NAME" python "$WINDOWS_REPO/method8/source/prepare_anomalyvfm_paper_subset.py" \
    --source "$RELEASE_DATA" --output "$SYNTHETIC_DATA" --per-class 5000 --seed 12
fi

# Paper configuration: RADIOv2.5 ViT-L/16; 768px; synthetic 10,000 images;
# LoRA rank 64 in Q, V and projection; AdamW lr=1e-4; effective batch=32;
# 500 iterations; seed=12. The paper uses alpha=0.1 and beta=5 internally.
# Current official code documents later stabilization changes; this command pins all paper-disclosed flags.
conda run -n "$ENV_NAME" python "$UPSTREAM/train.py" \
  --model radio \
  --peft-type lora \
  --peft-rank 64 \
  --image-size 768 \
  --batch-size 32 \
  --accumulation-steps 4 \
  --optimizer adamw \
  --learning-rate 1e-4 \
  --train-steps 500 \
  --seed 12 \
  --data-path "$SYNTHETIC_DATA" \
  --no-evaluate \
  --out-path "$RUN_DIR" \
  > "$RESULT_DIR/log.txt" 2>&1

cp "$RUN_DIR/model.pkl" "$RESULT_DIR/model.pkl"

# Only MVTec AD and VisA are installed on this PC. The paper's full industrial average needs seven more datasets.
conda run -n "$ENV_NAME" python "$UPSTREAM/test.py" \
  --model radio \
  --peft-type lora \
  --peft-rank 64 \
  --image-size 768 \
  --model-path "$RESULT_DIR/model.pkl" \
  --datasets mvtec_ad visa \
  --out-path "$RESULT_DIR/evaluation" \
  --mean-kernel-size 5 \
  >> "$RESULT_DIR/log.txt" 2>&1
