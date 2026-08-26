#!/usr/bin/env bash
set -euo pipefail

# AA-CLIP VisA(full-shot) -> MVTec reproduction.
# Both adapter stages are trained from initialization with TF32 + BF16 AMP.
# Evaluation deliberately uses TF32 without AMP because AMP evaluation previously
# produced NaN anomaly scores on this machine.
AA_CLIP_ROOT="/home/test/Project/AA-CLIP"
VISA_ROOT="/home/test/data/VisA_20220922"
CHECKPOINT_ROOT="/home/test/aaclip_checkpoints/visa_fullshot_seed111_tf32_amp_full"
RESULT_CSV="/mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method4/source/results/AA_CLIP_VisA_fullshot_to_MVTec_seed111_tf32_amp_full.csv"

source /home/test/miniforge3/bin/activate patchcore-gpu
test -d "$VISA_ROOT"
test -f "$AA_CLIP_ROOT/model/ViT-L-14-336px.pt"

cd "$AA_CLIP_ROOT"
mkdir -p "$CHECKPOINT_ROOT" "$(dirname "$RESULT_CSV")"

python train.py \
  --dataset VisA --training_mode full_shot --shot -1 \
  --img_size 518 --text_batch_size 16 --image_batch_size 2 \
  --text_epoch 5 --image_epoch 20 --text_lr 0.00001 --image_lr 0.0005 \
  --text_norm_weight 0.1 --text_adapt_weight 0.1 --image_adapt_weight 0.1 \
  --text_adapt_until 3 --image_adapt_until 6 --seed 111 \
  --tf32 --amp --save_path "$CHECKPOINT_ROOT"

python test.py \
  --dataset MVTec --img_size 518 --batch_size 16 \
  --text_norm_weight 0.1 --text_adapt_weight 0.1 --image_adapt_weight 0.1 \
  --text_adapt_until 3 --image_adapt_until 6 --seed 111 \
  --tf32 --save_path "$CHECKPOINT_ROOT" --checkpoint_path "$CHECKPOINT_ROOT/image_adapter.pth" \
  --results_csv "$RESULT_CSV"

echo "AA_CLIP_REPRODUCTION_FINISHED: $RESULT_CSV"
