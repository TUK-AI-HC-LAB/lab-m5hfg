#!/usr/bin/env bash
set -euo pipefail

# AA-CLIP VisA(full-shot) -> MVTec reproduction from adapter initialization.
# Official revision: 53db195f230442aa118c246876c94ba1c76139cc.
# Local, documented runtime fix in train.py: freeze the two CLIP backbones;
# only AA-CLIP text/image adapters remain trainable. TF32 is on; AMP is off.
AA_CLIP_ROOT="/home/test/Project/AA-CLIP"
VISA_ROOT="/home/test/data/VisA_20220922"
CHECKPOINT_ROOT="/home/test/aaclip_checkpoints/visa_fullshot_seed111_tf32_fp32_frozen"

source /home/test/miniforge3/bin/activate patchcore-gpu
test -d "$VISA_ROOT"
test -f "$AA_CLIP_ROOT/model/ViT-L-14-336px.pt"
test ! -e "$CHECKPOINT_ROOT"

cd "$AA_CLIP_ROOT"
mkdir -p "$CHECKPOINT_ROOT"

python -c '
import runpy
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
runpy.run_path("train.py", run_name="__main__")
' \
  --dataset VisA --training_mode full_shot --shot -1 \
  --img_size 518 --text_batch_size 16 --image_batch_size 2 \
  --text_epoch 5 --image_epoch 20 --text_lr 0.00001 --image_lr 0.0005 \
  --text_norm_weight 0.1 --text_adapt_weight 0.1 --image_adapt_weight 0.1 \
  --text_adapt_until 3 --image_adapt_until 6 --seed 111 \
  --save_path "$CHECKPOINT_ROOT"

python -c '
import runpy
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
runpy.run_path("test.py", run_name="__main__")
' \
  --dataset MVTec --img_size 518 --batch_size 16 \
  --text_norm_weight 0.1 --text_adapt_weight 0.1 --image_adapt_weight 0.1 \
  --text_adapt_until 3 --image_adapt_until 6 --seed 111 \
  --save_path "$CHECKPOINT_ROOT"

echo "AA_CLIP_REPRODUCTION_FINISHED: $CHECKPOINT_ROOT/test.log"
