#!/usr/bin/env bash
set -euo pipefail

# INP-Former CVPR 2025 zero-shot protocol:
# train the multi-class model on VisA 1cls, then evaluate it on MVTec AD.
# Official implementation revision: 17d265381d9b323a2ef6e05aab0665a85edebe84.
# Paper defaults: DINOv2-reg ViT-B/14, resize 448, crop 392, INP_num 6,
# 200 epochs, and batch size 16.
INP_ROOT="/home/test/Project/INP-Former"
VISA_ROOT="/home/test/data/VisA_pytorch/1cls"
MVTEC_ROOT="/home/test/data/mvtec"
SAVE_DIR="/home/test/inpformer_checkpoints"

source /home/test/miniforge3/bin/activate patchcore-gpu
test -d "$VISA_ROOT"
test -d "$MVTEC_ROOT"
test -d "$INP_ROOT/.git"

cd "$INP_ROOT"

python INP_Former_Multi_Class.py \
  --dataset VisA --data_path "$VISA_ROOT" --phase train \
  --save_dir "$SAVE_DIR" --encoder dinov2reg_vit_base_14 \
  --input_size 448 --crop_size 392 --INP_num 6 \
  --total_epochs 200 --batch_size 16

python INP_Former_Zero_Shot.py \
  --source_dataset VisA --dataset MVTec-AD --data_path "$MVTEC_ROOT" \
  --save_dir "$SAVE_DIR" --encoder dinov2reg_vit_base_14 \
  --input_size 448 --crop_size 392 --INP_num 6 --batch_size 16
