#!/usr/bin/env bash
set -euo pipefail

source /home/test/miniforge3/bin/activate patchcore-gpu
mkdir -p results/MVTecAD_Results/simplenet_mvtec/official_default_bottle

exec python main.py \
  --gpu 0 \
  --seed 0 \
  --log_group simplenet_mvtec \
  --log_project MVTecAD_Results \
  --results_path results \
  --run_name official_default_bottle \
  net \
  -b wideresnet50 \
  -le layer2 \
  -le layer3 \
  --pretrain_embed_dimension 1536 \
  --target_embed_dimension 1536 \
  --patchsize 3 \
  --meta_epochs 40 \
  --embedding_size 256 \
  --gan_epochs 4 \
  --noise_std 0.015 \
  --dsc_hidden 1024 \
  --dsc_layers 2 \
  --dsc_margin .5 \
  --pre_proj 1 \
  dataset \
  --batch_size 8 \
  --resize 329 \
  --imagesize 288 \
  -d bottle \
  mvtec \
  /mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method1/source/patchcore-inspection/data/mvtec \
  > results/MVTecAD_Results/simplenet_mvtec/official_default_bottle/console.log 2>&1
