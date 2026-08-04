#!/usr/bin/env bash
set -euo pipefail

SIMPLE_NET_ROOT="/mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method2/source/simplenet"
MVTEC_ROOT="/mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method1/source/patchcore-inspection/data/mvtec"
CARPET_RESULT="$SIMPLE_NET_ROOT/results/MVTecAD_Results/simplenet_mvtec/carpet_official_default_workers2/results.csv"
REMAINING_CATEGORIES=(grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper)

cd "$SIMPLE_NET_ROOT"
source /home/test/miniforge3/bin/activate patchcore-gpu

while [[ ! -f "$CARPET_RESULT" ]]; do
  echo "Waiting for carpet to finish..."
  sleep 60
done

for category in "${REMAINING_CATEGORIES[@]}"; do
  python main.py \
    --gpu 0 \
    --seed 0 \
    --log_group simplenet_mvtec \
    --log_project MVTecAD_Results \
    --results_path results \
    --run_name "${category}_official_default_workers2" \
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
    --num_workers 2 \
    --resize 329 \
    --imagesize 288 \
    -d "$category" \
    mvtec "$MVTEC_ROOT"
done

echo "ALL_SIMPLENET_CATEGORIES_FINISHED"
