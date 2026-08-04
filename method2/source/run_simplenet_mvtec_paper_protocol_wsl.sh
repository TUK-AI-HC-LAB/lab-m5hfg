#!/usr/bin/env bash
set -euo pipefail

SIMPLE_NET_ROOT="/mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method2/source/simplenet"
# MVTec AD 데이터는 Windows 마운트 경로가 아니라 WSL Linux 파일시스템에 둠.
# DataLoader가 이미지를 읽을 때 /mnt/c 경로보다 빠르고 안정적으로 접근할 수 있음.
MVTEC_ROOT="/home/test/data/mvtec"
CATEGORIES=(bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper)

cd "$SIMPLE_NET_ROOT"
source /home/test/miniforge3/bin/activate patchcore-gpu

for category in "${CATEGORIES[@]}"; do
  # 재부팅 뒤 다시 실행해도 이미 끝난 category는 건너뜀.
  # results.csv가 있어야 완료로 판단하므로, 중간에 멈춘 실행은 다시 수행함.
  result_csv="results/MVTecAD_Results/simplenet_mvtec/${category}_paper_protocol_workers2/results.csv"
  if [[ -f "$result_csv" ]]; then
    echo "SKIP_COMPLETED: $category"
    continue
  fi

  python main.py \
    --gpu 0 \
    --seed 0 \
    --log_group simplenet_mvtec \
    --log_project MVTecAD_Results \
    --results_path results \
    --run_name "${category}_paper_protocol_workers2" \
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
    --dsc_lr 0.0002 \
    --pre_proj 1 \
    dataset \
    --batch_size 4 \
    --num_workers 2 \
    --resize 256 \
    --imagesize 224 \
    -d "$category" \
    mvtec "$MVTEC_ROOT"
done

echo "SIMPLENET_PAPER_PROTOCOL_FINISHED"
