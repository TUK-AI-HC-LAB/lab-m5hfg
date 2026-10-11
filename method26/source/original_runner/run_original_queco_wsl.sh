#!/usr/bin/env bash
# 원본 run.py의 기본 설정으로 MVTec 전체 15개 카테고리를 평가한다.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export QUECO_ORIGINAL_SOURCE="${QUECO_ORIGINAL_SOURCE:-/mnt/c/Users/test/Downloads/QueCo/QueCo}"
export QUECO_SUPPORT_SOURCE="${QUECO_SUPPORT_SOURCE:-/mnt/c/Users/test/Downloads/dinomaly_share_codebase/dinomaly_share_codebase}"
PYTHON="${QUECO_PYTHON:-/home/test/miniforge3/envs/patchcore-gpu/bin/python}"
DATA="${MVTEC_DATA_PATH:-/home/test/data/mvtec}"
mkdir -p "$ROOT/results"
OUT="$(mktemp -d "$ROOT/results/queco_original_20261010_XXXXXX")"
echo "RESULT_ROOT=$OUT"
trap 'code=$?; printf "%s\n" "$code" > "$OUT/exit_code.txt"' EXIT
cd "$OUT"
date -Is > started_at.txt
"$PYTHON" -u "$ROOT/launch_original_queco.py" \
  --train_data_path "$DATA" --test_data_path "$DATA" \
  --num_workers 1 --wan "$@" 2>&1 | tee run.log
date -Is > finished_at.txt
