#!/usr/bin/env bash
# MVTec AD 15개 category를 공통 framework의 QueCo로 평가합니다.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$ROOT/results"
OUT="$(mktemp -d "$ROOT/results/queco_all_categories_20261010_XXXXXX")"
echo "RESULT_ROOT=$OUT"
trap 'code=$?; printf "%s\n" "$code" > "$OUT/exit_code.txt"' EXIT
categories=(bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper)
DATA="${MVTEC_DATA_PATH:-/home/test/data/mvtec}"
for category in "${categories[@]}"; do
  test -d "$DATA/$category/test" || { echo "Missing category: $category"; exit 1; }
done
bash "$ROOT/run_queco_wsl.sh" \
  --category "${categories[@]}" \
  --data-path "$DATA" --results-path "$OUT" --seed 0 \
  2>&1 | tee "$OUT/run.log"
