#!/usr/bin/env bash
# W42 실행 후 정리한 재실행 스크립트입니다. 스크립트 자체는 재실행하지 않았습니다.
# 사용법: bash run_queco_screw_w42.sh <프레임워크 경로> <MVTec 경로> <Python 실행파일>
# 이번에 검증한 수정본과 같은 코드·configs/queco.yaml을 지정해야 합니다.
set -euo pipefail

framework_path="${1:?프레임워크 경로를 지정하세요}"
dataset_path="${2:?MVTec 경로를 지정하세요}"
python_path="${3:?필요한 패키지가 설치된 Python 실행파일을 지정하세요}"

# 작업 디렉터리를 바꾸기 전에 경로를 절대경로로 확정합니다.
framework_path="$(realpath "$framework_path")"
dataset_path="$(realpath "$dataset_path")"
python_path="$(realpath "$python_path")"
test -f "$framework_path/main.py"
test -d "$dataset_path/screw/test"
test -x "$python_path"

cd "$framework_path"
mkdir -p results
# 기존 실행 결과와 겹치지 않는 새 폴더를 만듭니다.
output_path="$(mktemp -d "$framework_path/results/queco_screw_w42_XXXXXX")"
echo "결과 저장 위치: $output_path"

PYTHONDONTWRITEBYTECODE=1 "$python_path" -u main.py \
  --method queco --category screw --data-path "$dataset_path" \
  --results-path "$output_path" 2>&1 | tee "$output_path/run.log"
