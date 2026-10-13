#!/usr/bin/env bash
# MVTec AD bottle에서 정상 train 90%/validation 10% 분할을 적용해
# 등록된 모든 방법을 재실행하는 재현 스크립트입니다.
# 각 방법은 실패해도 다음 방법을 계속 실행하며, 상태와 표준 출력은 결과 폴더에 남깁니다.

set -uo pipefail

ROOT="/mnt/c/Users/test/Downloads/dinomaly_share_codebase/dinomaly_share_codebase"
PYTHON="/home/test/miniforge3/envs/patchcore-gpu/bin/python"
DATA_ROOT="/home/test/data/mvtec"
RESULT_ROOT="/home/test/shared_framework_all_methods_w40_validation_20260928"
METHODS=(patchcore padim winclip coad simple rd rd_orig promptad glass dinomaly uniad anomalyclip)

mkdir -p "$RESULT_ROOT/logs"
STATUS_FILE="$RESULT_ROOT/status.tsv"
printf 'method\tstatus\tstarted_at\tfinished_at\texit_code\tresult_root\tlog\n' > "$STATUS_FILE"

run_method() {
  local method="$1"
  local method_root="$RESULT_ROOT/$method"
  local log_file="$RESULT_ROOT/logs/${method}.log"
  local started_at
  local finished_at
  local exit_code
  started_at="$(date -Is)"
  mkdir -p "$method_root"

  if [[ "$method" == "anomalyclip" ]]; then
    "$PYTHON" "$ROOT/main.py" \
      --config "$ROOT/experiment_anomalyclip.yaml" \
      --results-path "$method_root" \
      --seed 0 --num-workers 1 2>&1 | tee "$log_file"
    exit_code=${PIPESTATUS[0]}
  else
    "$PYTHON" "$ROOT/main.py" \
      --config "$ROOT/experiment.yaml" \
      --method "$method" --dataset mvtec --category bottle \
      --data-path "$DATA_ROOT" --results-path "$method_root" \
      --seed 0 --num-workers 1 2>&1 | tee "$log_file"
    exit_code=${PIPESTATUS[0]}
  fi

  finished_at="$(date -Is)"
  if [[ "$exit_code" -eq 0 ]]; then
    printf '%s\tsuccess\t%s\t%s\t%s\t%s\t%s\n' \
      "$method" "$started_at" "$finished_at" "$exit_code" "$method_root" "$log_file" >> "$STATUS_FILE"
  else
    printf '%s\tfailed\t%s\t%s\t%s\t%s\t%s\n' \
      "$method" "$started_at" "$finished_at" "$exit_code" "$method_root" "$log_file" >> "$STATUS_FILE"
  fi
}

for method in "${METHODS[@]}"; do
  run_method "$method"
done
