#!/usr/bin/env bash
# 의존성 설치 후 남은 RD 계열과 PromptAD의 기본 설정 실행을 기록합니다.

set -uo pipefail

ROOT="/mnt/c/Users/test/Downloads/dinomaly_share_codebase/dinomaly_share_codebase"
PYTHON="/home/test/miniforge3/envs/patchcore-gpu/bin/python"
DATA_ROOT="/home/test/data/mvtec"
RESULT_ROOT="/home/test/shared_framework_all_methods_w40_20260926/retry_after_dependencies_round2"
METHODS=(rd rd_orig promptad)

mkdir -p "$RESULT_ROOT"
STATUS_FILE="$RESULT_ROOT/status.tsv"
printf 'method\tstatus\tstarted_at\tfinished_at\texit_code\tresult_root\tlog\n' > "$STATUS_FILE"

for method in "${METHODS[@]}"; do
  method_root="$RESULT_ROOT/$method"
  log_file="$RESULT_ROOT/${method}.log"
  started_at="$(date -Is)"
  mkdir -p "$method_root"
  "$PYTHON" "$ROOT/main.py" \
    --config "$ROOT/experiment.yaml" \
    --method "$method" --dataset mvtec --category bottle \
    --data-path "$DATA_ROOT" --results-path "$method_root" \
    --seed 0 --num-workers 1 2>&1 | tee "$log_file"
  exit_code=${PIPESTATUS[0]}
  finished_at="$(date -Is)"
  if [[ "$exit_code" -eq 0 ]]; then status="success"; else status="failed"; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$method" "$status" "$started_at" "$finished_at" "$exit_code" "$method_root" "$log_file" >> "$STATUS_FILE"
done
