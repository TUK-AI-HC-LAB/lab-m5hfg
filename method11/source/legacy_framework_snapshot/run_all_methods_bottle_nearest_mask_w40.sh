#!/usr/bin/env bash
# bilinear -> nearest GT mask 변경 후 GLASS를 제외한 모든 registry method를
# MVTec AD bottle에서 순차 실행한다. 각 method가 실패해도 다음 method를 계속한다.

set -uo pipefail

ROOT="/mnt/c/Users/test/Downloads/dinomaly_share_codebase/dinomaly_share_codebase"
PYTHON="/home/test/miniforge3/envs/patchcore-gpu/bin/python"
DATA_ROOT="/home/test/data/mvtec"
RESULT_ROOT="/home/test/shared_framework_nearest_mask_w40_20260928"
METHODS=(patchcore padim winclip coad simple rd rd_orig promptad dinomaly uniad anomalyclip)

mkdir -p "$RESULT_ROOT/logs"
STATUS_FILE="$RESULT_ROOT/status.tsv"
printf 'method\tstatus\tstarted_at\tfinished_at\texit_code\tresult_root\tlog\n' > "$STATUS_FILE"

run_method() {
  local method="$1"
  local method_root="$RESULT_ROOT/$method"
  local log_file="$RESULT_ROOT/logs/${method}.log"
  local started_at finished_at exit_code status
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
  if [[ "$exit_code" -eq 0 ]]; then status="success"; else status="failed"; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$method" "$status" "$started_at" "$finished_at" "$exit_code" "$method_root" "$log_file" >> "$STATUS_FILE"
}

for method in "${METHODS[@]}"; do
  run_method "$method"
done
