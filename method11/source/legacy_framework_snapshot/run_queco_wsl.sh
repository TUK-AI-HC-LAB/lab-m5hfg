#!/usr/bin/env bash
# snapshot의 기존 main.py에서 QueCo를 선택하고 평가합니다.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${QUECO_PYTHON:-/home/test/miniforge3/envs/patchcore-gpu/bin/python}"
# snapshot에 없는 공통 utils 등의 모듈은 기존 framework에서 참조합니다.
BASE="${FRAMEWORK_BASE:-/mnt/c/Users/test/Downloads/dinomaly_share_codebase/dinomaly_share_codebase}"
export PYTHONPATH="$ROOT:$BASE${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
if [[ "${1:-}" == "--smoke" ]]; then
  "$PYTHON" "$ROOT/smoke_queco.py"
  exit 0
fi
"$PYTHON" "$ROOT/main.py" --config "$ROOT/experiment.yaml" --method queco "$@"
