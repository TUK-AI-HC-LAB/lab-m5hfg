#!/usr/bin/env bash
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
LOG_ROOT="${LOG_ROOT:?Set LOG_ROOT to an external directory for logs and outputs}"
mkdir -p "$LOG_ROOT"
"${PYTHON_EXECUTABLE:-python}" -u "$ROOT/run_mvtec_all.py" 2>&1 | tee -a "$LOG_ROOT/batch.log"
