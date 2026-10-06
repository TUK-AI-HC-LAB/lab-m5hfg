#!/usr/bin/env bash
# Reproduce the W41 common-framework condition into a fresh result directory.
set -euo pipefail
framework="${FRAMEWORK_ROOT:?Set FRAMEWORK_ROOT to the common-framework checkout}"
python="${PYTHON_EXECUTABLE:-python}"
data="${MVTEC_ROOT:?Set MVTEC_ROOT to the MVTec AD dataset root}"
output="${1:?Provide a new output directory; preserve the original W41 result}"
if [[ -e "$output" || -e "${output}.log" ]]; then
    echo 'Output already exists; choose a fresh directory.' >&2
    exit 1
fi
mkdir -p "$(dirname "$output")"
cd "$framework"
"$python" main.py --method musc --dataset mvtec --category bottle \
    --data-path "$data" --results-path "$output" \
    --seed 42 --num-workers 1 2>&1 | tee "${output}.log"
