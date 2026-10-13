#!/usr/bin/env bash
# 필요한 패키지와 CLIP/DINO 사전학습 가중치를 준비합니다.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${QUECO_PYTHON:-/home/test/miniforge3/envs/patchcore-gpu/bin/python}"
cd "$ROOT"
"$PYTHON" -m pip install -r "$ROOT/requirements_queco.txt"
"$PYTHON" "$ROOT/prepare_queco_weights.py"
