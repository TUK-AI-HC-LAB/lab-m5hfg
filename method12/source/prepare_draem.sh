#!/usr/bin/env bash
set -euo pipefail
# Download script is separate from evaluation: existing weights are reused.
if [ ! -d /home/test/DRAEM/.git ]; then
  git clone https://github.com/VitjanZ/DRAEM.git /home/test/DRAEM
  git -C /home/test/DRAEM checkout 2dbf67397ab5c10a1494e5ae70ab59a25d7c35ef
fi
test "$(git -C /home/test/DRAEM rev-parse HEAD)" = 2dbf67397ab5c10a1494e5ae70ab59a25d7c35ef
if [ ! -x /home/test/draem-tools/bin/gdown ]; then
  /home/test/miniforge3/envs/patchcore-gpu/bin/python -m venv --system-site-packages /home/test/draem-tools
  /home/test/draem-tools/bin/python -m pip install gdown==6.4.1
fi
mkdir -p /home/test/draem_weights
if [ ! -f /home/test/draem_weights/DRAEM_checkpoints.zip ]; then
  /home/test/draem-tools/bin/gdown 'https://drive.google.com/uc?id=1eOE8wXNihjsiDvDANHFbg_mQkLesDrs1' -O /home/test/draem_weights/DRAEM_checkpoints.zip
fi
if [ ! -d /home/test/draem_weights/DRAEM_checkpoints ]; then
  /home/test/draem-tools/bin/python -m zipfile -e /home/test/draem_weights/DRAEM_checkpoints.zip /home/test/draem_weights
fi
