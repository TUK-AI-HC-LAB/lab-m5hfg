#!/usr/bin/env bash
set -euo pipefail
BASE=/home/test/miniforge3/envs/patchcore-gpu/bin/python
if [ ! -d /home/test/open-iad/.git ]; then
  git clone https://github.com/M-3LAB/open-iad.git /home/test/open-iad
fi
git -C /home/test/open-iad checkout 05044fedab142ce4bfd71bb618b3200c0e43f198
if [ ! -x /home/test/graphcore-env/bin/python ]; then
  "$BASE" -m venv --system-site-packages /home/test/graphcore-env
fi
/home/test/graphcore-env/bin/python -m pip install timm==1.0.28 faiss-cpu==1.14.3
mkdir -p /home/test/graphcore_assets
if [ ! -s /home/test/graphcore_assets/pvig_ti_78.5.pth.tar ]; then
  curl -L --fail https://github.com/huawei-noah/Efficient-AI-Backbones/releases/download/pyramid-vig/pvig_ti_78.5.pth.tar -o /home/test/graphcore_assets/pvig_ti_78.5.pth.tar
fi
