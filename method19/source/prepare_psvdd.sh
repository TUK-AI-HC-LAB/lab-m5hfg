#!/usr/bin/env bash
set -euo pipefail
if [ ! -d /home/test/PatchSVDD/.git ]; then git clone https://github.com/nuclearboy95/Anomaly-Detection-PatchSVDD-PyTorch.git /home/test/PatchSVDD; fi
git -C /home/test/PatchSVDD checkout 934d6238e5e0ad511e2a0e7fc4f4899010e7d892
if [ ! -x /home/test/psvdd-env/bin/python ]; then
 /home/test/miniforge3/envs/patchcore-gpu/bin/python -m venv --system-site-packages /home/test/psvdd-env
fi
/home/test/psvdd-env/bin/python -m pip install ngt==2.8.0.post1
# Use existing original BTAD dataset and method11 meta.json provenance.
