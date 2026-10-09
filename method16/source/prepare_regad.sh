#!/usr/bin/env bash
set -euo pipefail
if [ ! -d /home/test/RegAD/.git ]; then
  git clone https://github.com/MediaBrain-SJTU/RegAD.git /home/test/RegAD
fi
git -C /home/test/RegAD checkout 5e2c1f8c18d302b0354471567846fee3ed2ff063
if [ ! -x /home/test/regad-env/bin/python ]; then
  /home/test/miniforge3/envs/patchcore-gpu/bin/python -m venv --system-site-packages /home/test/regad-env
fi
/home/test/regad-env/bin/pip install gdown==6.4.1 kornia==0.6.5
mkdir -p /home/test/regad_assets
if [ ! -s /home/test/regad_assets/save_checkpoints.tar ]; then
  /home/test/regad-env/bin/gdown 1guZBh40btPRmxcnY_lud88V1NoT-eWWX -O /home/test/regad_assets/save_checkpoints.tar
fi
if [ ! -s /home/test/regad_assets/support_set.tar ]; then
  /home/test/regad-env/bin/gdown 1AZcc77cmDfkWA8f8cs-j-CUuFFQ7tPoK -O /home/test/regad_assets/support_set.tar
fi
tar --warning=no-unknown-keyword -xf /home/test/regad_assets/save_checkpoints.tar -C /home/test/regad_assets
tar --warning=no-unknown-keyword -xf /home/test/regad_assets/support_set.tar -C /home/test/regad_assets
/home/test/regad-env/bin/python /mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method16/source/repair_support.py
