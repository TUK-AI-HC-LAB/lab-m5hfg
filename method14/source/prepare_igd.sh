#!/usr/bin/env bash
set -euo pipefail
if [ ! -d /home/test/IGD/.git ]; then
  git clone https://github.com/tianyu0207/IGD.git /home/test/IGD
  git -C /home/test/IGD checkout 1ce995214ef8adf09f6c5d3dc01ce6042b472a34
fi
test "$(git -C /home/test/IGD rev-parse HEAD)" = 1ce995214ef8adf09f6c5d3dc01ce6042b472a34
/home/test/miniforge3/envs/patchcore-gpu/bin/python -m venv --system-site-packages /home/test/igd-env
/home/test/igd-env/bin/python -m pip install pytorch-msssim==0.2.1 tensorboardX==2.6.5
