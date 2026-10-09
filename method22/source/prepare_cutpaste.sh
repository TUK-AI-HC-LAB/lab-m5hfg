#!/usr/bin/env bash
set -euo pipefail
export GIT_TERMINAL_PROMPT=0
if [ ! -d /home/test/CutPaste/.git ]; then
  git clone https://github.com/Runinho/pytorch-cutpaste.git /home/test/CutPaste
fi
git -C /home/test/CutPaste checkout --detach 10d8bf71df76d3a97f0106efee1d76f81d983149
/home/test/miniforge3/envs/patchcore-gpu/bin/python -c 'import torch,torchvision,numpy,scipy,sklearn; assert torch.cuda.is_available(); print(torch.__version__,torch.cuda.get_device_name())'
/home/test/miniforge3/envs/patchcore-gpu/bin/python -m pip freeze
