#!/usr/bin/env bash
set -euo pipefail
if [ ! -d /home/test/SPADE-pytorch/.git ]; then
  git clone https://github.com/byungjae89/SPADE-pytorch.git /home/test/SPADE-pytorch
fi
git -C /home/test/SPADE-pytorch checkout --detach 077c67be21d68a38b4442db7311c87e708728286
/home/test/miniforge3/envs/patchcore-gpu/bin/python -c 'import torch,torchvision,cv2,numpy,scipy,sklearn; assert torch.cuda.is_available(); print(torch.__version__,torch.cuda.get_device_name())'
# Runtime is the existing patchcore-gpu environment; preserve its package lock
# with pip freeze, rather than reinstalling the community's 2020 requirements.
/home/test/miniforge3/envs/patchcore-gpu/bin/python -m pip freeze
