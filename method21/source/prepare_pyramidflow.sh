#!/usr/bin/env bash
set -euo pipefail
export GIT_TERMINAL_PROMPT=0
if [ ! -d /home/test/PyramidFlow/.git ]; then
  git clone https://github.com/FourthM/PyramidFlow.git /home/test/PyramidFlow
fi
git -C /home/test/PyramidFlow checkout --detach c463b1cb0c2b084cdbae290234e3f8657298e981
# Original README points to the now unavailable gasharper/autoFlow. Use the
# retained autoFlow.py from this pinned public fork, without editing checkout.
if [ ! -d /home/test/PyramidFlow-dependency/.git ]; then
  git clone https://github.com/twimclee/pyramidflow-moai.git /home/test/PyramidFlow-dependency
fi
git -C /home/test/PyramidFlow-dependency checkout --detach fa322f9175966ebed2759b0422460b401c318b05
/home/test/miniforge3/envs/patchcore-gpu/bin/python -c 'import torch,torchvision,albumentations,numpy,scipy,sklearn; assert torch.cuda.is_available(); print(torch.__version__,torch.cuda.get_device_name())'
/home/test/miniforge3/envs/patchcore-gpu/bin/python -m pip freeze
