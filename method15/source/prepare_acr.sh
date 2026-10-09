#!/usr/bin/env bash
set -euo pipefail
if [ ! -d /home/test/ACR/.git ]; then
  git clone https://github.com/aodongli/zero-shot-ad-via-batch-norm.git /home/test/ACR
fi
git -C /home/test/ACR checkout 54f1a6026cda4a501d870e49b7d49004b38a16fe
if [ ! -x /home/test/acr-env/bin/python ]; then
  /home/test/miniforge3/envs/patchcore-gpu/bin/python -m venv --system-site-packages /home/test/acr-env
fi
# The existing GPU runtime already provides torch/torchvision/yaml/scipy/sklearn/tensorboard.
/home/test/acr-env/bin/python -c 'import torch,torchvision,yaml,scipy,sklearn; import torch.utils.tensorboard'
if [ ! -d /home/test/VAND-APRIL-GAN/.git ]; then
  git clone https://github.com/ByChelsea/VAND-APRIL-GAN.git /home/test/VAND-APRIL-GAN
fi
git -C /home/test/VAND-APRIL-GAN checkout f13b8a634e04f9fde8fa03db125b25af5695d8e1
