#!/usr/bin/env bash
# WSL2 setup used for this run. Large assets stay outside the student repository.
set -euo pipefail
ENV=/home/test/simple3d-env
DATA=/home/test/data/MiniShift
REPO=/mnt/c/Users/test/Desktop/Codex/Project/MiniShift-Simple3D
if [[ ! -d "$REPO/.git" ]]; then
  git clone https://github.com/hustCYQ/MiniShift-Simple3D.git "$REPO"
fi
git -C "$REPO" checkout 5951e097ed5f6c52c1e058490f8c64f9b6323501
if [[ ! -x "$ENV/bin/python" ]]; then
  /home/test/miniforge3/bin/conda create -p "$ENV" python=3.11 pip -y
fi
"$ENV/bin/pip" install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
"$ENV/bin/pip" install open3d timm tifffile scikit-learn scipy pandas tabulate tqdm opencv-python-headless psutil cupy-cuda12x
mkdir -p "$ENV/system-libs"
(cd "$ENV/system-libs"; apt-get download libusb-1.0-0; dpkg-deb -x libusb*.deb .)
mkdir -p "$DATA"
if [[ ! -f "$DATA/capsule.zip" ]]; then
  curl -L --fail --retry 3 -o "$DATA/capsule.zip" https://huggingface.co/datasets/ChengYuQi99/MiniShift/resolve/main/capsule.zip
fi
if [[ ! -d "$DATA/capsule/train" ]]; then
  mkdir -p "$DATA/capsule"
  "$ENV/bin/python" -c 'import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' "$DATA/capsule.zip" "$DATA/capsule"
fi
