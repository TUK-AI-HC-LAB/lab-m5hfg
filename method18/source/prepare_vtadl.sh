#!/usr/bin/env bash
set -euo pipefail
if [ ! -d /home/test/VT-ADL/.git ]; then git clone https://github.com/pankajmishra000/VT-ADL.git /home/test/VT-ADL; fi
git -C /home/test/VT-ADL checkout 20b58e2dd810e1d747b9a4132899cc474691377e
if [ ! -x /home/test/vtadl-env/bin/python ]; then
  /home/test/miniforge3/envs/patchcore-gpu/bin/python -m venv --system-site-packages /home/test/vtadl-env
fi
/home/test/vtadl-env/bin/python -m pip install einops==0.7.0
# Dataset download/provenance already retained in method11/source/results.
# Original root /home/test/data/btad_original/BTech_Dataset_transformed
