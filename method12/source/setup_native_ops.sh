#!/usr/bin/env bash
set -euo pipefail
TASK_ENV=/home/test/simple3d-native-env
TOOLKIT=/home/test/simple3d-native-toolkit
export CUDA_HOME="$TOOLKIT"
export PATH="$TOOLKIT/bin:$PATH"
export CC="$TOOLKIT/bin/x86_64-conda-linux-gnu-cc"
export CXX="$TOOLKIT/bin/x86_64-conda-linux-gnu-c++"
export CUDAHOSTCXX="$CXX"
export TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=2
export LD_LIBRARY_PATH="$TOOLKIT/lib:/home/test/simple3d-env/system-libs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export CPLUS_INCLUDE_PATH="$(find /home/test/simple3d-env/lib/python3.11/site-packages/nvidia -type d -name include | paste -sd:):$TOOLKIT/targets/x86_64-linux/include"
if [[ ! -x "$TASK_ENV/bin/python" ]]; then
 /home/test/simple3d-env/bin/python -m venv --system-site-packages "$TASK_ENV"
fi
"$TASK_ENV/bin/pip" install ninja setuptools wheel
ROOT=/mnt/c/Users/test/Desktop/Codex/Project
mkdir -p /home/test/simple3d-native-src
if [[ ! -f /home/test/simple3d-native-src/Pointnet2/pointnet2_ops_lib/setup.py ]]; then
 cp -a "$ROOT/Simple3D-native-Pointnet2" /home/test/simple3d-native-src/Pointnet2
fi
"$TASK_ENV/bin/python" - <<'PY'
from pathlib import Path
p=Path('/home/test/simple3d-native-src/Pointnet2/pointnet2_ops_lib/setup.py')
s=p.read_text().replace('"3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"','"12.0"')
p.write_text(s)
PY
"$TASK_ENV/bin/pip" install --no-build-isolation --no-deps /home/test/simple3d-native-src/Pointnet2/pointnet2_ops_lib
"$TASK_ENV/bin/pip" install --no-build-isolation --no-deps "$ROOT/Simple3D-native-KNN_CUDA"
"$TASK_ENV/bin/python" -u - <<'PY'
import torch
from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN
x=torch.randn(1,257,3,device='cuda')
i=pointnet2_utils.furthest_point_sample(x,16)
y=pointnet2_utils.gather_operation(x.transpose(1,2).contiguous(),i).transpose(1,2).contiguous()
d,j=KNN(7,transpose_mode=True)(x,y)
ref=torch.cdist(y,x,compute_mode='donot_use_mm_for_euclid_dist').topk(7,largest=False)
torch.testing.assert_close(d,ref.values,rtol=1e-4,atol=1e-5)
assert torch.equal(j,ref.indices)
print('NATIVE_OPS_SMOKE_PASSED',torch.__version__,torch.cuda.get_device_name(),flush=True)
PY
