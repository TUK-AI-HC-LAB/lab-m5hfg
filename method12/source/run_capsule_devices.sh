#!/usr/bin/env bash
set -euo pipefail
TOOLKIT=/home/test/simple3d-native-toolkit
export CUDA_HOME="$TOOLKIT" PATH="$TOOLKIT/bin:$PATH"
export CC="$TOOLKIT/bin/x86_64-conda-linux-gnu-cc" CXX="$TOOLKIT/bin/x86_64-conda-linux-gnu-c++"
export CUDAHOSTCXX="$CXX" TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=2
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 SIMPLE3D_NATIVE_OPS=1
export LD_LIBRARY_PATH="$TOOLKIT/lib:/home/test/simple3d-env/system-libs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export CPLUS_INCLUDE_PATH="$(find /home/test/simple3d-env/lib/python3.11/site-packages/nvidia -type d -name include | paste -sd:):$TOOLKIT/targets/x86_64-linux/include"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec /home/test/simple3d-native-env/bin/python -u "$SCRIPT_DIR/benchmark_capsule_devices.py" "$@"
