# Capsule paper-CUDA device measurement

- Paper/code: https://github.com/hustCYQ/MiniShift-Simple3D
- Measured upstream: `5951e097ed5f6c52c1e058490f8c64f9b6323501`
- Main result: `results/capsule_devices_20261101_paper_cuda_float32/summary.json`
- Reports: `../markdown/Simple3D_capsule_measurement_results_20261101.md` and
  `../../related_work/markdown/W45_Simple3D_paper_vs_execution_devices.md`.

## Execution on the recorded WSL machine

```bash
bash run_capsule_devices.sh --paper-cuda-fpfh --smoke
bash run_capsule_devices.sh --paper-cuda-fpfh
```

This runner refuses to overwrite an existing completed/timed run. Preserve the
published output directory and select a new output directory in the runner for
a fresh measurement. It never resumes from old banks or scores.

The prerequisites are the existing `/home/test/simple3d-env` environment and
`/home/test/simple3d-native-env`, CUDA toolkit and native extensions prepared by
`setup_simple3d.sh` and `setup_native_ops.sh`. Full capsule data must be at
`/home/test/data/MiniShift/capsule`. Large dependencies, data and weights are
not included in Git. The official checkout is read from
`/mnt/c/Users/test/Desktop/Codex/Project/MiniShift-Simple3D`.

`benchmark_simple3d.py` and `simple3d_compat.py` supply the shared loader/path
fixes. `SIMPLE3D_NATIVE_OPS=1` retains compiled native CUDA KNN and PointNet2.
`simple3d_paper_cuda_fpfh.py` replaces only the FPFH frontend with the Open3D
tensor CUDA path. The paper states CUDA FPFH, whereas the published legacy API
uses CPU. This is a disclosed backend adaptation, not an assertion of bitwise
or complete paper-protocol equality. The reports list remaining differences.

## Evidence retained

The result folder contains settings, full sample manifest, per-sample and
per-stage timings, device audit, fit state, storage inventory and final metrics.
Point score NPZs and the bank remain at the `raw_assets` path in `summary.json`.
They are not uploaded along with datasets, environments, caches or run logs.

Metadata erratum: the immutable original `summary.json` timing description
still says `CPU normals/FPFH`. Actual CUDA execution is independently recorded
in `environment.json`, `device_audit.json` and `stage_timings.csv`; numerical
results have not been changed.
