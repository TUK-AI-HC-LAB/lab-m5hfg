"""Detach the long paper-CUDA capsule measurement and retain its log/PID."""
import json
import os
import subprocess
from pathlib import Path

here = Path(__file__).resolve().parent
out = here / 'results/capsule_devices_20261101_paper_cuda_float32'
out.mkdir(parents=True, exist_ok=True)
if (out / 'launcher.json').exists():
    previous = json.loads((out / 'launcher.json').read_text())
    try:
        os.kill(previous['pid'], 0)
    except ProcessLookupError:
        pass
    else:
        raise RuntimeError('An existing benchmark launcher PID is still alive')
with (out / 'run.log').open('ab', buffering=0) as log:
    process = subprocess.Popen(
        ['bash', str(here / 'run_capsule_devices.sh'), '--paper-cuda-fpfh'],
        cwd=here, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True, close_fds=True)
record = dict(pid=process.pid, log=str(out / 'run.log'), command='bash run_capsule_devices.sh --paper-cuda-fpfh')
(out / 'launcher.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record))
