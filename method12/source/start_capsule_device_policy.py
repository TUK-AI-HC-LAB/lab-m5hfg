"""Start a fresh detached capsule run using the explicit paper/code device policy."""
import argparse
import json
import os
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--smoke', action='store_true')
opts = parser.parse_args()
here = Path(__file__).resolve().parent
out = here / ('results/capsule_paper_device_policy_20261102' + ('_smoke' if opts.smoke else ''))
out.mkdir(parents=True, exist_ok=True)
if (out / 'launcher.json').exists():
    previous = json.loads((out / 'launcher.json').read_text())
    try:
        os.kill(previous['pid'], 0)
    except ProcessLookupError:
        pass
    else:
        raise RuntimeError('Previous launcher PID is still alive; inspect before starting')
if (out / 'summary.json').exists() or (out / 'train_timings.csv').exists():
    raise RuntimeError('Existing results must not be overwritten')
command = ['bash', str(here / 'run_capsule_devices.sh'), '--paper-device-policy']
if opts.smoke:
    command.append('--smoke')
with (out / 'run.log').open('ab', buffering=0) as log:
    process = subprocess.Popen(command, cwd=here, stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True, close_fds=True)
record = dict(pid=process.pid, command=command, log=str(out / 'run.log'))
(out / 'launcher.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record))
