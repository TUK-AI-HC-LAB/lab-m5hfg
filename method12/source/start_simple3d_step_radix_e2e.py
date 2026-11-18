"""Keep the bounded four-worker follow-up alive after the launcher returns."""
import subprocess
from pathlib import Path

here=Path(__file__).resolve().parent
root=here/'results/step_radix_e2e_20261118'
root.mkdir(parents=True,exist_ok=True)
with (root/'launcher.pid').open('x') as marker,(root/'controller.log').open('ab',buffering=0) as log:
    p=subprocess.Popen(['bash',str(here/'run_simple3d_step_radix_e2e.sh')],stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    marker.write(str(p.pid)+'\n')
print(p.pid)
