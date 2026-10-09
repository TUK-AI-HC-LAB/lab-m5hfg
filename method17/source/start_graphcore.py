"""Start once as a detached worker; retain output and PID."""
import os
import subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
log=Path('/home/test/graphcore_results/run_20261008.log')
log.parent.mkdir(exist_ok=True)
with log.open('a') as f:
    p=subprocess.Popen(['bash',str(root/'run_graphcore.sh')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,cwd=root)
(log.parent/'run_20261008.pid').write_text(str(p.pid))
print('Started GraphCore PID',p.pid)
