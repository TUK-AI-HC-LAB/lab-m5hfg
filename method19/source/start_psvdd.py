"""Launch once, refuse duplicate training workers."""
import subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
if subprocess.run(['pgrep','-f','run_psvdd.py'],stdout=subprocess.DEVNULL).returncode==0:raise SystemExit('P-SVDD already running')
log=Path('/home/test/psvdd_results/run_20261008.log');log.parent.mkdir(exist_ok=True)
with log.open('a') as f:p=subprocess.Popen(['bash',str(root/'run_psvdd.sh')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,cwd=root)
(log.parent/'run_20261008.pid').write_text(str(p.pid));print('Started P-SVDD PID',p.pid)
