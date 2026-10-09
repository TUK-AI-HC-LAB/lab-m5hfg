"""One-time detached VT-ADL launch; no monitoring process."""
import subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
log=Path('/home/test/vtadl_results/run_20261008.log');log.parent.mkdir(exist_ok=True)
# Refuse a duplicate worker in addition to the verified-run guard.
running=subprocess.check_output(['pgrep','-af','run_vtadl.py'],text=True) if subprocess.run(['pgrep','-f','run_vtadl.py'],stdout=subprocess.DEVNULL).returncode==0 else ''
if running:raise SystemExit('Existing VT-ADL process: '+running)
with log.open('a') as f:p=subprocess.Popen(['bash',str(root/'run_vtadl.sh')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,cwd=root)
(log.parent/'run_20261008.pid').write_text(str(p.pid));print('Started VT-ADL PID',p.pid)
