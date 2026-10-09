"""Detached one-time SPADE run; no duplicate workers or completed reruns."""
import json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
proof=root/'results/btad_k50_seed42_20261008/verification.json'
if proof.exists() and json.loads(proof.read_text())['status']=='passed':raise SystemExit('SPADE already completed')
if subprocess.run(['pgrep','-f','run_spade.py'],stdout=subprocess.DEVNULL).returncode==0:raise SystemExit('SPADE already running')
log=Path('/home/test/spade_results/run_20261008.log');log.parent.mkdir(exist_ok=True)
with log.open('a') as f:p=subprocess.Popen(['bash',str(root/'run_spade.sh')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,cwd=root)
(log.parent/'run_20261008.pid').write_text(str(p.pid));print('Started SPADE PID',p.pid)
