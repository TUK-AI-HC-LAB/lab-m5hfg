import json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent;proof=root/'results/btad_res18_seed0_20261008/verification.json'
if proof.exists() and json.loads(proof.read_text())['status']=='passed':raise SystemExit('PyramidFlow already completed')
if subprocess.run(['pgrep','-f','run_pyramidflow.py'],stdout=subprocess.DEVNULL).returncode==0:raise SystemExit('PyramidFlow already running')
log=Path('/home/test/pyramidflow_results/run_20261008.log');log.parent.mkdir(exist_ok=True)
with log.open('a') as f:p=subprocess.Popen(['bash',str(root/'run_pyramidflow.sh')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,cwd=root)
(log.parent/'run_20261008.pid').write_text(str(p.pid));print('Started PyramidFlow PID',p.pid)
