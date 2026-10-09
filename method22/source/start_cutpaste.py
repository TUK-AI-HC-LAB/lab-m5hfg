import json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent;proof=root/'results/mvtec_scratch3way_seed42_20261008/verification.json'
if proof.exists() and json.loads(proof.read_text())['status']=='passed':raise SystemExit('CutPaste already completed')
if subprocess.run(['pgrep','-f','run_cutpaste.py'],stdout=subprocess.DEVNULL).returncode==0:raise SystemExit('CutPaste already running')
log=Path('/home/test/cutpaste_results/run_20261008.log');log.parent.mkdir(exist_ok=True)
with log.open('a') as f:p=subprocess.Popen(['bash',str(root/'run_cutpaste.sh')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,cwd=root)
(log.parent/'run_20261008.pid').write_text(str(p.pid));print('Started CutPaste PID',p.pid)
