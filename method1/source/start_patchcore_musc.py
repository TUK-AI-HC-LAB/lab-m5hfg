import json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
p=root/'results/musc_comparison_20261010/verification.json'
if p.exists() and json.loads(p.read_text())['status']=='passed':raise SystemExit('PatchCore comparisons already complete')
if subprocess.run(['pgrep','-f','run_patchcore_musc.py'],stdout=subprocess.DEVNULL).returncode==0:raise SystemExit('PatchCore already running')
log=Path('/home/test/patchcore_results/run_musc_20261010.log');log.parent.mkdir(parents=True,exist_ok=True)
with log.open('a') as f:
    process=subprocess.Popen(['bash',str(root/'run_patchcore_musc.sh')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,cwd=root)
(log.parent/'run_musc_20261010.pid').write_text(str(process.pid));print('Started PatchCore PID',process.pid)
