"""One GPU/RAM-heavy task at a time; exact arrays cached to disk for evidence."""
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
PY='/home/test/miniforge3/envs/patchcore-gpu/bin/python'
RAW=Path('/home/test/aprilgan_results')
env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
jobs=[('mvtec','few_shot',42),('visa','zero_shot',42),('visa','few_shot',42)]
jobs += [(ds,'few_shot',s) for ds in ['mvtec','visa'] for s in [43,44]]
jobs += [('btad','zero_shot',42)]+[('btad','few_shot',s) for s in [42,43,44]]
data={'mvtec':'/home/test/data/mvtec','visa':'/home/test/data/VisA_20220922','btad':'/home/test/data/btad_original/BTech_Dataset_transformed'}

def run(script,*args):
    subprocess.run([PY,'-u',str(ROOT/script),*args],check=True,env=env)

for ds,mode,seed in jobs:
    tag=f'{ds}_{4 if mode=="few_shot" else 0}shot_seed{seed}_20261006'
    raw=RAW/tag
    statepath=ROOT/'results'/tag/'environment.json'
    if statepath.exists():
        state=json.loads(statepath.read_text())
        if state['status']=='completed': continue
        raise RuntimeError((tag,state['status']))
    print('SERIAL START',tag,flush=True)
    cmd=[PY,'-u',str(ROOT/'run_aprilgan_experiment.py'),'--dataset',ds,'--mode',mode,
         '--seed',str(seed),'--data',data[ds],'--tag',tag,'--output',str(raw)]
    with (RAW/(tag+'.log')).open('w') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,env=env)
    print('SERIAL COMPLETE',tag,flush=True)
    if tag=='visa_0shot_seed42_20261006': run('verify_aprilgan_recovery.py')
    if ds!='btad':
        run('evaluate_aprilgan_rscin.py','--tag',tag)
        if tag=='mvtec_4shot_seed42_20261006':
            run('evaluate_aprilgan_rscin.py','--tag','mvtec_zero_shot_20261006','--feature-tag',tag)
    run('summarize_aprilgan_remaining.py')
    run('../../method10/source/build_musc_reproduction_tables.py')
run('finish_aprilgan_remaining.py')
print('SERIAL EXPERIMENTS AND BENCHMARKS COMPLETE',flush=True)
