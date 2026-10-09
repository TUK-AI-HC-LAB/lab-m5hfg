import argparse,json,subprocess,time,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--lane',choices=['mvtec','visa'],required=True);o=p.parse_args()
source=Path(__file__).resolve().parent
out=Path('/home/test/aprilgan_results')
wait='mvtec_4shot_seed42_20261006' if o.lane=='mvtec' else 'visa_0shot_seed42_20261006'
print('Waiting for',wait,flush=True)
while True:
 state=json.loads((out/wait/'environment.json').read_text())
 if state['status']=='completed': break
 if state['status']=='failed': raise RuntimeError(state)
 time.sleep(5)
jobs=([( 'mvtec','few_shot',s) for s in [43,44]] + [('btad','zero_shot',42)] + [('btad','few_shot',s) for s in [42,43,44]]) if o.lane=='mvtec' else [('visa','few_shot',s) for s in [42,43,44]]
for ds,mode,seed in jobs:
 tag=f'{ds}_{4 if mode=="few_shot" else 0}shot_seed{seed}_20261006'
 folder=out/tag
 if (folder/'environment.json').exists():
  status=json.loads((folder/'environment.json').read_text())['status']
  if status=='completed': continue
  raise RuntimeError((tag,status))
 data={'mvtec':'/home/test/data/mvtec','visa':'/home/test/data/VisA_20220922','btad':'/home/test/data/btad_original/BTech_Dataset_transformed'}[ds]
 assert Path(data,'meta.json').exists(),data
 cmd=['/home/test/miniforge3/envs/patchcore-gpu/bin/python','-u',str(source/'run_aprilgan_experiment.py'),'--dataset',ds,'--mode',mode,'--seed',str(seed),'--data',data,'--tag',tag,'--output',str(folder)]
 print('START',tag,flush=True)
 env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
 with (out/(tag+'.log')).open('w') as log:
  subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,env=env)
 print('COMPLETE',tag,flush=True)
print('LANE COMPLETE',o.lane,flush=True)