"""Wait for eval lanes, run RsCIN, train missing backbone, isolated benchmarks."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
PYTHON='/home/test/miniforge3/envs/patchcore-gpu/bin/python'
env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
def run(script,*args):
    print('START',script,args,flush=True)
    subprocess.run([PYTHON,'-u',str(ROOT/script),*args],env=env,check=True)

tags=['mvtec_zero_shot_20261006','visa_0shot_seed42_20261006','btad_0shot_seed42_20261006']
tags += [f'{ds}_4shot_seed{s}_20261006' for ds in ['mvtec','visa','btad'] for s in [42,43,44]]
done=set()
while True:
    all_done=True
    for tag in tags:
        state=ROOT/'results'/tag/'environment.json'
        if not state.exists(): all_done=False;continue
        status=json.loads(state.read_text())['status']
        if status=='failed': raise RuntimeError((tag,json.loads(state.read_text())))
        if status!='completed': all_done=False;continue
        if tag.startswith('btad') or tag in done: continue
        feature_tag='mvtec_4shot_seed42_20261006' if tag=='mvtec_zero_shot_20261006' else tag
        feature_state=ROOT/'results'/feature_tag/'environment.json'
        if not feature_state.exists() or json.loads(feature_state.read_text())['status']!='completed': continue
        run('evaluate_aprilgan_rscin.py','--tag',tag,'--feature-tag',feature_tag)
        done.add(tag)
    run('summarize_aprilgan_remaining.py')
    run('../../method10/source/build_musc_reproduction_tables.py')
    if all_done: break
    time.sleep(45)
def completed(relative, filename='environment.json'):
    state = ROOT/'results'/relative/filename
    return state.exists() and json.loads(state.read_text()).get('status') == 'completed'

if not completed('benchmark_ViT-L-14-336_20261006', 'benchmark.json'):
    run('benchmark_aprilgan.py')
if not completed('train_vit_b16_plus_240_20261006'):
    run('train_aprilgan_small_backbone.py')
small_tag='mvtec_0shot_vit_b16_plus_240_20261006'
if not completed(small_tag):
    run('run_aprilgan_experiment.py','--dataset','mvtec','--mode','zero_shot','--seed','42',
        '--data','/home/test/data/mvtec','--tag',small_tag,
        '--output','/home/test/aprilgan_results/'+small_tag,'--backbone','ViT-B-16-plus-240',
        '--checkpoint','/home/test/aprilgan_results/train_vit_b16_plus_240_20261006/epoch_15.pth')
if not completed('benchmark_ViT-B-16-plus-240_20261006', 'benchmark.json'):
    run('benchmark_aprilgan.py','--backbone','ViT-B-16-plus-240','--checkpoint',
        '/home/test/aprilgan_results/train_vit_b16_plus_240_20261006/epoch_15.pth')
run('summarize_aprilgan_remaining.py')
run('../../method10/source/build_musc_reproduction_tables.py')
run('verify_aprilgan_recovery.py')
run('verify_aprilgan_remaining.py')
print('ALL APRIL-GAN REMAINING JOBS COMPLETED',flush=True)
