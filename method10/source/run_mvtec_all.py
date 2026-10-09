"""Sequential full-pool category evaluation; resume only verified complete runs."""
import csv, json, os, shutil, subprocess, sys, time
from pathlib import Path

SOURCE = Path(__file__).resolve().parent
ROOT = Path('/home/test/musc_results/mvtec_all_paper_20261002')
SMALL = SOURCE / 'results/mvtec_all_paper_20261002'
DATA = Path('/home/test/data/mvtec')
CATEGORIES = ['bottle','cable','capsule','carpet','grid','hazelnut','leather','metal_nut','pill','screw','tile','toothbrush','transistor','wood','zipper']
COMMIT = 'b76b93da8bd3096a99964a96ae29d46f197a0651'
METRICS = ['image_auroc','image_f1_max','image_ap','pixel_auroc','pixel_f1_max','pixel_ap','aupro']
ROOT.mkdir(parents=True, exist_ok=True)
SMALL.mkdir(parents=True, exist_ok=True)
assert subprocess.check_output(['git','-C','/home/test/MuSc','rev-parse','HEAD'],text=True).strip() == COMMIT

def complete(path):
    required = ['environment.json','metrics.csv','image_scores.csv','config.json','raw_predictions.npz']
    if not all((path / n).exists() for n in required):
        return False
    env = json.loads((path / 'environment.json').read_text())
    return env.get('status') == 'completed' and env.get('commit') == COMMIT

audit = []
for category in CATEGORIES:
    images = sorted((DATA/category/'test').glob('*/*.png'))
    abnormal = [p for p in images if p.parent.name != 'good']
    missing = [str(p) for p in abnormal if not (DATA/category/'ground_truth'/p.parent.name/(p.stem+'_mask.png')).exists()]
    assert images and len(abnormal) < len(images) and not missing, (category, missing)
    audit.append(dict(category=category, test_images=len(images), normal=len(images)-len(abnormal), abnormal=len(abnormal), missing_masks=len(missing)))
with (SMALL/'dataset_audit.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(audit[0])); w.writeheader(); w.writerows(audit)

statuses=[]
start=time.perf_counter()
for category in CATEGORIES:
    result = Path('/home/test/musc_results/bottle_paper_20261002') if category == 'bottle' else ROOT/category
    reused = complete(result)
    category_start=time.perf_counter()
    print(f'START {category} reuse={reused}',flush=True)
    exit_code=0
    if not reused:
        result.mkdir(parents=True,exist_ok=True)
        # A partial metrics file must not masquerade as a completed run.
        if (result/'metrics.csv').exists():
            raise RuntimeError(f'Incomplete result needs a fresh output directory: {result}')
        with (result/'run.log').open('w') as log:
            proc=subprocess.run([sys.executable,'-u',str(SOURCE/'run_musc_local.py'),'--category',category,'--output',str(result)],stdout=log,stderr=subprocess.STDOUT)
        exit_code=proc.returncode
    succeeded = exit_code==0 and complete(result)
    state=dict(category=category,status='completed' if succeeded else 'failed',reused=reused,exit_code=exit_code,elapsed_this_batch_seconds=time.perf_counter()-category_start,raw_output=str(result))
    statuses.append(state)
    with (SMALL/'status.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(state)); w.writeheader(); w.writerows(statuses)
    if succeeded:
        destination=SMALL/category
        destination.mkdir(exist_ok=True)
        for name in ['metrics.csv','image_scores.csv','environment.json','config.json']:
            shutil.copy2(result/name,destination/name)
        row=next(csv.DictReader((result/'metrics.csv').open()))
        print(f"DONE {category} image_AUROC={float(row['image_auroc'])*100:.4f} pixel_AUROC={float(row['pixel_auroc'])*100:.4f} elapsed={state['elapsed_this_batch_seconds']:.1f}s",flush=True)
    else:
        print(f'FAILED {category} code={exit_code} log={result}/run.log',flush=True)

rows=[]
for category in CATEGORIES:
    path=SMALL/category/'metrics.csv'
    if path.exists() and next(s for s in statuses if s['category']==category)['status']=='completed':
        rows.append(next(csv.DictReader(path.open())))
with (SMALL/'category_metrics.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['dataset','category','n_images']+METRICS); w.writeheader(); w.writerows(rows)
if len(rows)==len(CATEGORIES):
    mean=dict(dataset='mvtec_ad',category='macro_mean',n_images=sum(int(r['n_images']) for r in rows))
    mean.update({k:sum(float(r[k]) for r in rows)/len(rows) for k in METRICS})
    with (SMALL/'mean_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(mean)); w.writeheader(); w.writerow(mean)
    print('ALL_COMPLETED',json.dumps(mean),flush=True)
else:
    print(f'INCOMPLETE {len(rows)}/{len(CATEGORIES)}',flush=True)
(SMALL/'batch_summary.json').write_text(json.dumps(dict(completed_categories=len(rows),requested_categories=len(CATEGORIES),wall_seconds_this_batch=time.perf_counter()-start,bottle_reused=True),indent=2))
sys.exit(0 if len(rows)==len(CATEGORIES) else 1)
