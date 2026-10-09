"""Execute pinned official test(), capturing exact outputs before rounding."""
import ast
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve
from skimage import measure

ROOT = Path(__file__).resolve().parent
REPO = Path('/home/test/DRAEM')
DATA = Path('/home/test/data/mvtec')
WEIGHTS = Path('/home/test/draem_weights/DRAEM_checkpoints')
TAG = 'mvtec_public_20261007'
RAW = Path('/home/test/draem_results') / TAG
OUT = ROOT / 'results' / TAG
BASE = 'DRAEM_seg_large_ae_large_0.0001_800_bs8'
CATEGORIES = ['capsule','bottle','carpet','leather','pill','transistor','tile','cable',
              'zipper','toothbrush','metal_nut','hazelnut','screw','grid','wood']
KEYS = ['image_auroc','image_f1_max','image_ap','pixel_auroc','pixel_f1_max','pixel_ap','aupro']
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO))
OUT.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)
if (OUT/'environment.json').exists():
    raise SystemExit('Existing evidence: use a fresh tag instead of overwriting.')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def csvwrite(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def f1max(labels,scores):
    p,r,_=precision_recall_curve(labels,scores)
    return float(np.nanmax(np.divide(2*p*r,p+r,out=np.zeros_like(p),where=(p+r)!=0)))

# Import the unchanged official test dataset class via AST. imgaug is only
# needed by the training class, and is incompatible with current NumPy 2.
# No test-loader statements or image/mask transforms are changed.
loader_path=REPO/'data_loader.py'
loader_tree=ast.parse(loader_path.read_text())
test_class=next(n for n in loader_tree.body if isinstance(n,ast.ClassDef) and n.name=='MVTecDRAEMTestDataset')
loader=types.ModuleType('data_loader')
loader.__dict__.update(os=os,np=np,torch=torch,cv2=cv2,glob=__import__('glob'),Dataset=torch.utils.data.Dataset)
exec(compile(ast.Module(body=[test_class],type_ignores=[]),str(loader_path),'exec'),loader.__dict__)
sys.modules['data_loader']=loader
import test_DRAEM as official

# AUPRO is supplementary, using the same 200-threshold function as APRIL-GAN.
# DRAEM's four native AUROC/AP metrics still execute in official.test().
pro_path=Path('/home/test/VAND-APRIL-GAN/test.py')
pro_tree=ast.parse(pro_path.read_text())
pro_fn=next(n for n in pro_tree.body if isinstance(n,ast.FunctionDef) and n.name=='cal_pro_score')
pro_scope=dict(np=np,measure=measure,auc=auc)
exec(compile(ast.Module(body=[pro_fn],type_ignores=[]),str(pro_path),'exec'),pro_scope)

torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
# Eval-only stochastic display selection does not affect predictions.
np.random.seed(42)
env=dict(status='running',official_repo='https://github.com/VitjanZ/DRAEM',
    commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
    python=sys.version,platform=platform.platform(),gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,
    packages={k:importlib.metadata.version(k) for k in ['torch','torchvision','numpy','opencv-python','scikit-learn','scikit-image']},
    official_source_sha256={k:sha(REPO/k) for k in ['test_DRAEM.py','data_loader.py','model_unet.py']},
    wrapper_sha256=sha(__file__),base_model_name=BASE,checkpoint_source='official public Google Drive archive',
    checkpoint_archive_sha256=sha(WEIGHTS.parent/'DRAEM_checkpoints.zip'),
    checkpoint_hashes={p.name:sha(p) for p in sorted(WEIGHTS.glob('*.pckl'))},
    training_executed=False,training_epochs_verified=False,checkpoint_filename_epochs=800,
    image_size=256,image_channels='OpenCV BGR',normalization='divide by 255',precision='FP32; no AMP',
    tf32=False,batch_size=1,mask_resize='OpenCV default INTER_LINEAR; divide by 255 then uint8 truncation for pixel metrics',
    image_score='max of avg_pool2d(anomaly softmax channel, kernel=21, stride=1, padding=10)',
    pixel_score='unpooled anomaly softmax channel',dataset_root=str(DATA),raw_output=str(RAW),
    compatibility='AST imports unchanged official test class, omitting unused training-only imgaug import',
    inference_context='torch.no_grad (official test omits it); eval models unchanged',
    supplementary_metrics=['image_f1_max','pixel_f1_max','aupro'],
    aupro_source=str(pro_path),aupro_source_sha256=sha(pro_path),aupro_steps=200,aupro_fpr_cutoff=.3,
    command=sys.argv,source_edits=False)

def state():
    for directory in [OUT,RAW]: (directory/'environment.json').write_text(json.dumps(env,indent=2))

state()
rows=[]
official_file=Path(official.__file__).resolve()
source=official_file.read_text().splitlines()
capture_line=next(i+1 for i,l in enumerate(source) if l.strip()=='print(obj_name)')

def trace(frame,event,arg):
    if frame.f_code.co_filename!=str(official_file) or frame.f_code.co_name!='test': return None
    if event=='line' and frame.f_lineno==capture_line:
        local=frame.f_locals
        category=local['obj_name']
        labels=local['anomaly_score_gt'].copy()
        scores=local['anomaly_score_prediction'].copy()
        n=len(labels)
        maps=local['total_pixel_scores'].reshape(n,256,256)
        masks=local['total_gt_pixel_scores'].reshape(n,256,256)
        assert np.isfinite(scores).all() and np.isfinite(maps).all()
        folder=RAW/category;folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/'raw_predictions.npz',image_labels=labels,image_scores=scores,
                            anomaly_maps=maps,masks=masks)
        images=local['dataset'].images
        image_rows=[dict(image_path=p,label=int(label),score=float(score),sha256=sha(p))
                    for p,label,score in zip(images,labels,scores)]
        assert len(image_rows)==n
        e=OUT/category;e.mkdir(exist_ok=True)
        csvwrite(e/'image_scores.csv',image_rows)
        csvwrite(folder/'image_scores.csv',image_rows)
        print('NATIVE METRICS CAPTURED',category,n,'computing supplementary metrics',flush=True)
        row=dict(category=category,n_images=n,image_auroc=float(local['auroc']),
                 image_f1_max=f1max(labels,scores),image_ap=float(local['ap']),
                 pixel_auroc=float(local['auroc_pixel']),pixel_f1_max=f1max(masks.ravel(),maps.ravel()),
                 pixel_ap=float(local['ap_pixel']),aupro=float(pro_scope['cal_pro_score'](masks,maps)))
        rows.append(row)
        csvwrite(OUT/'category_metrics.csv',rows);csvwrite(RAW/'category_metrics.csv',rows)
        (e/'raw_manifest.json').write_text(json.dumps(dict(path=str(folder/'raw_predictions.npz'),
            sha256=sha(folder/'raw_predictions.npz'),n_images=n,map_shape=list(maps.shape),
            map_dtype=str(maps.dtype),mask_dtype=str(masks.dtype)),indent=2))
        print('CATEGORY COMPLETE',json.dumps(row),flush=True)
    return trace

start=time.monotonic()
os.chdir(RAW)
torch.cuda.reset_peak_memory_stats()
try:
    sys.settrace(trace)
    with torch.cuda.device(0),torch.no_grad():
        official.test(CATEGORIES,str(DATA)+'/',str(WEIGHTS),BASE)
    sys.settrace(None)
    assert len(rows)==15 and sum(r['n_images'] for r in rows)==1725
    assert not subprocess.check_output(['git','-C',str(REPO),'diff','--name-only'],text=True).strip()
    mean=dict(category='macro_mean',n_images=1725,**{k:float(np.mean([r[k] for r in rows])) for k in KEYS})
    csvwrite(OUT/'mean_metrics.csv',[mean]);csvwrite(RAW/'mean_metrics.csv',[mean])
    env.update(status='completed',wall_seconds=time.monotonic()-start,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    state()
    print('ALL COMPLETE',json.dumps(mean),flush=True)
except Exception as error:
    sys.settrace(None);env.update(status='failed',error=repr(error),wall_seconds=time.monotonic()-start);state();raise
