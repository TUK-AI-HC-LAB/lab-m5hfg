"""Inference-only Table 13 measurement on all MVTec test images, isolated GPU."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

p=argparse.ArgumentParser()
p.add_argument('--backbone',default='ViT-L-14-336')
p.add_argument('--checkpoint',default='/home/test/VAND-APRIL-GAN/exps/pretrained/visa_pretrained.pth')
opt=p.parse_args()
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'/('benchmark_'+opt.backbone+'_20261006')
if (OUT/'benchmark.json').exists(): raise SystemExit('Existing benchmark preserved.')
OUT.mkdir(exist_ok=True)
repo=Path('/home/test/VAND-APRIL-GAN');os.chdir(repo);sys.path.insert(0,str(repo));sys.dont_write_bytecode=True
import test as official
small=opt.backbone=='ViT-B-16-plus-240'
args=SimpleNamespace(data_path='/home/test/data/mvtec',save_path='/home/test/aprilgan_results/benchmark_'+opt.backbone,
    checkpoint_path=opt.checkpoint,config_path=str(repo/'open_clip/model_configs'/f'{opt.backbone}.json'),
    dataset='mvtec',model=opt.backbone,pretrained='laion400m_e31' if small else 'openai',
    features_list=[3,6,9,12] if small else [6,12,18,24],few_shot_features=[3,6,9,12] if small else [6,12,18,24],
    image_size=518,mode='zero_shot',k_shot=0,seed=42)
official.setup_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
filename=str(Path(official.__file__).resolve())
lines=Path(filename).read_text().splitlines()
start=next(i+1 for i,l in enumerate(lines) if l.strip()=="image = items['img'].to(device)")
end=next(i+1 for i,l in enumerate(lines) if l.strip()=="results['anomaly_maps'].append(anomaly_map)")+1
stop=next(i+1 for i,l in enumerate(lines) if l.strip()=='table_ls = []')
times=[];clock=None;setup_allocated=0;setup_reserved=0
class Finished(Exception): pass
def trace(frame,event,arg):
    global clock,setup_allocated,setup_reserved
    if frame.f_code.co_filename!=filename or frame.f_code.co_name!='test': return None
    if event=='line':
        if frame.f_lineno==start:
            torch.cuda.synchronize()
            if not times:
                setup_allocated=torch.cuda.max_memory_allocated()
                setup_reserved=torch.cuda.max_memory_reserved()
                torch.cuda.reset_peak_memory_stats()
            clock=time.perf_counter()
        elif frame.f_lineno==stop: raise Finished()
        elif clock is not None and frame.f_lineno>=end:
            torch.cuda.synchronize();times.append(time.perf_counter()-clock);clock=None
    return trace
try:
    torch.cuda.reset_peak_memory_stats()
    sys.settrace(trace);official.test(args)
    raise AssertionError('Expected end-of-inference stop.')
except Finished:
    sys.settrace(None)
assert len(times)==1725,len(times)
result=dict(status='completed',backbone=opt.backbone,args=vars(args),n_images=len(times),
    mean_ms=1000*float(np.mean(times[1:])),first_image_excluded=True,mean_ms_all=1000*float(np.mean(times)),
    gpu_allocated_mib=max(setup_allocated,torch.cuda.max_memory_allocated())/1024**2,
    gpu_reserved_mib=max(setup_reserved,torch.cuda.max_memory_reserved())/1024**2,
    inference_only_gpu_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
    memory_scope='maximum PyTorch allocated/reserved throughout model/text setup and inference; MiB=bytes/1024^2',
    gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
    checkpoint_sha256=hashlib.sha256(Path(opt.checkpoint).read_bytes()).hexdigest(),
    official_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    timing_scope='H2D and official zero-shot scoring; excludes loader, visualization, model/text setup, metrics, evidence writes',
    synchronization='CUDA synchronized per-image boundaries; Python line trace',
    resource_condition='run only after all evaluation lanes complete; no competing task from this study',
    small_backbone_training='own VisA projection training; original small-backbone training details not published' if small else 'official public VisA checkpoint')
with (OUT/'per_image.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['index','milliseconds']);w.writerows((i,1000*v) for i,v in enumerate(times))
(OUT/'benchmark.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
