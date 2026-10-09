"""Pinned IGD runtime and compatibility edits, separate from official checkout."""
import difflib
import hashlib
import importlib
import inspect
import json
import os
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parent
REPO=Path('/home/test/IGD')
TAG=os.environ.get('IGD_RUN_TAG','mvtec_full_seed42_20261007')
ACCELERATED=os.environ.get('IGD_ACCELERATED','0')=='1'
RAW=Path('/home/test/igd_results')/TAG
RUNTIME=RAW/'runtime'
OUT=ROOT/'results'/TAG
DATA=Path('/home/test/data/mvtec')
COMMIT='1ce995214ef8adf09f6c5d3dc01ce6042b472a34'
CATEGORIES=['bottle','hazelnut','capsule','metal_nut','leather','pill','wood','carpet','tile','grid',
            'cable','transistor','toothbrush','screw','zipper']
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def prepare():
    OUT.mkdir(parents=True,exist_ok=True);RAW.mkdir(parents=True,exist_ok=True)
    edits=[];manifest=[]
    for path in sorted(REPO.rglob('*.py')):
        if '.git' in path.parts:continue
        rel=path.relative_to(REPO)
        source=path.read_text()
        modified=source.replace('/home/user/Documents/Public_Dataset/MVTec_AD',str(DATA))
        # Modern iterator API, identical iteration semantics.
        modified=modified.replace('train_data.next()', 'next(train_data)')
        if ACCELERATED:
            # channels-last tensors may be non-contiguous for view(); reshape
            # preserves the intended indexing and handles their strides.
            modified=modified.replace('.view(','.reshape(')
            modified=modified.replace('.to(device)', '.to(device, non_blocking=True)')
        target=RUNTIME/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(modified)
        manifest.append(dict(path=str(rel),official_sha256=sha(path),runtime_sha256=sha(target)))
        if modified!=source:
            edits.extend(difflib.unified_diff(source.splitlines(True),modified.splitlines(True),fromfile=str(rel),tofile='runtime/'+str(rel)))
    (OUT/'compatibility.patch').write_text(''.join(edits))
    (OUT/'runtime_sources.json').write_text(json.dumps(manifest,indent=2))
    encoder=RAW/'Encoder_KD_ckpt'
    if not encoder.exists():encoder.symlink_to(REPO/'Encoder_KD_ckpt')

def setup():
    os.environ.setdefault('MPLBACKEND','Agg')
    os.chdir(RAW)
    sys.path.insert(0,str(RUNTIME))
    sys.dont_write_bytecode=True
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=ACCELERATED
    torch.backends.cudnn.allow_tf32=ACCELERATED
    if ACCELERATED:
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark=True
        torch.backends.cudnn.deterministic=False
    # Library uses five-level guard even when IGD supplies four weights.
    # Only the size assertion is adapted; loss computations are unchanged.
    module=importlib.import_module('pytorch_msssim.ssim')
    source=inspect.getsource(module)
    adapted=source.replace('2 ** 4','2 ** (len(weights) - 1 if weights is not None else 4)')
    scope=dict(module.__dict__)
    exec(compile(adapted,str(RAW/'msssim_compat.py'),'exec'),scope)
    (RAW/'msssim_compat.py').write_text(adapted)
    import pytorch_msssim
    pytorch_msssim.ms_ssim=scope['ms_ssim']
    return pytorch_msssim

class SilentRange:
    def __init__(self,iterable,*a,**kw):self.iterable=iterable
    def __iter__(self):return iter(self.iterable)
    def set_description(self,*a,**kw):pass
