import csv,hashlib,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
ROOT=Path(__file__).resolve().parent
REPO=Path('/home/test/CutPaste')
COMMIT='10d8bf71df76d3a97f0106efee1d76f81d983149'
DATA=Path('/home/test/data/mvtec')
RAW=Path('/home/test/cutpaste_results/mvtec_scratch3way_seed42_20261008')
OUT=ROOT/'results/mvtec_scratch3way_seed42_20261008'
CATS=['bottle','cable','capsule','carpet','grid','hazelnut','leather','metal_nut','pill','screw','tile','toothbrush','transistor','wood','zipper']
ALIGNED=['bottle','cable','capsule','metal_nut','pill','toothbrush','transistor','zipper']
STEPS=256*256
NORM=T.Compose([T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,rows):
    with open(p,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def read(p):
    with open(p,encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def manifest(cat,phase):
    rows=[]
    for p in sorted((DATA/cat/phase).glob('*/*.png')):
        label=int(phase=='test' and p.parent.name!='good');mp=DATA/cat/'ground_truth'/p.parent.name/(p.stem+'_mask.png') if label else None
        assert mp is None or mp.exists()
        rows.append(dict(image_path=str(p),sha256=sha(p),label=label,mask_path=str(mp) if mp else '',mask_sha256=sha(mp) if mp else ''))
    assert rows
    return rows
def image(p):
    with Image.open(p) as im:return im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)
def masks(rows):
    result=[]
    for r in rows:
        m=np.zeros((256,256),np.uint8)
        if r['mask_path']:
            with Image.open(r['mask_path']) as im:m=(np.array(im.convert('L').resize((256,256),Image.Resampling.NEAREST))>0).astype(np.uint8)
        result.append(m)
    return np.stack(result)
