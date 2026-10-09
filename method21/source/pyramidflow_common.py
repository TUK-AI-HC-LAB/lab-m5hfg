import csv,hashlib,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
ROOT=Path(__file__).resolve().parent
REPO=Path('/home/test/PyramidFlow')
DEPENDENCY=Path('/home/test/PyramidFlow-dependency')
COMMIT='c463b1cb0c2b084cdbae290234e3f8657298e981'
DEP_COMMIT='fa322f9175966ebed2759b0422460b401c318b05'
DATA=Path('/home/test/data/btad_original/BTech_Dataset_transformed')
RAW=Path('/home/test/pyramidflow_results/btad_res18_seed0_20261008')
OUT=ROOT/'results/btad_res18_seed0_20261008'
XFORM=T.Compose([T.Resize((1024,1024)),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
MFORM=T.Compose([T.ToPILImage(),T.Resize((256,256)),T.ToTensor()])
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
    paths=sorted(p for p in (DATA/cat/phase).glob('*/*') if p.is_file() and p.suffix.lower() in ['.bmp','.png','.jpg','.jpeg','.tif','.tiff'])
    for p in paths:
        y=int(phase=='test' and p.parent.name!='ok');mp=None
        if y:
            candidates=list((DATA/cat/'ground_truth'/p.parent.name).glob(p.stem+'.*'));assert len(candidates)==1
            mp=candidates[0]
        rows.append(dict(image_path=str(p),sha256=sha(p),label=y,mask_path=str(mp) if mp else '',mask_sha256=sha(mp) if mp else ''))
    return rows
def mask(r):
    if not r['mask_path']:return torch.zeros(1,256,256,dtype=torch.int32)
    with Image.open(r['mask_path']) as im:a=np.array(im.convert('L'),dtype=np.float32)/255
    return MFORM(a).round().int()
def inputs(rows):return np.stack([mask(r).numpy()[0] for r in rows])
