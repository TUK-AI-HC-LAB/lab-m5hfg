"""Pinned VT-ADL and original BTAD dataset paths."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parent
REPO=Path('/home/test/VT-ADL')
COMMIT='20b58e2dd810e1d747b9a4132899cc474691377e'
DATA=Path('/home/test/data/btad_original/BTech_Dataset_transformed')
OUT=ROOT/'results/btad_seed123_20261008'
RAW=Path('/home/test/vtadl_results/btad_seed123_20261008')
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
def metrics(y,s,masks,maps):
    return dict(image_auroc=float(roc_auc_score(y,s)),pixel_auroc=float(roc_auc_score(masks.ravel(),maps.ravel())))
