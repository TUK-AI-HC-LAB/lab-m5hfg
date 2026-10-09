"""Pinned author-benchmark GraphCore runtime and auditable metric helpers."""
import ast
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
from skimage import measure

ROOT=Path(__file__).resolve().parent
REPO=Path('/home/test/open-iad')
COMMIT='05044fedab142ce4bfd71bb618b3200c0e43f198'
RAW=Path('/home/test/graphcore_results/mvtec_pvig_fp32_20261008')
OUT=ROOT/'results/mvtec_pvig_fp32_20261008'
WEIGHT=Path('/home/test/graphcore_assets/pvig_ti_78.5.pth.tar')
KEYS=['image_auroc','image_f1_max','image_ap','pixel_auroc','pixel_f1_max','pixel_ap','aupro']
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):
    with open(p,encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(p,rows):
    with open(p,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def classification(y,s):
    p,r,_=precision_recall_curve(y,s)
    return dict(image_auroc=float(roc_auc_score(y,s)),image_f1_max=float(np.divide(2*p*r,p+r,out=np.zeros_like(p),where=p+r>0).max()),image_ap=float(average_precision_score(y,s)))
def pro_function():
    path=Path('/home/test/VAND-APRIL-GAN/test.py')
    n=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='cal_pro_score')
    scope=dict(np=np,measure=measure,auc=auc)
    exec(compile(ast.Module(body=[n],type_ignores=[]),str(path),'exec'),scope)
    return scope['cal_pro_score'],sha(path)
def metrics(y,s,masks,maps):
    px=classification(masks.ravel(),maps.ravel())
    pro,_=pro_function()
    return dict(**classification(y,s),**{k.replace('image_','pixel_'):v for k,v in px.items()},aupro=float(pro(masks,maps)))
def native_class():
    """Use original train/prediction methods; omit unrelated benchmark shell."""
    import torch,cv2,faiss,math
    import torch.nn.functional as F
    from scipy.ndimage import gaussian_filter
    from sklearn.random_projection import SparseRandomProjection
    sys.path.insert(0,str(REPO))
    from models._patchcore.kcenter_greedy import KCenterGreedy
    path=REPO/'arch/graphcore.py'
    source=path.read_text()
    # Native fold buffer and NumPy processing are CPU-only. Transfer features
    # before concat; graph backbone still runs on CUDA. No numerical cast.
    source=source.replace('GraphCore.embedding_concate(embeddings[0], embeddings[1])','GraphCore.embedding_concate(embeddings[0].cpu(), embeddings[1].cpu())')
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='GraphCore')
    class Base:
        def clear_all_list(self):
            for name in ['img_pred_list','img_gt_list','pixel_pred_list','pixel_gt_list','img_path_list']:setattr(self,name,[])
    scope=dict(torch=torch,F=F,cv2=cv2,np=np,os=__import__('os'),faiss=faiss,math=math,gaussian_filter=gaussian_filter,ModelBase=Base,KCenterGreedy=KCenterGreedy,SparseRandomProjection=SparseRandomProjection)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    return scope['GraphCore'],SparseRandomProjection
