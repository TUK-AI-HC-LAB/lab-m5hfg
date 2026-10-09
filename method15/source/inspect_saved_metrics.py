"""Inspect retained metrics after an interrupted validation; no model training."""
import ast
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import auc,average_precision_score,precision_recall_curve,roc_auc_score
from skimage import measure
from prepare_acr import ROOT,RAW,OUT,KEYS
tree=ast.parse((ROOT/'run_acr.py').read_text())
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['f1','metrics']]
scope=dict(np=np,auc=auc,average_precision_score=average_precision_score,precision_recall_curve=precision_recall_curve,roc_auc_score=roc_auc_score)
exec(compile(ast.Module(body=functions,type_ignores=[]),'metrics','exec'),scope)
pro_path=Path('/home/test/VAND-APRIL-GAN/test.py')
node=next(n for n in ast.parse(pro_path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='cal_pro_score')
pro_scope=dict(np=np,measure=measure,auc=auc)
exec(compile(ast.Module(body=[node],type_ignores=[]),str(pro_path),'exec'),pro_scope)
category=sys.argv[1] if len(sys.argv)>1 else 'leather'
with np.load(RAW/f'{category}_predictions.npz') as z:
    labels,scores,maps,masks=[z[k] for k in ['image_labels','image_scores','anomaly_maps','masks']]
row=dict(category=category,n_images=len(labels),**scope['metrics'](labels,scores,masks,maps),aupro=float(pro_scope['cal_pro_score'](masks,maps)))
print(json.dumps(row),flush=True)
(OUT/category/'metric_diagnostic.json').write_text(json.dumps(dict(raw_metrics=row,out_of_range={k:row[k] for k in KEYS if not np.isfinite(row[k]) or not 0<=row[k]<=1}),indent=2))
