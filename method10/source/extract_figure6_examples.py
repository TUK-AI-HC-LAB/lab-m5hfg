"""CPU-only figure extraction from retained, verified MuSc predictions."""
from pathlib import Path
import csv, json, hashlib
import numpy as np
import torch
from PIL import Image
from matplotlib import colormaps

root=Path('/mnt/c/Users/test/Desktop/Codex/lab-m5hfg')
out=root/'method10/source/figures/figure6_local'
out.mkdir(parents=True,exist_ok=True)
evidence=root/'method10/source/results/tables3_7_seed42_20261012'
assert json.loads((evidence/'verification.json').read_text())['status']=='passed'
samples=[]
for dataset,category in [('mvtec_ad','bottle'),('mvtec_ad','metal_nut'),('mvtec_ad','pill'),('visa','pcb1'),('visa','cashew'),('visa','pipe_fryum')]:
    rows=list(csv.DictReader((evidence/dataset/category/'images.csv').open()))
    # Deterministic selection: the first anomalous image in the preserved loader order.
    i=next(i for i,r in enumerate(rows) if int(r['label'])==1)
    row=rows[i]; image_path=Path(row['image_path'])
    assert hashlib.sha256(image_path.read_bytes()).hexdigest()==row['sha256']
    raw=Path('/home/test/musc_results/tables3_7_seed42_20261012')/dataset/category/'patch_predictions.npz'
    with np.load(raw) as z:
        scores=z['s1'][i].reshape(37,37).copy()
        mask=z['masks'][i].copy()
    amap=torch.nn.functional.interpolate(torch.from_numpy(scores)[None,None],size=(518,518),mode='bilinear',align_corners=True)[0,0].numpy()
    normalized=(amap-amap.min())/(amap.max()-amap.min()+1e-12)
    heat=(colormaps['jet'](normalized)[...,:3]*255).astype(np.uint8)
    name=f'{dataset}_{category}'
    Image.open(image_path).convert('RGB').resize((518,518),Image.Resampling.BILINEAR).save(out/f'{name}_image.png')
    Image.fromarray((mask>0).astype(np.uint8)*255).save(out/f'{name}_gt.png')
    Image.fromarray(heat).save(out/f'{name}_musc.png')
    np.savez_compressed(out/f'{name}_figure_data.npz',patch_scores=scores,anomaly_map=amap,mask=mask)
    samples.append(dict(dataset=dataset,category=category,index=i,**row,raw_path=str(raw),condition='s1; r={1,3,5}; bottom30% mean',heatmap_normalization='per image min-max',aprilgan_0='unavailable',aprilgan_4='unavailable'))
(out/'samples.json').write_text(json.dumps(samples,indent=2),encoding='utf-8')
print('Extracted six verified local examples without new inference.')
