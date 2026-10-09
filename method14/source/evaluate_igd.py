"""IGD multiscale image score and official residual-map arithmetic, all test images."""
import ast
import csv
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_curve
from common import ROOT,REPO,RAW,RUNTIME,OUT,DATA,CATEGORIES,setup,sha,ACCELERATED
from accelerate_igd import fast_model

scalar=setup().ms_ssim
from p32.ssim_module import twoin1Generator
from p256.mvtec_module import twoin1Generator256
spec=importlib.util.spec_from_file_location('igd_residual',RUNTIME/'multi_scale/pytorch_msssim_residual_map.py')
residual=importlib.util.module_from_spec(spec);spec.loader.exec_module(residual)
# Official localization entry point references unavailable external modules
# and incompatible loader signatures. Reuse its actual map assembly functions.
tree=ast.parse((RUNTIME/'multi_scale/inference_loc.py').read_text())
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['rgb2gray','heat_map_printer']]
scope=dict(torch=torch,F=F,numpy=np)
exec(compile(ast.Module(body=functions,type_ignores=[]),'official_igd_map_functions','exec'),scope)
transform=T.Compose([T.Resize(256),T.CenterCrop(256),T.ToTensor(),
    T.Normalize([.485,.456,.406],[.229,.224,.225])])
mask_transform=T.Compose([T.Resize(256),T.CenterCrop(256),T.ToTensor()])
RANGE=2.1179+2.6400
W32=[.0516,.3295,.3463,.2726];W256=[.0448,.2856,.3001,.2363,.1333]
KEYS=['image_auroc','image_f1_max','image_ap','pixel_auroc','pixel_f1_max','pixel_ap']
def write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def f1(labels,scores):
    p,r,_=precision_recall_curve(labels,scores)
    return float(np.divide(2*p*r,p+r,out=np.zeros_like(p),where=(p+r)!=0).max())
def feature_score(model,x,win,weights,local):
    z=model.encoder(x);recon=model(x)
    ss=1-scalar(x,recon,data_range=RANGE,size_average=False if local else True,win_size=win,weights=weights)
    l1=(x-recon).mean((1,2,3))/RANGE if local else F.l1_loss(x,recon)/RANGE
    reconstruction=.85*ss+.15*l1
    gaussian=1-torch.exp(-((z-model.c)**2).sum(1)/model.sigma)
    score=(.9*reconstruction+.1*gaussian).max() if local else (.1*reconstruction+.9*gaussian).max()
    _,maps=residual.ms_ssim(x,recon,data_range=RANGE,size_average=True,win_size=win,weights=weights)
    combined=sum((1-F.interpolate(m,size=x.shape[-1]))*w for m,w in zip(maps,weights))
    pixel=F.relu(.85*combined+.15*torch.abs(x-recon)/RANGE)
    return score,pixel
rows=[]
with torch.no_grad():
    for category in CATEGORIES:
        model32=twoin1Generator(64,latent_dimension=128).cuda()
        model256=twoin1Generator256(64,latent_dimension=128).cuda()
        if ACCELERATED:fast_model(model32);fast_model(model256)
        for scale,model in [(32,model32),(256,model256)]:
            ckpt=torch.load(RAW/'checkpoints'/f'p{scale}'/category/'final.pt',map_location='cuda',weights_only=True)
            model.load_state_dict(ckpt['model']);model.c=ckpt['c'];model.sigma=ckpt['sigma'];model.eval()
        images=sorted((DATA/category/'test').glob('*/*.png'))
        scores=[];maps=[];masks=[];labels=[];image_rows=[];native_anomaly_aucs=[]
        for path in images:
            image=transform(Image.open(path).convert('RGB'))[None].cuda()
            patches=image.unfold(2,32,16).unfold(3,32,16).permute(0,2,3,1,4,5).reshape(-1,3,32,32)
            s32,map32=feature_score(model32,patches,3,W32,True)
            s256,map256=feature_score(model256,image,11,W256,False)
            score=float((.5*s32+.5*s256).cpu())
            localmap=scope['heat_map_printer'](map32,image,None,plot=False)
            globalmap=scope['rgb2gray'](map256[0].cpu().numpy().transpose(1,2,0).astype(np.float32))
            heatmap=.5*localmap+.5*globalmap
            label=int(path.parent.name!='good')
            if label:
                mp=DATA/category/'ground_truth'/path.parent.name/(path.stem+'_mask.png')
                mask=mask_transform(Image.open(mp).convert('L'))[0].numpy().astype(np.uint8)
            else:mask=np.zeros((256,256),dtype=np.uint8)
            assert heatmap.shape==mask.shape==(256,256) and np.isfinite(heatmap).all() and np.isfinite(score)
            scores.append(score);labels.append(label);maps.append(heatmap);masks.append(mask)
            image_rows.append(dict(image_path=str(path),label=label,score=score,sha256=sha(path)))
            if label and len(np.unique(mask))==2:native_anomaly_aucs.append(roc_auc_score(mask.ravel(),heatmap.ravel()))
        maps=np.stack(maps);masks=np.stack(masks);scores=np.asarray(scores);labels=np.asarray(labels)
        folder=RAW/category;folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/'raw_predictions.npz',image_labels=labels,image_scores=scores,anomaly_maps=maps,masks=masks)
        evidence=OUT/category;evidence.mkdir(exist_ok=True)
        write(evidence/'image_scores.csv',image_rows)
        row=dict(category=category,n_images=len(images),image_auroc=float(roc_auc_score(labels,scores)),
            image_f1_max=f1(labels,scores),image_ap=float(average_precision_score(labels,scores)),
            pixel_auroc=float(roc_auc_score(masks.ravel(),maps.ravel())),pixel_f1_max=f1(masks.ravel(),maps.ravel()),
            pixel_ap=float(average_precision_score(masks.ravel(),maps.ravel())),
            official_style_anomaly_image_pixel_auroc=float(np.mean(native_anomaly_aucs)))
        rows.append(row);write(OUT/'category_metrics.csv',rows)
        (evidence/'raw_manifest.json').write_text(json.dumps(dict(path=str(folder/'raw_predictions.npz'),
            sha256=sha(folder/'raw_predictions.npz')),indent=2))
        del model32,model256,maps,masks
        torch.cuda.empty_cache()
assert len(rows)==15 and sum(r['n_images'] for r in rows)==1725
metric_keys=KEYS+['official_style_anomaly_image_pixel_auroc']
write(OUT/'mean_metrics.csv',[dict(category='macro_mean',n_images=1725,
    **{k:float(np.mean([r[k] for r in rows])) for k in metric_keys})])
(OUT/'evaluation_provenance.json').write_text(json.dumps(dict(status='completed',image_score_source='multi_scale/inference_det.py',
    pixel_map_source='multi_scale/inference_loc.py + pytorch_msssim_residual_map.py',
    official_image_source_sha256=sha(REPO/'multi_scale/inference_det.py'),
    official_pixel_source_sha256=sha(REPO/'multi_scale/inference_loc.py'),
    official_entrypoint_executed=False,reason='official localization loader signature and external checkpoint imports are broken',
    checkpoint_models='same models trained by official job modules; unavailable external models are not substituted',
    pixel_auroc_definition='pooled all test-image pixels per category',
    alternative_metric='mean per anomalous-image Pixel AUROC, matching official entrypoint aggregation style',
    mask_resize='bilinear Resize256+CenterCrop256, ToTensor then uint8 truncation',
    aupro_required_by_musc_table2=False,aupro_measured=False),indent=2))
print('IGD EVALUATION COMPLETE',flush=True)
