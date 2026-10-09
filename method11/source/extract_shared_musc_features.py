"""Extract local official MuSc CLIP features for uniform Table 11 re-scoring."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['mvtec','visa'],required=True);opt=p.parse_args()
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'/f'rscin_shared_features_{opt.dataset}_20261006'
if (OUT/'environment.json').exists() and json.loads((OUT/'environment.json').read_text())['status']=='completed':
    raise SystemExit(0)
OUT.mkdir(exist_ok=True)
repo=Path('/home/test/MuSc');os.chdir(repo);sys.path.insert(0,str(repo));sys.dont_write_bytecode=True
from models.musc import MuSc
ds='mvtec_ad' if opt.dataset=='mvtec' else 'visa'
data='/home/test/data/mvtec' if opt.dataset=='mvtec' else '/home/test/data/VisA_20220922'
cfg=dict(device='0',datasets=dict(data_path=data,dataset_name=ds,class_name='all',img_resize=518,divide_num=1),
         models=dict(backbone_name='ViT-L-14-336',pretrained='openai',batch_size=1,feature_layers=[5,11,17,23],r_list=[1,3,5]),
         testing=dict(vis=False,vis_type='single_norm',save_excel=False,output_dir='/home/test/aprilgan_results/rscin_shared_musc_features'))
torch.manual_seed(42);torch.cuda.manual_seed_all(42);np.random.seed(42)
torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
env=dict(status='running',config=cfg,official_musc_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
         feature_source='official MuSc load_backbone/load_datasets; normalized projected class token; CUDA AMP',
         paper_provided_features_used=False,batch_size=1)
(OUT/'environment.json').write_text(json.dumps(env,indent=2))
model=MuSc(cfg,seed=42);total=0
for category in model.categories:
    dataset=model.load_datasets(category)
    loader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0)
    vectors=[];rows=[]
    for item in loader:
        with torch.no_grad(),torch.cuda.amp.autocast():
            feats,_=model.clip_model.encode_image(item['image'].float().to(model.device),model.features_list)
            feats/=feats.norm(dim=-1,keepdim=True)
        path=item['image_path'][0]
        rows.append(dict(image_path=path,label=int(item['is_anomaly'][0]),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()))
        vectors.append(feats.cpu().numpy())
    raw=Path('/home/test/aprilgan_results/rscin_shared_musc_features')/opt.dataset/category
    raw.mkdir(parents=True,exist_ok=True);np.save(raw/'features.npy',np.concatenate(vectors))
    folder=OUT/category;folder.mkdir(exist_ok=True)
    with (folder/'images.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    total+=len(rows);print('SHARED MUSC FEATURES',opt.dataset,category,len(rows),flush=True)
assert total==(1725 if opt.dataset=='mvtec' else 2162)
env.update(status='completed',n_images=total)
(OUT/'environment.json').write_text(json.dumps(env,indent=2))
