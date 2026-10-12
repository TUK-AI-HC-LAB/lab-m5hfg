"""Recreate missing RsCIN class tokens using pinned local official MuSc."""
import argparse,csv,hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--category',required=True);args=p.parse_args()
repo=Path('/home/test/MuSc');sys.path.insert(0,str(repo));os.chdir(repo)
from models.musc import MuSc
raw=Path('/home/test/patchcore_results/musc_comparison_20261010/rscin_features')/args.category
out=ROOT/'results/musc_comparison_20261010/rscin_features'/args.category;raw.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
cfg=dict(device='0',datasets=dict(data_path='/home/test/data/mvtec',dataset_name='mvtec_ad',class_name=args.category,img_resize=518,divide_num=1),
    models=dict(backbone_name='ViT-L-14-336',pretrained='openai',batch_size=1,feature_layers=[5,11,17,23],r_list=[1,3,5]),
    testing=dict(vis=False,vis_type='single_norm',save_excel=False,output_dir=str(raw)))
torch.manual_seed(42);torch.cuda.manual_seed_all(42);np.random.seed(42)
torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
model=MuSc(cfg,seed=42);rows=[];vectors=[]
for item in torch.utils.data.DataLoader(model.load_datasets(args.category),batch_size=1,shuffle=False,num_workers=0):
    with torch.no_grad(),torch.autocast('cuda'):
        feats,_=model.clip_model.encode_image(item['image'].float().to(model.device),model.features_list);feats/=feats.norm(dim=-1,keepdim=True)
    path=item['image_path'][0];rows.append(dict(image_path=path,label=int(item['is_anomaly'][0]),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()));vectors.append(feats.cpu().numpy())
np.save(raw/'features.npy',np.concatenate(vectors))
with (out/'images.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(out/'environment.json').write_text(json.dumps(dict(status='completed',n_images=len(rows),config=cfg,commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),feature_path=str(raw/'features.npy'),sha256=hashlib.sha256((raw/'features.npy').read_bytes()).hexdigest()),indent=2))
print('RECREATED OFFICIAL RsCIN FEATURES',args.category,len(rows),flush=True)
