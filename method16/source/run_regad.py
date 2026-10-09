"""Official public RegAD evaluation: 4/2/8-shot, 15 categories, 10 support rounds."""
import ast
import csv
import gc
import importlib.metadata
import importlib.util
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from sklearn.metrics import auc,average_precision_score,precision_recall_curve,roc_auc_score
from skimage import measure
from prepare_regad import ROOT,REPO,RAW,OUT,RUNTIME,ASSETS,COMMIT,KEYS,prepare,sha
def write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def read(path):
    with path.open() as f:return list(csv.DictReader(f))
def classification(y,s):
    p,r,_=precision_recall_curve(y,s)
    return dict(image_auroc=float(roc_auc_score(y,s)),image_f1_max=float(np.divide(2*p*r,p+r,out=np.zeros_like(p),where=p+r>0).max()),image_ap=float(average_precision_score(y,s)))
def all_metrics(y,s,masks,maps):
    px=classification(masks.ravel(),maps.ravel())
    return dict(**classification(y,s),pixel_auroc=px['image_auroc'],pixel_f1_max=px['image_f1_max'],pixel_ap=px['image_ap'],aupro=float(pro_scope['cal_pro_score'](masks,maps)))

prepare()
support_overrides=json.loads((OUT/'support_overrides.json').read_text()) if (OUT/'support_overrides.json').exists() else []
support_lookup={(r['category'],r['shot']):r for r in support_overrides}
assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==COMMIT
assert not subprocess.check_output(['git','-C',str(REPO),'diff','--name-only'],text=True).strip()
if (OUT/'verification.json').exists() and json.loads((OUT/'verification.json').read_text()).get('status')=='passed':raise SystemExit('Existing verified run preserved')
sys.path.insert(0,str(RUNTIME));os.chdir(RUNTIME)
import test as native
from models.stn import stn_net
from models.siamese import Encoder,Predictor
from datasets.mvtec import FSAD_Dataset_test,CLASS_NAMES
torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=True
env=dict(status='starting',official_repo='https://github.com/MediaBrain-SJTU/RegAD',commit=COMMIT,
    gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,python=sys.version,
    packages={p:importlib.metadata.version(p) for p in ['torch','torchvision','numpy','kornia','scipy','scikit-learn','scikit-image']},
    training_executed=False,checkpoint_source='official Google Drive save_checkpoints.tar',support_source='official Google Drive support_set.tar',
    checkpoint_archive_sha256=sha(ASSETS/'save_checkpoints.tar'),support_archive_sha256=sha(ASSETS/'support_set.tar'),
    shots=[4,2,8],rounds_per_category=10,seed=668,seed_definition='official default668; additionally seed CPU/random/numpy for reproducibility',
    image_size=224,image_resize='LANCZOS',normalization='ToTensor only; original ImageNet normalization commented out',mask_resize='NEAREST; binarize>0.5',
    stn_mode='rotation_scale',precision='FP32; convolution TF32 enabled; matmul TF32 disabled',inference_context='torch.no_grad',
    covariance='original per-position torch.cov +0.01I',distance='batched inverse covariance Mahalanobis, native formula; first real-feature subset verified',
    native_source_sha256=sha(REPO/'test.py'),wrapper_sha256=sha(__file__),original_checkout_edits=False,
    pending_conditions=['MVTec32-shot requires matching trained model/support','BTAD4-shot requires explicit transfer/training protocol'],
    raw_output=str(RAW))
env['support_overrides']=support_overrides
def state():(OUT/'environment.json').write_text(json.dumps(env,indent=2))
state()
try:
    pro_path=Path('/home/test/VAND-APRIL-GAN/test.py')
    node=next(n for n in ast.parse(pro_path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='cal_pro_score')
    pro_scope=dict(np=np,measure=measure,auc=auc)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(pro_path),'exec'),pro_scope)
    env['aupro_source_sha256']=sha(pro_path)
    rscin_path=Path('/home/test/MuSc/models/RsCIN_features/RsCIN.py')
    spec=importlib.util.spec_from_file_location('official_rscin',rscin_path)
    rscin=importlib.util.module_from_spec(spec);spec.loader.exec_module(rscin)
    env['rscin_source_sha256']=sha(rscin_path)
    shared=ROOT.parents[1]/'method11/source/results/rscin_shared_features_mvtec_20261006'
    all_rows=[];rscin_rows=[]
    for shot in [4,2,8]:
        for category in CLASS_NAMES:
            folder=OUT/f'{shot}shot'/category;folder.mkdir(parents=True,exist_ok=True)
            rawfolder=RAW/f'{shot}shot'/category;rawfolder.mkdir(parents=True,exist_ok=True)
            env.update(status='evaluating',current_shot=shot,current_category=category);state()
            random.seed(668);np.random.seed(668);torch.manual_seed(668);torch.cuda.manual_seed_all(668)
            args=SimpleNamespace(obj=category,shot=shot,stn_mode='rotation_scale',input_channel=3,img_size=224)
            STN=stn_net(args,pretrained=False).cuda();ENC=Encoder().cuda();PRED=Predictor().cuda()
            checkpoint=ASSETS/'save_checkpoints'/str(shot)/category/f'{category}_{shot}_rotation_scale_model.pt'
            saved=torch.load(checkpoint,map_location='cuda',weights_only=True)
            for key,model in zip(['STN','ENC','PRED'],[STN,ENC,PRED]):model.load_state_dict(saved[key]);model.eval()
            support_path=ASSETS/'support_set'/category/f'{shot}_10.pt'
            if (category,shot) in support_lookup:support_path=Path(support_lookup[(category,shot)]['path'])
            support=torch.load(support_path,map_location='cpu',weights_only=True)
            assert len(support)==10 and support[0].shape==(shot,3,224,224)
            dataset=FSAD_Dataset_test('/home/test/data/mvtec',class_name=category,is_train=False,resize=224,shot=shot)
            loader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False,num_workers=4,pin_memory=True,persistent_workers=True)
            paths=dataset.query_dir
            references=read(shared/category/'images.csv');lookup={r['image_path']:i for i,r in enumerate(references)}
            indices=[lookup[p] for p in paths]
            cls_path=Path('/home/test/aprilgan_results/rscin_shared_musc_features/mvtec')/category/'features.npy'
            features=np.load(cls_path)[indices]
            (folder/'inputs.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                support=str(support_path),support_sha256=sha(support_path),rscin_feature_path=str(cls_path),rscin_feature_sha256=sha(cls_path)),indent=2))
            category_rows=[]
            for round_id in range(10):
                marker=folder/f'round{round_id}_metrics.json'
                if marker.exists():
                    row=json.loads(marker.read_text());all_rows.append(row);category_rows.append(row)
                    rscin_rows.extend(json.loads((folder/f'round{round_id}_rscin.json').read_text()));continue
                env.update(current_round=round_id,status='evaluating');state()
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
                with torch.no_grad():maps,_,labels,masks=native.test(args,[STN,ENC,PRED],round_id,support,loader,num_workers=4,pin_memory=True)
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                maps=np.asarray(maps);maps=(maps-maps.min())/(maps.max()-maps.min())
                labels=np.asarray(labels,dtype=np.int64);masks=(np.asarray(masks)>.5).astype(np.uint8)
                # Official test masks retain a singleton channel; flattening
                # metrics are unchanged when that axis is removed.
                if masks.ndim==4 and masks.shape[1]==1:masks=masks[:,0]
                assert maps.shape==masks.shape==(len(paths),224,224) and np.isfinite(maps).all()
                scores=maps.reshape(len(paths),-1).max(1)
                for p,y in zip(paths,labels):
                    reference=references[lookup[p]]
                    assert int(reference['label'])==int(y) and sha(p)==reference['sha256']
                rawfile=rawfolder/f'round{round_id}.npz'
                np.savez_compressed(rawfile,image_labels=labels,image_scores=scores,anomaly_maps=maps,masks=masks)
                write(folder/f'round{round_id}_images.csv',[dict(image_path=p,sha256=sha(p),label=int(y),score=float(s)) for p,y,s in zip(paths,labels,scores)])
                (folder/f'round{round_id}_raw.json').write_text(json.dumps(dict(path=str(rawfile),sha256=sha(rawfile)),indent=2))
                env['status']='computing_metrics';state()
                row=dict(dataset='mvtec',shot=shot,category=category,round=round_id,n_images=len(paths),
                    **all_metrics(labels,scores,masks,maps),evaluation_seconds=elapsed,gpu_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)
                assert all(np.isfinite(row[k]) and -1e-12<=row[k]<=1+1e-12 for k in KEYS)
                after=rscin.Mobile_RsCIN(scores.copy(),dataset_name='mvtec_ad',cls_tokens=features)
                round_rscin=[dict(shot=shot,category=category,round=round_id,rscin=flag,n_images=len(paths),**classification(labels,values)) for flag,values in [('w/o',scores),('w',after)]]
                (folder/f'round{round_id}_rscin.json').write_text(json.dumps(round_rscin,indent=2))
                write(folder/f'round{round_id}_rscin_scores.csv',[dict(image_path=p,label=int(y),score_before=float(s),score_after=float(a)) for p,y,s,a in zip(paths,labels,scores,after)])
                marker.write_text(json.dumps(row,indent=2));all_rows.append(row);category_rows.append(row);rscin_rows.extend(round_rscin)
                write(OUT/'round_metrics.csv',all_rows);write(OUT/'rscin_round_metrics.csv',rscin_rows)
                print('ROUND COMPLETE',shot,category,round_id,flush=True)
                del maps,masks,scores,after;gc.collect();torch.cuda.empty_cache()
            del STN,ENC,PRED,saved,support,dataset,loader,features;gc.collect();torch.cuda.empty_cache()
    env['status']='evaluated';state()
    subprocess.run([sys.executable,str(ROOT/'finish_regad.py')],check=True)
except Exception as error:
    env.update(status='failed',error=repr(error));state();raise
