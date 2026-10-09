"""Run official ACR feature extraction, 15 leave-one-category-out fits and capture maps."""
import ast
import csv
import gc
import importlib.metadata
import json
import os
import random
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import torch
import yaml
from sklearn.metrics import auc,average_precision_score,precision_recall_curve,roc_auc_score
from skimage import measure
from prepare_acr import ROOT,RAW,OUT,RUNTIME,COMMIT,KEYS,prepare,sha

DATA='/home/test/data/mvtec'
def write(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def f1(y,s):
    p,r,_=precision_recall_curve(y,s)
    return float(np.max(np.divide(2*p*r,p+r,out=np.zeros_like(p),where=(p+r)>0)))
def metrics(labels,scores,masks,maps):
    return dict(image_auroc=float(roc_auc_score(labels,scores)),image_f1_max=f1(labels,scores),
        image_ap=float(average_precision_score(labels,scores)),pixel_auroc=float(roc_auc_score(masks.ravel(),maps.ravel())),
        pixel_f1_max=f1(masks.ravel(),maps.ravel()),pixel_ap=float(average_precision_score(masks.ravel(),maps.ravel())))

prepare()
assert subprocess.check_output(['git','-C','/home/test/ACR','rev-parse','HEAD'],text=True).strip()==COMMIT
assert not subprocess.check_output(['git','-C','/home/test/ACR','diff','--name-only'],text=True).strip()
if (OUT/'verification.json').exists() and json.loads((OUT/'verification.json').read_text()).get('status')=='passed':
    raise SystemExit('Run already completed; existing verified evidence preserved.')
torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
torch.backends.cudnn.benchmark=True
sys.path.insert(0,str(RUNTIME));sys.path.insert(0,str(RUNTIME/'data_loader'))
os.chdir(RUNTIME)
config=yaml.safe_load((RUNTIME/'config_files/config_mvtec.yml').read_text())
env=dict(status='extracting_features',official_repo='https://github.com/aodongli/zero-shot-ad-via-batch-norm',commit=COMMIT,
    gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,python=sys.version,
    packages={k:importlib.metadata.version(k) for k in ['torch','torchvision','numpy','scipy','scikit-learn','scikit-image','PyYAML']},
    config=config,seed=42,seed_definition='local reproducibility choice; original main does not set a training seed',
    feature_seed=1024,precision='FP32; TF32 enabled; no AMP',tf32=True,cudnn_benchmark=True,
    protocol='15 leave-one-category-out models: only other 14 normal training categories used for meta-training',
    target_test_labels_used_for_training=False,iterations=50,tasks_per_update=32,normal_per_task=30,noisy_per_task=30,
    synthetic_noise_std=.1,query_anomaly_ratio=.5,feature='torchvision WideResNet50-2 ImageNet V1 layer3, 1024x14x14, no channel subsampling',
    resize=256,crop=224,image_interpolation='Pillow LANCZOS',mask_interpolation='NEAREST',
    normalization='ImageNet mean/std',inference_bn='train mode; target images and equal number of Gaussian-corrupted copies per spatial position',
    score_map='bilinear interpolate to224, Gaussian sigma4, per-category min-max',image_score='max normalized map',
    checkpoint_selection='final iteration50; no test-based selection',native_test_calls_per_category=50,
    raw_output=str(RAW),wrapper_sha256=sha(__file__),official_checkout_edits=False,
    compatibility='LANCZOS enum; local trusted torch.load; mmap features; omit unused cross-category caches/layer1-2 output retention; fix undefined optional logger; capture original score maps',
    aupro='APRIL-GAN cal_pro_score, 200 thresholds, FPR<0.3; supplementary metric')
def state():(OUT/'environment.json').write_text(json.dumps(env,indent=2))
state()
try:
    # A prior interruption can leave a partially written feature file.
    for cached in (RAW/'features/wide_resnet50_2').glob('*.pt'):
        try:
            value=torch.load(cached,map_location='cpu',weights_only=False,mmap=True)
            del value
        except (RuntimeError,EOFError,OSError):
            cached.rename(cached.with_suffix('.interrupted'))
    import extract_embedding
    argv=sys.argv[:]
    sys.argv=['extract_embedding.py','--data_path',DATA,'--save_path',str(RAW/'features')]
    extract_embedding.main();sys.argv=argv
    # Extraction imports top-level mvtec from its directory. Training imports
    # data_loader as a package; remove the extraction-only path first.
    sys.path.remove(str(RUNTIME/'data_loader'))
    from torchvision.models import Wide_ResNet50_2_Weights
    weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1
    weight_file=torch.hub.get_dir()+'/checkpoints/'+weights.url.rsplit('/',1)[-1]
    env['backbone_checkpoint_sha256']=sha(weight_file)
    feature_files=sorted((RAW/'features/wide_resnet50_2').glob('*.pt'))
    assert len(feature_files)==60
    (OUT/'feature_manifest.json').write_text(json.dumps([dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in feature_files],indent=2))
    from data_loader.mvtec import MVTecDataset,CLASS_NAMES
    from data_loader.data_loader import dataloader
    from config.parser import model_config_reader
    from main import run_dataset
    from trainers.zeroshot_trainer import ZeroShotMetaTrainer
    pro_path=__import__('pathlib').Path('/home/test/VAND-APRIL-GAN/test.py')
    pro_node=next(n for n in ast.parse(pro_path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='cal_pro_score')
    scope=dict(np=np,measure=measure,auc=auc)
    exec(compile(ast.Module(body=[pro_node],type_ignores=[]),str(pro_path),'exec'),scope)
    env['aupro_source_sha256']=sha(pro_path)
    config_runtime=model_config_reader('config_mvtec.yml')
    config_runtime['result_folder']=str(RAW/'models')
    native_train=ZeroShotMetaTrainer._train
    native_fit=ZeroShotMetaTrainer.train
    holder={}
    def checked_train(self,*a,**kw):
        loss=native_train(self,*a,**kw)
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None)
        holder.setdefault('history',[]).append(dict(iteration=int(a[0]),loss=float(loss.detach()),finite_gradients=True))
        write(OUT/env['current_category']/'training_history.csv',holder['history'])
        if not holder.get('first_step'):
            holder['first_step']=True
            (OUT/env['current_category']/'first_training_step.json').write_text(json.dumps(dict(device=str(self.device),
                loss=float(loss.detach()),finite_gradients=True,training_executed=True),indent=2))
        return loss
    def captured_fit(self,*a,**kw):
        result=native_fit(self,*a,**kw)
        holder['trainer']=self
        return result
    ZeroShotMetaTrainer._train=checked_train;ZeroShotMetaTrainer.train=captured_fit
    rows=[]
    for category in CLASS_NAMES:
        catout=OUT/category;catout.mkdir(exist_ok=True)
        if (catout/'metrics.json').exists():
            rows.append(json.loads((catout/'metrics.json').read_text()));continue
        env.update(status='training',current_category=category);state()
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        holder.clear()
        args=SimpleNamespace(dataset_name='mvtec',class_name=category,contamination_ratio=.5,ckpt_path='unspecified')
        dataset=dataloader(category,args,config_runtime)
        run_dataset(dataset,args,config_runtime)
        trainer=holder['trainer']
        maps=np.asarray(trainer.last_scores,dtype=np.float32)
        masks=np.asarray(trainer.last_masks).squeeze(1).astype(np.uint8)
        labels=np.asarray(trainer.last_labels,dtype=np.int64)
        scores=maps.reshape(len(labels),-1).max(1)
        assert maps.shape==masks.shape==(len(labels),224,224) and np.isfinite(maps).all()
        paths=MVTecDataset(DATA,class_name=category,is_train=False).x
        assert len(paths)==len(labels)
        raw=RAW/f'{category}_predictions.npz'
        np.savez_compressed(raw,image_labels=labels,image_scores=scores,anomaly_maps=maps,masks=masks)
        (catout/'raw_manifest.json').write_text(json.dumps(dict(path=str(raw),sha256=sha(raw),shape=list(maps.shape)),indent=2))
        checkpoint=RAW/'models/mvtec'/category/'final.pt'
        torch.save(dict(model=trainer.model.state_dict(),iterations=50,category=category,seed=42,config=config),checkpoint)
        (catout/'checkpoint.json').write_text(json.dumps(dict(path=str(checkpoint),sha256=sha(checkpoint),iterations=50),indent=2))
        write(catout/'image_scores.csv',[dict(category=category,index=i,image_path=p,sha256=sha(p),label=int(y),score=float(s)) for i,(p,y,s) in enumerate(zip(paths,labels,scores))])
        env.update(status='computing_metrics');state()
        row=dict(category=category,n_images=len(labels),**metrics(labels,scores,masks,maps),aupro=float(scope['cal_pro_score'](masks,maps)))
        assert abs(row['image_auroc']-trainer.last_native_auroc[0])<1e-12
        assert abs(row['pixel_auroc']-trainer.last_native_auroc[1])<1e-12
        (catout/'native_auroc.json').write_text(json.dumps(dict(image_auroc=trainer.last_native_auroc[0],pixel_auroc=trainer.last_native_auroc[1]),indent=2))
        # sklearn AP can exceed1 by machine-rounding (e.g.1.0000000000000002).
        # Preserve the raw value and allow only negligible numeric error.
        assert all(np.isfinite(row[k]) and -1e-12<=row[k]<=1+1e-12 for k in KEYS)
        # Native final detection uses the same pooled AUROC formulas; raw values are preserved.
        (catout/'metrics.json').write_text(json.dumps(row,indent=2));rows.append(row)
        write(OUT/'category_metrics.csv',rows)
        print('CATEGORY COMPLETE',category,flush=True)
        del dataset,trainer,maps,masks,labels,scores;holder.clear();gc.collect();torch.cuda.empty_cache()
    assert len(rows)==15 and sum(r['n_images'] for r in rows)==1725
    mean=dict(category='Mean',n_images=1725,**{k:float(np.mean([r[k] for r in rows])) for k in KEYS})
    write(OUT/'mean_metrics.csv',[mean]);env.update(status='trained_and_evaluated');state()
    subprocess.run([sys.executable,str(ROOT/'finish_acr.py')],check=True)
except Exception as error:
    env.update(status='failed',error=repr(error));state();raise
