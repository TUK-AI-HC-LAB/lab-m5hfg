"""Pinned official NSA logistic training and evaluation; no ETA polling."""
import csv
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault('MPLBACKEND','Agg')
import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import precision_recall_curve

ROOT=Path(__file__).resolve().parent
REPO=Path('/home/test/NSA')
RAW=Path('/home/test/nsa_results/logistic_seed923874273_20261007')
OUT=ROOT/'results/logistic_seed923874273_20261007'
DATA=Path('/home/test/nsa_data')
SEED=923874273
COMMIT='919591685307ce030fe27cb77687509dc277189c'
KEYS=['image_auroc','image_f1_max','image_ap','pixel_auroc','pixel_f1_max','pixel_ap','aupro']
sys.dont_write_bytecode=True
sys.path.insert(0,str(REPO))
# Compatibility aliases preserve old semantics without editing official files.
Image.ANTIALIAS=Image.Resampling.LANCZOS
import train_mvtec as official
import experiments.training_utils as training
import experiments.plotting_utils as plotting
import experiments.mvtec_tasks as evaluation
grid=plotting.make_grid
def compatible_grid(*a,**kw):
    if 'range' in kw: kw['value_range']=kw.pop('range')
    return grid(*a,**kw)
plotting.make_grid=compatible_grid
training.tqdm=lambda iterable,*a,**kw:iterable
evaluation.tqdm=lambda iterable,*a,**kw:iterable
native_loader=official.DataLoader
def bounded_loader(*a,**kw):
    kw['num_workers']=8
    return native_loader(*a,**kw)
official.DataLoader=bounded_loader
torch.set_num_threads(8)
cv2.setNumThreads(1)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
OUT.mkdir(parents=True,exist_ok=True);RAW.mkdir(parents=True,exist_ok=True)
assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==COMMIT
assert not subprocess.check_output(['git','-C',str(REPO),'diff','--name-only'],text=True).strip()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def f1(labels,scores):
    p,r,_=precision_recall_curve(labels,scores)
    return float(np.divide(2*p*r,p+r,out=np.zeros_like(p),where=p+r!=0).max())
env=dict(status='starting',official_repo='https://github.com/hmsch/natural-synthetic-anomalies',commit=COMMIT,
    gpu=torch.cuda.get_device_name(0),python=sys.version,cuda=torch.version.cuda,
    packages={p:importlib.metadata.version(p) for p in ['torch','torchvision','numpy','Pillow','scikit-learn','scikit-image','opencv-python']},
    wrapper_sha256=sha(__file__),source_sha256={p:sha(REPO/p) for p in ['train_mvtec.py','self_sup_data/mvtec.py',
    'self_sup_data/self_sup_tasks.py','model/resnet.py','experiments/training_utils.py','experiments/mvtec_tasks.py']},
    seed=SEED,batch_size=64,num_workers=8,precision='FP32; no AMP',tf32=False,
    epochs=official.EPOCHS,training_executed=True,raw_output=str(RAW),
    setting='Shift-Intensity-923874273 objects; Shift-Intensity-M-923874273 textures',
    optimizer='Adam lr=0.001; CosineAnnealingLR eta_min=1e-6',
    architecture='official resnet18_enc_dec; pool=True, preact=False, sigmoid',
    normalization='official ImageNet RGB mean/std',
    test_size=256,object_test='CenterCrop224 then Pad16; image score mean before pad',
    texture_test='full256; image score mean',
    compatibility=['Pillow ANTIALIAS alias LANCZOS','torchvision make_grid range renamed value_range (plots only)',
    'DataLoader workers limited to 8, OpenCV threads=1; official algorithm unchanged','tqdm display disabled'],
    source_edits=False,eta_calculation=False)
def state(): (OUT/'environment.json').write_text(json.dumps(env,indent=2))
state()
rows=[]
training_path=Path(training.__file__).resolve()
lines=training_path.read_text().splitlines()
train_line=next(i+1 for i,l in enumerate(lines) if l.strip()=='model.train()')
step_line=next(i+1 for i,l in enumerate(lines) if l.strip()=='optimizer.step()')
current={};logged=set();first_step=False
def trace(frame,event,arg):
    global first_step
    if frame.f_code.co_filename!=str(training_path) or frame.f_code.co_name!='train_and_save_model':return None
    if event=='line':
        local=frame.f_locals
        if frame.f_lineno==step_line and not first_step:
            loss=float(local['loss'].detach().cpu());assert np.isfinite(loss)
            (OUT/'first_batch.json').write_text(json.dumps(dict(category=current['category'],
                loss=loss,batch_size=int(local['data'].shape[0]),device=str(local['data'].device),
                input_shape=list(local['data'].shape),backward_completed=True),indent=2))
            first_step=True
        if frame.f_lineno==train_line:
            epoch=local['epoch']
            if epoch>0 and epoch-1 not in logged:
                with (current['folder']/'training_loss.csv').open('a') as f:
                    f.write(f"{epoch-1},{float(local['train_loss'][epoch-1])}\n")
                logged.add(epoch-1)
                if epoch%80==0:
                    torch.save(dict(epoch=epoch-1,model_state_dict=local['model'].state_dict(),
                        optimizer_state_dict=local['optimizer'].state_dict(),scheduler_state_dict=local['scheduler'].state_dict(),
                        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state()),
                        RAW/current['category']/'recovery.pt')
    return trace

try:
    for category in official.CLASS_NAMES:
        folder=OUT/category;folder.mkdir(exist_ok=True)
        raw=RAW/category;raw.mkdir(exist_ok=True)
        name='Shift-Intensity-M-923874273' if category in official.TEXTURES else 'Shift-Intensity-923874273'
        setting=official.SETTINGS[name]
        checkpoint=RAW/'weights'/setting['out_dir']/category/('final_'+category+'_'+setting['fname'])
        current=dict(category=category,folder=folder);logged=set()
        env.update(status='training',current_category=category,current_setting=name);state()
        # Category checkpoint plus completion marker is reusable; partial runs
        # remain preserved. Recovery checkpoints require explicit resume logic.
        if not (folder/'training_complete.json').exists():
            (folder/'training_loss.csv').write_text('epoch,loss\n')
            sys.settrace(trace)
            official.train(category,str(DATA),str(RAW/'weights'),setting,'cuda',True,False,False,
                           True,False,False)
            sys.settrace(None)
            assert checkpoint.is_file()
            (folder/'training_complete.json').write_text(json.dumps(dict(status='completed',epochs=official.EPOCHS[category],
                checkpoint=str(checkpoint),sha256=sha(checkpoint),seed=SEED,setting=name),indent=2))
        env.update(status='evaluating');state()
        model=official.resnet18_enc_dec(num_classes=1,pool=True,preact=False,final_activation='sigmoid').cuda()
        model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True))
        dataset=official.SelfSupMVTecDataset(root_path=str(DATA),class_name=category,is_train=False,low_res=256,download=False)
        paths=sorted(Path('/home/test/data/mvtec',category,'test').glob('*/*.png'))
        assert len(paths)==len(dataset)
        original_pro=evaluation.pro_score
        def capture_pro(masks,preds,*a,**kw):
            local=sys._getframe(1).f_locals
            labels=local['sample_labels'].copy();scores=local['sample_preds'].copy()
            assert np.isfinite(preds).all() and np.isfinite(scores).all()
            np.savez_compressed(raw/'raw_predictions.npz',image_labels=labels,image_scores=scores,
                anomaly_maps=preds,masks=masks)
            images=[dict(image_path=str(p),label=int(y),score=float(s),sha256=sha(p)) for p,y,s in zip(paths,labels,scores)]
            assert [r['label'] for r in images]==[int(p.parent.name!='good') for p in paths]
            write(folder/'image_scores.csv',images)
            return original_pro(masks,preds,*a,**kw)
        evaluation.pro_score=capture_pro
        native=evaluation.test_real_anomalies(model,dataset,device='cuda',batch_size=16,show=False,
            full_size=category in official.OBJECTS)
        evaluation.pro_score=original_pro
        with np.load(raw/'raw_predictions.npz') as z:
            labels,scores,masks,maps=[z[k] for k in ['image_labels','image_scores','masks','anomaly_maps']]
        row=dict(category=category,n_images=len(paths),image_auroc=float(native[1]),image_f1_max=f1(labels,scores),
            image_ap=float(native[0]),pixel_auroc=float(native[3]),pixel_f1_max=f1(masks.ravel(),maps.ravel()),
            pixel_ap=float(native[2]),aupro=float(native[4]))
        assert all(np.isfinite(row[k]) for k in KEYS)
        rows.append(row);write(OUT/'category_metrics.csv',rows)
        (folder/'raw_manifest.json').write_text(json.dumps(dict(path=str(raw/'raw_predictions.npz'),sha256=sha(raw/'raw_predictions.npz')),indent=2))
        del model,dataset,labels,scores,masks,maps
        torch.cuda.empty_cache()
    assert len(rows)==15 and sum(r['n_images'] for r in rows)==1725
    write(OUT/'mean_metrics.csv',[dict(category='macro_mean',n_images=1725,
        **{k:float(np.mean([r[k] for r in rows])) for k in KEYS})])
    env.update(status='completed');state()
    subprocess.run([sys.executable,str(ROOT/'finish_nsa.py')],check=True)
except Exception as error:
    sys.settrace(None);env.update(status='failed',error=repr(error));state();raise
