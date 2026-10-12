"""SPADE image classification and RsCIN for MuSc Table11 (no pixel task)."""
import importlib.util,json,platform,random,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torchvision.models import wide_resnet50_2,Wide_ResNet50_2_Weights
from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_curve
from spade_common import XFORM,sha,write,read,REPO,COMMIT

ROOT=Path(__file__).resolve().parent
LAB=ROOT.parents[1]
DATA=Path('/home/test/data/mvtec')
OUT=ROOT/'results/mvtec_k50_seed42_20261010'
RAW=Path('/home/test/spade_results/mvtec_k50_seed42_20261010')
PATCH=LAB/'method1/source/results/musc_comparison_20261010/rscin_features'
FEATURES=Path('/home/test/patchcore_results/musc_comparison_20261010/rscin_features')
RSCIN=Path('/home/test/MuSc/models/RsCIN_features/RsCIN.py')
class Images(Dataset):
    def __init__(self,rows):self.rows=rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        with Image.open(self.rows[i]['image_path']) as im:return XFORM(im.convert('RGB'))

def manifest(meta,cat,phase):
    rows=[]
    for r in meta[phase][cat]:
        p=DATA/r['img_path']
        rows.append(dict(image_path=str(p),label=int(r['anomaly']),sha256=sha(p)))
    assert len({r['image_path'] for r in rows})==len(rows)
    return rows

@torch.inference_mode()
def extract(model,captured,rows):
    batches=[]
    loader=DataLoader(Images(rows),batch_size=32,num_workers=4,pin_memory=True,persistent_workers=True)
    for x in loader:
        captured.clear();model(x.cuda(non_blocking=True))
        assert len(captured)==1
        batches.append(captured[0].flatten(1).cpu().numpy())
    return np.concatenate(batches)

def metrics(labels,scores):
    p,r,_=precision_recall_curve(labels,scores)
    f=np.divide(2*p*r,p+r,out=np.zeros_like(p),where=p+r>0)
    return dict(image_auroc=float(roc_auc_score(labels,scores)),image_f1_max=float(f.max()),
                image_ap=float(average_precision_score(labels,scores)))

def run_category(model,captured,meta,module,cat):
    out=OUT/cat;raw=RAW/cat;out.mkdir(exist_ok=True);raw.mkdir(exist_ok=True)
    if (out/'verification.json').exists():
        proof=json.loads((out/'verification.json').read_text())
        if proof['status']=='passed':
            for artifact in proof['raw_artifacts']:assert sha(artifact['path'])==artifact['sha256']
            return read(out/'rscin_metrics.csv')
    train=manifest(meta,cat,'train');test=manifest(meta,cat,'test')
    assert all(r['label']==0 for r in train) and len(train)>=50
    assert set(r['image_path'] for r in train).isdisjoint(r['image_path'] for r in test)
    write(out/'train_images.csv',train);write(out/'test_images.csv',test)
    t=time.perf_counter()
    tr=extract(model,captured,train);te=extract(model,captured,test)
    assert tr.shape==(len(train),2048) and te.shape==(len(test),2048)
    np.save(raw/'train_avgpool.npy',tr);np.save(raw/'test_avgpool.npy',te)
    gallery=torch.from_numpy(tr).cuda();scores=[];neighbors=[]
    # Direct squared differences, same K50 image formula as previous BTAD run.
    for q in torch.from_numpy(te).cuda():
        d=((gallery-q[None,:])**2).sum(1)
        vals,idx=d.topk(50,largest=False)
        scores.append(float(vals.mean()));neighbors.append(idx.cpu().numpy())
    scores=np.array(scores,dtype=np.float32);neighbors=np.array(neighbors)
    # Every image score and neighbor set against independent FP64 CPU formula.
    gallery64=tr.astype(np.float64);maxerr=0.
    for i,q in enumerate(te.astype(np.float64)):
        delta=gallery64-q;d=np.einsum('ij,ij->i',delta,delta)
        selected=d[neighbors[i]];best=np.sort(d)[:50]
        assert len(np.unique(neighbors[i]))==50 and np.allclose(np.sort(selected),best,rtol=1e-5,atol=1e-4)
        assert np.isclose(scores[i],best.mean(),rtol=1e-5,atol=1e-4)
        maxerr=max(maxerr,abs(float(scores[i])-float(best.mean())))
    refs=read(PATCH/cat/'images.csv');lookup={r['image_path']:i for i,r in enumerate(refs)}
    indices=[]
    for r in test:
        i=lookup[r['image_path']]
        assert refs[i]['sha256']==r['sha256'] and int(refs[i]['label'])==r['label']
        indices.append(i)
    feature_file=FEATURES/cat/'features.npy'
    cls=np.load(feature_file)[indices]
    after=module.Mobile_RsCIN(scores.copy(),dataset_name='mvtec_ad',cls_tokens=cls)
    labels=np.array([r['label'] for r in test],dtype=np.uint8)
    assert np.isfinite(scores).all() and np.isfinite(after).all()
    pred=raw/'predictions.npz'
    np.savez_compressed(pred,labels=labels,image_scores=scores,rscin_scores=after,neighbor_indices=neighbors)
    rows=[dict(dataset='mvtec',category=cat,rscin=flag,n_train=len(train),n_images=len(test),
               **metrics(labels,s)) for flag,s in [('w/o',scores),('w',after)]]
    write(out/'rscin_metrics.csv',rows)
    write(out/'image_scores.csv',[dict(**r,score_before_rscin=float(s),score_after_rscin=float(a))
                                 for r,s,a in zip(test,scores,after)])
    saved=np.load(pred)
    assert np.array_equal(saved['labels'],labels)
    for row,key in zip(rows,['image_scores','rscin_scores']):
        assert max(abs(float(row[k])-v) for k,v in metrics(saved['labels'],saved[key]).items())<1e-12
    assert np.array_equal(module.Mobile_RsCIN(saved['image_scores'].copy(),dataset_name='mvtec_ad',cls_tokens=cls),saved['rscin_scores'])
    for r in train+test:assert sha(r['image_path'])==r['sha256']
    artifacts=[dict(path=str(p),sha256=sha(p)) for p in [pred,raw/'train_avgpool.npy',raw/'test_avgpool.npy']]
    proof=dict(status='passed',n_train=len(train),n_images=len(test),
               all_global_K50_scores_and_neighbors_FP64_checked=True,max_absolute_score_error=maxerr,
               metrics_recomputed_from_saved_predictions=True,rscin_recomputed_exactly=True,
               rscin_image_label_SHA_alignment_passed=True,feature_path=str(feature_file),feature_sha256=sha(feature_file),
               module_sha256=sha(RSCIN),windows=[1,2,3],all_input_SHA_rechecked=True,
               raw_artifacts=artifacts,seconds=time.perf_counter()-t,
               scope='Table11 image classification only; pixel maps not required or computed')
    (out/'verification.json').write_text(json.dumps(proof,indent=2))
    print('CATEGORY VERIFIED',cat,rows,flush=True)
    return rows

def main():
    OUT.mkdir(parents=True,exist_ok=True);RAW.mkdir(parents=True,exist_ok=True)
    if (OUT/'verification.json').exists() and json.loads((OUT/'verification.json').read_text())['status']=='passed':return
    assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==COMMIT
    assert not subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip()
    meta=json.loads((DATA/'meta.json').read_text());cats=sorted(meta['test']);assert len(cats)==15
    for cat in cats:assert (FEATURES/cat/'features.npy').exists() and (PATCH/cat/'images.csv').exists()
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');torch.backends.cudnn.benchmark=True
    weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1
    model=wide_resnet50_2(weights=weights).cuda().eval();captured=[]
    model.avgpool.register_forward_hook(lambda m,x,y:captured.append(y.detach()))
    weight_file=Path(torch.hub.get_dir())/'checkpoints'/Path(weights.url).name
    env=dict(status='running',dataset='mvtec',categories=cats,seed=42,K=50,training_epochs=0,
             backbone='WideResNet50x2 ImageNetV1 avgpool2048',input='Resize shorter side256 LANCZOS CenterCrop224 ImageNet normalize',
             image_score='mean K50 squared L2; no coreset, full normal train',precision='FP32 no TF32 no AMP',
             batch_size=32,workers=4,gpu=torch.cuda.get_device_name(),torch=torch.__version__,python=platform.python_version(),
             public_repo='https://github.com/byungjae89/SPADE-pytorch',commit=COMMIT,official_author_code=False,
             weights_path=str(weight_file),weights_sha256=sha(weight_file),wrapper_sha256=sha(__file__),
             rscin_commit=subprocess.check_output(['git','-C','/home/test/MuSc','rev-parse','HEAD'],text=True).strip(),
             rscin_windows=[1,2,3],paper_window_difference='paper text MVTec2/3, official default1/2/3 retained consistently with previous comparisons',
             scope='MuSc Table11 image classification before/after RsCIN; no pixel evaluation',raw_root=str(RAW),error=None)
    (OUT/'environment.json').write_text(json.dumps(env,indent=2))
    spec=importlib.util.spec_from_file_location('official_rscin',RSCIN);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=[]
    for cat in cats:
        env['category']=cat;(OUT/'environment.json').write_text(json.dumps(env,indent=2))
        rows+=run_category(model,captured,meta,module,cat);write(OUT/'rscin_category_metrics.csv',rows)
    assert len(rows)==30 and sum(int(r['n_images']) for r in rows if r['rscin']=='w')==1725
    means=[dict(dataset='mvtec',rscin=flag,n_categories=15,n_images=1725,
                **{k:float(np.mean([float(r[k]) for r in rows if r['rscin']==flag]))
                   for k in ['image_auroc','image_f1_max','image_ap']}) for flag in ['w/o','w']]
    write(OUT/'rscin_mean_metrics.csv',means)
    assert sha(__file__)==env['wrapper_sha256'] and sha(weight_file)==env['weights_sha256']
    (OUT/'verification.json').write_text(json.dumps(dict(status='passed',n_categories=15,n_images=1725,
        all_category_verifications_passed=all(json.loads((OUT/c/'verification.json').read_text())['status']=='passed' for c in cats),
        classification_only=True,all_global_scores_FP64_checked=True,all_RsCIN_alignment_and_metrics_checked=True),indent=2))
    env['status']='completed';(OUT/'environment.json').write_text(json.dumps(env,indent=2))
    md=ROOT.parent/'markdown/SPADE_mvtec_rscin_execution.md'
    text=md.read_text(encoding='utf-8').split('\n## 자체 측정 결과')[0]
    text+='\n## 자체 측정 결과\n\n15 category 비가중 평균, 단위 %.\n\n| RsCIN | Image AUROC | F1-max | AP |\n|---|---:|---:|---:|\n'
    for r in means:text+=f"| {r['rscin']} | {100*r['image_auroc']:.4f} | {100*r['image_f1_max']:.4f} | {100*r['image_ap']:.4f} |\n"
    md.write_text(text,encoding='utf-8')
    subprocess.run([sys.executable,str(LAB/'method10/source/build_musc_reproduction_tables.py')],check=True)
    subprocess.run([sys.executable,str(LAB/'method10/source/audit_musc_remaining_work.py')],check=True)
    subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',r'C:\Users\test\Desktop\Codex\tools\update_obsidian_links.ps1'],check=True)
    print('SPADE MVTEC RSCIN VERIFIED COMPLETE',means,flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        p=OUT/'environment.json'
        if p.exists():
            env=json.loads(p.read_text());env.update(status='failed',error=traceback.format_exc());p.write_text(json.dumps(env,indent=2))
        raise
