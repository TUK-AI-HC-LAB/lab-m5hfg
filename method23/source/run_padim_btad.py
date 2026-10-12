"""BTAD full-shot PaDiM WR50-Rd550, batched GPU covariance and inference."""
import ast,csv,hashlib,json,platform,random,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms as T
from torchvision.models import wide_resnet50_2,Wide_ResNet50_2_Weights
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parent;LAB=ROOT.parents[1]
REPO=Path('/home/test/PaDiM-public');COMMIT='616004b9b7fa6a507d196dd3eeeba9e0a93f160f'
DATA=Path('/home/test/data/btad_original/BTech_Dataset_transformed')
RAW=Path('/home/test/padim_results/btad_wr50_d550_seed1024_20261010')
OUT=ROOT/'results/btad_wr50_d550_seed1024_20261010'
XFORM=T.Compose([T.Resize((256,256),interpolation=T.InterpolationMode.BICUBIC),T.CenterCrop(224),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
MFORM=T.Compose([T.Resize((256,256),interpolation=T.InterpolationMode.NEAREST),T.CenterCrop(224)])
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,rows):
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def manifest(meta,cat,phase):
    rows=[]
    for r in meta[phase][cat]:
        p=DATA/r['img_path'];mp=DATA/r['mask_path'] if r.get('mask_path') else None
        rows.append(dict(image_path=str(p),label=int(r['anomaly']),sha256=sha(p),
                         mask_path=str(mp) if mp else '',mask_sha256=sha(mp) if mp else ''))
    return rows
class Images(Dataset):
    def __init__(self,rows):self.rows=rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        with Image.open(self.rows[i]['image_path']) as im:return XFORM(im.convert('RGB'))
def concat(outputs):
    return torch.cat([outputs[0],F.interpolate(outputs[1],size=(56,56),mode='nearest'),F.interpolate(outputs[2],size=(56,56),mode='nearest')],1)
def check_concat(outputs):
    # Execute the pinned public unfold/fold function without running its main().
    tree=ast.parse((REPO/'main.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='embedding_concat')
    ns={'torch':torch,'F':F};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(REPO/'main.py'),'exec'),ns)
    actual=[x[:1].cpu() for x in outputs]
    reference=ns['embedding_concat'](ns['embedding_concat'](actual[0],actual[1]),actual[2])
    assert torch.equal(reference,concat(actual))
@torch.inference_mode()
def extract(model,outputs,rows,path,idx):
    a=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(len(rows),550,3136))
    offset=0
    for x in DataLoader(Images(rows),batch_size=16,num_workers=4,pin_memory=True,persistent_workers=True):
        outputs.clear();model(x.cuda(non_blocking=True));assert len(outputs)==3
        if offset==0:check_concat(outputs)
        z=concat(outputs).index_select(1,idx).flatten(2).cpu().numpy()
        a[offset:offset+len(x)]=z;offset+=len(x)
    assert offset==len(rows);a.flush();outputs.clear();return a
@torch.inference_mode()
def fit(train,raw):
    n=len(train);mean=np.lib.format.open_memmap(raw/'mean.npy',mode='w+',dtype='float32',shape=(3136,550))
    precision=np.lib.format.open_memmap(raw/'precision.npy',mode='w+',dtype='float32',shape=(3136,550,550))
    proofs=[]
    for start in range(0,3136,32):
        stop=min(start+32,3136)
        a=torch.from_numpy(np.array(train[:,:,start:stop],copy=True)).permute(2,0,1).cuda().double()
        avg=a.mean(1);delta=a-avg[:,None,:]
        cov=(delta.transpose(1,2)@delta)/(n-1)
        cov+=.01*torch.eye(550,device='cuda',dtype=torch.float64)
        # Native public implementation stores np.cov in float32 before inverse.
        cov32=cov.float();inv=torch.linalg.inv(cov32)
        assert torch.isfinite(inv).all()
        mean[start:stop]=avg.float().cpu().numpy();precision[start:stop]=inv.cpu().numpy()
        if start in [0,1568,3104]:
            p=start;x=np.array(train[:,:,p],dtype=np.float64)
            ref_cov=np.cov(x,rowvar=False)+.01*np.eye(550)
            assert np.allclose(cov[0].cpu().numpy(),ref_cov,rtol=1e-10,atol=1e-10)
            ref_inv=np.linalg.inv(ref_cov.astype(np.float32).astype(np.float64))
            got=inv[0].cpu().numpy()
            assert np.allclose(got,ref_inv,rtol=2e-3,atol=2e-3)
            proofs.append(dict(position=p,covariance_max_error=float(np.max(np.abs(cov[0].cpu().numpy()-ref_cov))),
                               precision_max_error=float(np.max(np.abs(got-ref_inv)))))
    mean.flush();precision.flush();return mean,precision,proofs
@torch.inference_mode()
def distances(test,mean,precision):
    results=np.empty((len(test),3136),dtype=np.float32)
    for start in range(0,3136,32):
        stop=min(start+32,3136)
        q=torch.from_numpy(np.array(test[:,:,start:stop],copy=True)).permute(2,0,1).cuda()
        mu=torch.from_numpy(np.array(mean[start:stop],copy=True)).cuda()
        inv=torch.from_numpy(np.array(precision[start:stop],copy=True)).cuda()
        delta=q-mu[:,None,:]
        squared=((delta@inv)*delta).sum(-1)
        assert torch.isfinite(squared).all() and float(squared.min())>-1e-3
        results[:,start:stop]=squared.clamp_min(0).sqrt().T.cpu().numpy()
    return results.reshape(-1,56,56)
def maps_from_dist(d):
    a=F.interpolate(torch.from_numpy(d).unsqueeze(1),size=(224,224),mode='bicubic',align_corners=False).squeeze(1).numpy()
    return np.stack([gaussian_filter(m,sigma=4) for m in a])
def masks(rows):
    result=[]
    for r in rows:
        m=np.zeros((224,224),dtype=np.uint8)
        if r['mask_path']:
            with Image.open(r['mask_path']) as im:m=(np.array(MFORM(im.convert('L')))>128).astype(np.uint8)
        result.append(m)
    return np.stack(result)
def category(model,outputs,meta,cat,idx):
    raw=RAW/cat;out=OUT/cat;raw.mkdir(exist_ok=True);out.mkdir(exist_ok=True)
    if (out/'verification.json').exists():
        proof=json.loads((out/'verification.json').read_text())
        if proof['status']=='passed':
            for a in proof['raw_artifacts']:assert sha(a['path'])==a['sha256']
            return json.loads((out/'metrics.json').read_text())
    t=time.perf_counter();train=manifest(meta,cat,'train');test=manifest(meta,cat,'test')
    assert all(r['label']==0 for r in train) and set(r['image_path'] for r in train).isdisjoint(r['image_path'] for r in test)
    write(out/'train_images.csv',train);write(out/'test_images.csv',test)
    tr=extract(model,outputs,train,raw/'temporary_train_embeddings.npy',idx)
    print('FIT START',cat,flush=True)
    mu,inv,fitproof=fit(tr,raw)
    te=extract(model,outputs,test,raw/'temporary_test_embeddings.npy',idx)
    d=distances(te,mu,inv);mdproof=[]
    for p in [0,1568,3104]:
        cov=np.cov(np.array(tr[:,:,p],dtype=np.float64),rowvar=False)+.01*np.eye(550)
        ci=np.linalg.inv(cov.astype(np.float32).astype(np.float64))
        for i in [0,len(test)//2,len(test)-1]:
            delta=np.array(te[i,:,p],dtype=np.float64)-mu[p].astype(np.float64)
            ref=float(np.sqrt(delta@ci@delta))
            actual=float(d.reshape(len(test),-1)[i,p])
            assert np.isclose(actual,ref,rtol=1e-3,atol=1e-3),(actual,ref)
            mdproof.append(dict(image=i,position=p,gpu=actual,independent_fp64=ref))
    maps=maps_from_dist(d);gt=masks(test);labels=np.array([r['label'] for r in test],dtype=np.uint8)
    scores=maps.reshape(len(test),-1).max(1)
    assert np.isfinite(maps).all() and np.isfinite(scores).all()
    pred=raw/'predictions.npz';np.savez_compressed(pred,labels=labels,masks=gt,image_scores=scores,score_maps=maps,mahalanobis56=d)
    np.save(raw/'channel_indices.npy',idx.cpu().numpy())
    row=dict(dataset='btad',category=cat,n_train=len(train),n_images=len(test),
             image_auroc=float(roc_auc_score(labels,scores)),pixel_auroc=float(roc_auc_score(gt.ravel(),maps.ravel())),seconds=time.perf_counter()-t)
    saved=np.load(pred);assert np.array_equal(saved['labels'],labels) and np.array_equal(saved['masks'],masks(test))
    assert np.array_equal(saved['score_maps'],maps_from_dist(saved['mahalanobis56']))
    assert np.array_equal(saved['image_scores'],saved['score_maps'].reshape(len(test),-1).max(1))
    assert abs(roc_auc_score(saved['labels'],saved['image_scores'])-row['image_auroc'])<1e-12
    assert abs(roc_auc_score(saved['masks'].ravel(),saved['score_maps'].ravel())-row['pixel_auroc'])<1e-12
    for r in train+test:
        assert sha(r['image_path'])==r['sha256']
        if r['mask_path']:assert sha(r['mask_path'])==r['mask_sha256']
    artifacts=[dict(path=str(p),sha256=sha(p)) for p in [pred,raw/'mean.npy',raw/'precision.npy',raw/'channel_indices.npy']]
    (out/'metrics.json').write_text(json.dumps(row,indent=2))
    write(out/'image_scores.csv',[dict(image_path=r['image_path'],label=r['label'],score=float(s)) for r,s in zip(test,scores)])
    proof=dict(status='passed',embedding_concat_equals_pinned_public_function=True,fit_proof=fitproof,mahalanobis_proof=mdproof,
               all_input_mask_SHA_verified=True,all_maps_postprocess_and_metrics_recomputed=True,raw_artifacts=artifacts,
               limits='3 spatial covariance positions and9 actual image/patch distances independently verified; not every covariance or forward double-executed')
    (out/'verification.json').write_text(json.dumps(proof,indent=2))
    # Explicit temporary feature files; retained parameters reproduce predictions.
    del tr,te,mu,inv,saved
    for name in ['temporary_train_embeddings.npy','temporary_test_embeddings.npy']:(raw/name).unlink()
    torch.cuda.empty_cache();print('CATEGORY VERIFIED',row,flush=True);return row
def main():
    OUT.mkdir(parents=True,exist_ok=True);RAW.mkdir(parents=True,exist_ok=True)
    if (OUT/'verification.json').exists() and json.loads((OUT/'verification.json').read_text())['status']=='passed':return
    assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==COMMIT
    assert not subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip()
    random.seed(1024);np.random.seed(1024);torch.manual_seed(1024);torch.cuda.manual_seed_all(1024)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');torch.backends.cudnn.benchmark=True
    idx=torch.tensor(random.sample(range(1792),550),device='cuda')
    w=Wide_ResNet50_2_Weights.IMAGENET1K_V1;model=wide_resnet50_2(weights=w).cuda().eval();outputs=[]
    for module in [model.layer1[-1],model.layer2[-1],model.layer3[-1]]:module.register_forward_hook(lambda m,x,y:outputs.append(y.detach()))
    weight=Path(torch.hub.get_dir())/'checkpoints'/Path(w.url).name
    meta=json.loads((DATA/'meta.json').read_text());assert sorted(meta['test'])==['01','02','03']
    env=dict(status='running',category='01',dataset=str(DATA),seed=1024,backbone='WideResNet50x2 IMAGENET1K_V1',dimension=550,
             full_shot=True,gradient_training_epochs=0,epsilon=.01,covariance_ddof=1,
             input='256x256 BICUBIC ->crop224 ->ImageNet normalization',map='56->224 bicubic align_cornersFalse Gaussian sigma4',
             mask='256x256 NEAREST crop224 threshold128; original anomaly labels retained including empty BTAD02 mask',
             batch=16,workers=4,spatial_chunk=32,fit='FP64 sample covariance, storedFP32 before CUDAinverse; meanFP32',
             precision='features/distance FP32; no AMP or TF32',gpu=torch.cuda.get_device_name(),python=platform.python_version(),torch=torch.__version__,
             public_repo='https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master',official_author_code=False,commit=COMMIT,
             code_sha256=sha(__file__),source_sha256=sha(REPO/'main.py'),weights_sha256=sha(weight),raw_root=str(RAW),
             protocol_limit='MuSc exact PaDiM backbone/seed/resize not disclosed; public loader LANCZOS/bilinear replaced with paper bicubic; evaluate cropped224 field',error=None)
    (OUT/'environment.json').write_text(json.dumps(env,indent=2));rows=[]
    for cat in ['01','02','03']:
        env['category']=cat;(OUT/'environment.json').write_text(json.dumps(env,indent=2))
        rows.append(category(model,outputs,meta,cat,idx));write(OUT/'category_metrics.csv',rows)
    assert sum(r['n_images'] for r in rows)==741
    mean=dict(dataset='btad',n_categories=3,n_images=741,**{k:float(np.mean([float(r[k]) for r in rows])) for k in ['image_auroc','pixel_auroc']})
    write(OUT/'mean_metrics.csv',[mean])
    assert sha(__file__)==env['code_sha256'] and sha(REPO/'main.py')==env['source_sha256'] and sha(weight)==env['weights_sha256']
    assert all(json.loads((OUT/c/'verification.json').read_text())['status']=='passed' for c in ['01','02','03'])
    (OUT/'verification.json').write_text(json.dumps(dict(status='passed',n_categories=3,n_images=741,all_category_verifications_passed=True),indent=2))
    env['status']='completed';(OUT/'environment.json').write_text(json.dumps(env,indent=2))
    md=ROOT.parent/'markdown/PaDiM_btad_execution.md';text=md.read_text(encoding='utf-8').split('\n## 자체 측정 결과')[0]
    text+='\n## 자체 측정 결과\n\n제품3개 비가중 평균, 단위 %.\n\n| Product | Image AUROC | Pixel AUROC |\n|---|---:|---:|\n'
    for r in rows+[dict(category='Mean',**mean)]:text+=f"| {r['category']} | {100*r['image_auroc']:.4f} | {100*r['pixel_auroc']:.4f} |\n"
    md.write_text(text,encoding='utf-8')
    subprocess.run([sys.executable,str(LAB/'method10/source/build_musc_reproduction_tables.py')],check=True)
    subprocess.run([sys.executable,str(LAB/'method10/source/audit_musc_remaining_work.py')],check=True)
    subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',r'C:\Users\test\Desktop\Codex\tools\update_obsidian_links.ps1'],check=True)
    print('PADIM BTAD VERIFIED COMPLETE',mean,flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        p=OUT/'environment.json'
        if p.exists():
            env=json.loads(p.read_text());env.update(status='failed',error=traceback.format_exc());p.write_text(json.dumps(env,indent=2))
        raise
