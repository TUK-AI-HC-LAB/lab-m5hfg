"""Fig7 MSM upper-percent sweep and Fig8 native RsCIN window sweep."""
import gc,importlib,json,os,random,subprocess,sys
from collections import defaultdict
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from musc_tables3_7_common import ROOT,REPO,COMMIT,DATA,sha,read,write,image_metrics,upsample
BASE=ROOT/'results/tables3_7_seed42_20261012'
TAG='sweeps_seed42_20261012';OUT=ROOT/'results'/TAG
from pathlib import Path
RAW=Path('/home/test/musc_results')/TAG
OUT.mkdir(parents=True,exist_ok=True);RAW.mkdir(parents=True,exist_ok=True)
assert json.loads((BASE/'verification.json').read_text())['status']=='passed'
os.chdir(REPO);sys.path.insert(0,str(REPO));torch.set_num_threads(8)
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==COMMIT
assert not subprocess.check_output(['git','diff','--name-only'],text=True).strip()
from models.modules._RsCIN import RsCIN
from models.modules._LNAMD import LNAMD
from models import musc as module
msm=importlib.import_module('models.modules._MSM')
module.tqdm=lambda x,*a,**kw:x;msm.tqdm=lambda x,*a,**kw:x
PCTS=[10,30,50,70,90,100]
env=dict(status='starting',commit=COMMIT,seed=42,baseline=str(BASE),baseline_verification_sha256=sha(BASE/'verification.json'),
    dataset_categories={'mvtec_ad':15,'visa':12},training_executed=False,
    MSM='Fig7 intervals 0%–p%, p=10/30/50/70/90/100; shared native fast cdist; not Table10 lower-bound sweep',
    RsCIN='Fig8 k=2..9 and full-window, plus w/o and native multi-window reference',
    precision='native FP16 autocast, FP64 layer/r aggregation, FP32 RsCIN; TF32 disabled',
    raw_path=str(RAW),source_sha256=sha(__file__),no_eta_or_progress_poll=True)
def state():(OUT/'environment.json').write_text(json.dumps(env,indent=2))
def independent_rscin(before,cls,ks):
    sim=(cls@cls.T).astype(np.float32);n=len(before)
    score=((before-before.min())/(before.max()-before.min())).astype(np.float32);parts=[]
    for k in ks:
        ids=torch.topk(torch.from_numpy(sim),n-k,largest=False,sorted=True).indices.numpy()
        w=sim.copy();w[np.arange(n)[:,None],ids]=0;parts.append((w/w.sum(1,keepdims=True))@score)
    return np.mean(np.stack(parts,1),1)
def aggregate(rows,condition_key,name):
    groups=defaultdict(list)
    for r in rows:groups[(r['dataset'],str(r[condition_key]))].append(r)
    means=[]
    for (ds,condition),rs in groups.items():
        assert len(rs)==(15 if ds=='mvtec_ad' else 12)
        row=dict(dataset=ds,**{condition_key:condition},n_categories=len(rs),n_images=sum(int(r['n_images']) for r in rs))
        for key in ['image_auroc','image_f1_max','image_ap','pixel_auroc']:
            if key in rs[0]:row[key]=float(np.mean([float(r[key]) for r in rs]))
        means.append(row)
    write(OUT/name,means)

categories=sorted({(r['dataset'],r['category']) for r in read(BASE/'category_metrics.csv')})
state()
try:
    env['status']='running_rscin';state();rscin_rows=[];rscin_checks=[]
    for ds,cat in categories:
        base=BASE/ds/cat;manifest=json.loads((base/'raw_manifest.json').read_text())
        assert sha(manifest['path'])==manifest['sha256']
        with np.load(manifest['path']) as z:cls=z['class_tokens'];y=z['image_labels']
        records=[r for r in read(base/'image_scores.csv') if r['table']=='5' and r['condition']=='w/o']
        before=np.array([float(r['score_used']) for r in records]);assert len(before)==len(y)
        spec=[('w/o',None),('multi',[1,2,3] if ds=='mvtec_ad' else [1,8,9])]+[(str(k),[k]) for k in range(2,10)]+[('full',[len(y)])]
        scores_rows=[]
        for condition,ks in spec:
            scores=before if ks is None else RsCIN(before,cls,k_list=ks)
            assert np.isfinite(scores).all()
            error=0. if ks is None else float(np.max(np.abs(scores-independent_rscin(before,cls,ks))))
            assert error<=1e-5
            rscin_rows.append(dict(dataset=ds,category=cat,window=condition,n_images=len(y),**image_metrics(y,scores)))
            scores_rows.extend(dict(window=condition,image_path=r['image_path'],label=int(label),score=float(s)) for r,label,s in zip(records,y,scores))
            rscin_checks.append(dict(dataset=ds,category=cat,window=condition,numpy_smoothing_max_abs=error))
        folder=OUT/ds/cat;write(folder/'rscin_window_image_scores.csv',scores_rows)
    assert len(rscin_rows)==297
    write(OUT/'rscin_single_window_category_metrics.csv',rscin_rows)
    aggregate(rscin_rows,'window','rscin_single_window_mean_metrics.csv')
    (OUT/'rscin_verification.json').write_text(json.dumps(dict(status='passed',n_categories=27,n_images=3887,n_category_conditions=297,
        baseline_raw_hashes_checked=True,native_RsCIN_numpy_smoothing_checked=True,checks=rscin_checks),indent=2))
    print('RSCIN WINDOW SWEEP VERIFIED COMPLETE',flush=True)
    torch.manual_seed(42);np.random.seed(42);random.seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=json.loads((BASE/'config.json').read_text());cfg['testing']['output_dir']=str(RAW/'official')
    model=module.MuSc(cfg,seed=42);model.clip_model.eval()
    env.update(status='running_msm',gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,config=cfg);state()
    all_rows=[]
    for ds,cat in categories:
        folder=OUT/ds/cat;folder.mkdir(parents=True,exist_ok=True);rawfolder=RAW/ds/cat;rawfolder.mkdir(parents=True,exist_ok=True)
        marker=folder/'msm_verification.json'
        if marker.exists() and json.loads(marker.read_text())['status']=='passed':
            saved=json.loads((folder/'raw_manifest.json').read_text());assert sha(saved['path'])==saved['sha256']
            all_rows+=read(folder/'msm_metrics.csv');continue
        env.update(dataset=ds,category=cat);state();print('MSM CATEGORY START',ds,cat,flush=True)
        model.dataset=ds;model.path=DATA[ds];dataset=model.load_datasets(cat)
        base=BASE/ds/cat;image_records=read(base/'images.csv')
        assert [r[2] for r in dataset.data_to_iterate]==[r['image_path'] for r in image_records]
        with np.load(json.loads((base/'raw_manifest.json').read_text())['path']) as z:
            y=z['image_labels'];gt=z['masks'];baseline=z['s1'];old_cls=z['class_tokens']
        tokens=[];cls=[]
        with torch.no_grad():
            for info in torch.utils.data.DataLoader(dataset,batch_size=4,shuffle=False,num_workers=0,pin_memory=True):
                with torch.autocast('cuda',dtype=torch.float16):
                    features,pt=model.clip_model.encode_image(info['image'].float().to(model.device),model.features_list)
                    features/=features.norm(dim=-1,keepdim=True)
                cls.append(features.cpu().numpy());tokens.append([p.cpu() for p in pt])
        cls=np.concatenate(cls);assert np.allclose(cls,old_cls,rtol=1e-5,atol=1e-6)
        n=len(y);sums={p:torch.zeros((n,1369),dtype=torch.float64) for p in PCTS};proof={};checks=[]
        with torch.no_grad():
            for r in [1,3,5]:
                ln=LNAMD(model.device,r=r,feature_dim=1024,feature_layer=model.features_list);layers=defaultdict(list)
                for pt in tokens:
                    with torch.autocast('cuda',dtype=torch.float16):
                        f=ln._embed([p.to(model.device) for p in pt]);f/=f.norm(dim=-1,keepdim=True)
                    for layer in range(4):layers[layer].append(f[:,:,layer,:])
                for layer in range(4):
                    Z=torch.cat(layers[layer],0).to(model.device)
                    for i in range(n):
                        ref=torch.cat((Z[:i],Z[i+1:]),0)
                        dist=torch.cdist(Z[i:i+1],ref.reshape(-1,1024)).reshape(1369,n-1,1369).min(-1).values
                        for p in PCTS:
                            k=int((n-1)*(p/100)) if p<100 else n-1;assert k>0
                            vals=torch.topk(dist.float(),k,largest=False,sorted=True).values
                            vals=torch.topk(vals,k,largest=True,sorted=True).values
                            value=vals.mean(1)
                            sums[p][i]+=value.double().cpu()/12
                            if i==0:
                                # Official API treats >=1 as a count, not a fraction.
                                native=msm.compute_scores_fast(Z,i,model.device,0,p/100 if p<100 else n-1)
                                err=float((native-value).abs().max());assert torch.allclose(native,value,rtol=1e-5,atol=1e-6)
                                cpu=np.sort(dist.float().cpu().numpy(),axis=1)[:,:k].mean(1)
                                assert np.allclose(value.cpu().numpy(),cpu,rtol=1e-5,atol=1e-6)
                                checks.append(dict(r=r,layer=layer,percent=p,native_max_abs=err,numpy_mean_max_abs=float(np.max(np.abs(value.cpu().numpy()-cpu)))))
                        if i==0:proof[f'first_query_r{r}_layer{layer}']=dist.float().cpu().numpy()
                        del ref,dist
                    del Z,layers[layer];torch.cuda.empty_cache()
                del ln,layers;gc.collect()
        patches={f'p{p}':sums[p].numpy() for p in PCTS}
        baseline_error=float(np.max(np.abs(patches['p30']-baseline)))
        assert np.allclose(patches['p30'],baseline,rtol=1e-5,atol=1e-6),baseline_error
        path=rawfolder/'interval_patch_predictions.npz';np.savez_compressed(path,**patches,**proof,class_tokens=cls)
        (folder/'raw_manifest.json').write_text(json.dumps(dict(path=str(path),sha256=sha(path),baseline_manifest=str(base/'raw_manifest.json'),baseline_raw_sha256=json.loads((base/'raw_manifest.json').read_text())['sha256']),indent=2))
        rows=[];saved_scores=[]
        for p in PCTS:
            maps=upsample(patches[f'p{p}']);before=maps.reshape(n,-1).max(1)
            after=RsCIN(before,cls,k_list=[1,2,3] if ds=='mvtec_ad' else [1,8,9])
            assert np.isfinite(maps).all() and np.isfinite(after).all()
            rows.append(dict(dataset=ds,category=cat,percent=p,n_images=n,**image_metrics(y,after),pixel_auroc=float(roc_auc_score(gt.ravel(),maps.ravel()))))
            saved_scores.extend(dict(percent=p,image_path=im['image_path'],label=int(label),score_before=float(b),score_after=float(a)) for im,label,b,a in zip(image_records,y,before,after))
            del maps;gc.collect()
        write(folder/'msm_metrics.csv',rows);write(folder/'msm_image_scores.csv',saved_scores)
        marker.write_text(json.dumps(dict(status='passed',native_query_and_numpy_reduction_checks=checks,baseline30_patch_max_abs=baseline_error,raw_sha256=sha(path)),indent=2))
        all_rows+=rows;write(OUT/'msm_interval_ratio_category_metrics.csv',all_rows)
        print('MSM CATEGORY COMPLETE',ds,cat,flush=True)
        del tokens,sums,patches,proof;gc.collect();torch.cuda.empty_cache()
    assert len(all_rows)==162
    aggregate(all_rows,'percent','msm_interval_ratio_mean_metrics.csv')
    env['status']='verifying_msm';state()
    subprocess.run([sys.executable,str(ROOT/'finish_musc_sweeps.py')],check=True)
    env['status']='completed';state();print('MUSC SWEEPS ALL COMPLETE',flush=True)
except Exception as e:
    env.update(status='failed',error=repr(e));state();raise
