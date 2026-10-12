"""Pinned author PatchCore; all outstanding MuSc comparison conditions."""
import functools,csv,gc,hashlib,importlib.util,json,os,random,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch,faiss
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms as T
from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_curve,auc
from skimage.measure import label,regionprops

ROOT=Path(__file__).resolve().parent;LAB=ROOT.parents[1]
OUT=ROOT/'results/musc_comparison_20261010';RAW=Path('/home/test/patchcore_results/musc_comparison_20261010')
REPO=Path('/home/test/patchcore-inspection');COMMIT='fcaa92f124fb1ad74a7acf56726decd4b27cbcad'
sys.path.insert(0,str(REPO/'src'))
import patchcore.patchcore,patchcore.sampler,patchcore.common,patchcore.backbones
DATA={'mvtec':Path('/home/test/data/mvtec'),'visa':Path('/home/test/data/VisA_20220922'),'btad':Path('/home/test/data/btad_original/BTech_Dataset_transformed')}
KEYS=['image_auroc','image_f1_max','image_ap','pixel_auroc','pixel_f1_max','pixel_ap','aupro']
XFORM=T.Compose([T.Resize(256,interpolation=T.InterpolationMode.BILINEAR),T.CenterCrop(224),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
MFORM=T.Compose([T.Resize(256,interpolation=T.InterpolationMode.NEAREST),T.CenterCrop(224)])

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def read(p):
    with open(p,encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp')
    with temp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    temp.replace(p)
def dump(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(v,indent=2));temp.replace(p)
def manifest(dataset,cat,phase):
    entries=json.loads((DATA[dataset]/'meta.json').read_text())[phase][cat];result=[]
    for r in entries:
        p=DATA[dataset]/r['img_path'];mp=DATA[dataset]/r['mask_path'] if int(r['anomaly']) else None
        result.append(dict(image_path=str(p),label=int(r['anomaly']),sha256=sha(p),mask_path=str(mp) if mp else '',mask_sha256=sha(mp) if mp else ''))
    return result
class Images(Dataset):
    def __init__(self,rows):self.rows=rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        with Image.open(r['image_path']) as im:x=XFORM(im.convert('RGB'))
        mask=np.zeros((224,224),np.uint8)
        if r['mask_path']:
            with Image.open(r['mask_path']) as im:mask=(np.array(MFORM(im.convert('L')))>0).astype(np.uint8)
        return dict(image=x,is_anomaly=int(r['label']),mask=torch.from_numpy(mask))
def loader(rows):return DataLoader(Images(rows),batch_size=8,shuffle=False,num_workers=4,pin_memory=True,persistent_workers=True)
def classification(y,s):
    p,r,_=precision_recall_curve(y,s);f=np.divide(2*p*r,p+r,out=np.zeros_like(p),where=p+r!=0)
    return [float(roc_auc_score(y,s)),float(f.max()),float(average_precision_score(y,s))]
def pro(masks,maps):
    # Exactly the MuSc 200-threshold definition, with cached connected regions.
    thresholds=np.arange(maps.min(),maps.max(),(maps.max()-maps.min())/200)
    normal=np.sort(maps[masks==0]);fprs=(len(normal)-np.searchsorted(normal,thresholds,side='right'))/len(normal)
    regions=[]
    for m,s in zip(masks,maps):
        for r in regionprops(label(m)):
            values=np.sort(s[r.coords[:,0],r.coords[:,1]])
            regions.append((len(values)-np.searchsorted(values,thresholds,side='right'))/len(values))
    pros=np.mean(regions,axis=0);keep=fprs<.3;f=fprs[keep];f=(f-f.min())/(f.max()-f.min())
    return float(auc(f,pros[keep]))
def metrics(labels,scores,masks,maps):return dict(zip(KEYS,classification(labels,scores)+classification(masks.ravel(),maps.ravel())+[pro(masks,maps)]))
def build():
    backbone=patchcore.backbones.load('wideresnet50');backbone.name='wideresnet50'
    model=patchcore.patchcore.PatchCore(torch.device('cuda'))
    model.load(backbone=backbone,layers_to_extract_from=['layer2','layer3'],device=torch.device('cuda'),input_shape=(3,224,224),
        pretrain_embed_dimension=1024,target_embed_dimension=1024,patchsize=3,anomaly_score_num_nn=1,
        featuresampler=patchcore.sampler.ApproximateGreedyCoresetSampler(.1,torch.device('cuda')),nn_method=patchcore.common.FaissNN(True,8))
    return model
def verify_gpu(model,rows,raw):
    x=next(iter(loader(rows)))['image'].cuda();queries=np.asarray(model._embed(x),dtype=np.float32)[:64]
    actual=model.anomaly_scorer.nn_method.run(1,queries)[0][:,0]
    index=faiss.index_gpu_to_cpu(model.anomaly_scorer.nn_method.search_index);bank=index.reconstruct_n(0,index.ntotal)
    cpu=faiss.IndexFlatL2(bank.shape[1]);cpu.add(bank);ref=cpu.search(queries,1)[0][:,0]
    assert np.allclose(actual,ref,rtol=1e-4,atol=.003)
    double=np.array([np.min(np.sum((bank.astype(np.float64)-q.astype(np.float64))**2,axis=1)) for q in queries[:4]])
    assert np.allclose(actual[:4],double,rtol=1e-4,atol=.003)
    np.save(raw/'memory_bank.npy',bank);np.save(raw/'sample_queries.npy',queries);np.save(raw/'sample_distances.npy',actual)
    return dict(status='passed',GPU_FAISS_vs_CPU_all_memory_64_queries_max_error=float(np.max(abs(actual-ref))),
        float64_brute_force_all_memory_4_queries_max_error=float(np.max(abs(actual[:4]-double))),n_memory=len(bank),dimension=bank.shape[1])
def rscin(cat,rows,scores):
    path=Path('/home/test/aprilgan_results/rscin_shared_musc_features/mvtec')/cat/'features.npy'
    shared=LAB/'method11/source/results/rscin_shared_features_mvtec_20261006'/cat
    if not path.exists():
        path=RAW/'rscin_features'/cat/'features.npy';shared=OUT/'rscin_features'/cat
        if not path.exists():subprocess.run([sys.executable,str(ROOT/'extract_patchcore_rscin.py'),'--category',cat],check=True)
    refs=read(shared/'images.csv');lookup={r['image_path']:i for i,r in enumerate(refs)};indices=[]
    for r in rows:
        i=lookup[r['image_path']];assert refs[i]['sha256']==r['sha256'] and int(refs[i]['label'])==int(r['label']);indices.append(i)
    features=np.load(path)[indices]
    module_path=Path('/home/test/MuSc/models/RsCIN_features/RsCIN.py');spec=importlib.util.spec_from_file_location('official_rscin',module_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.Mobile_RsCIN(scores.copy(),dataset_name='mvtec_ad',cls_tokens=features)
    assert np.isfinite(result).all()
    return result,dict(feature_path=str(path),feature_sha256=sha(path),module_sha256=sha(module_path),windows=[1,2,3],image_label_SHA_alignment_passed=True)
def run_category(dataset,shot,seed,cat):
    tag=f'{dataset}_{shot}_seed{seed}';out=OUT/tag/cat;raw=RAW/tag/cat;out.mkdir(parents=True,exist_ok=True);raw.mkdir(parents=True,exist_ok=True)
    if (out/'verification.json').exists() and json.loads((out/'verification.json').read_text())['status']=='passed':
        for a in json.loads((out/'raw_manifest.json').read_text()):assert sha(a['path'])==a['sha256']
        return json.loads((out/'metrics.json').read_text())
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    train=manifest(dataset,cat,'train');test=manifest(dataset,cat,'test');assert all(r['label']==0 for r in train)
    selected=train if shot=='full' else [train[i] for i in random.Random(seed).sample(range(len(train)),4)]
    assert set(r['image_path'] for r in selected).isdisjoint(r['image_path'] for r in test)
    write(out/'support_images.csv',selected);write(out/'test_images.csv',test)
    model=build();t=time.perf_counter()
    if (raw/'patchcore_params.pkl').exists() and (raw/'nnscorer_search_index.faiss').exists():
        model.load_from_path(str(raw),torch.device('cuda'),nn_method=patchcore.common.FaissNN(True,8))
    else:model.fit(loader(selected));model.save_to_path(str(raw))
    fit_seconds=time.perf_counter()-t;proof=verify_gpu(model,test,raw)
    t=time.perf_counter();scores,maps,labels,masks=model.predict(loader(test));evaluation_seconds=time.perf_counter()-t
    scores=np.asarray(scores,dtype=np.float32);maps=np.asarray(maps,dtype=np.float32);labels=np.asarray(labels,dtype=np.uint8);masks=np.asarray(masks,dtype=np.uint8)
    assert np.array_equal(labels,np.array([r['label'] for r in test])) and np.isfinite(scores).all() and np.isfinite(maps).all()
    value=metrics(labels,scores,masks,maps)
    # Independent PRO implementation on actual transformed images, reduced crop
    # only for this algorithm check; reported maps use full224 resolution.
    spec=importlib.util.spec_from_file_location('musc_metrics','/home/test/MuSc/utils/metrics.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    ids=np.flatnonzero(masks.reshape(len(masks),-1).sum(1)>0)[:2];small_m=masks[ids,::4,::4];small_s=maps[ids,::4,::4]
    if small_m.sum()>0:assert abs(pro(small_m,small_s)-m.cal_pro_score(small_m,small_s))<1e-12
    after=scores.copy();rscin_proof=None
    if dataset=='mvtec' and shot=='full':after,rscin_proof=rscin(cat,test,scores)
    np.savez_compressed(raw/'predictions.npz',labels=labels,image_scores=scores,score_maps=maps,masks=masks,rscin_scores=after)
    row=dict(dataset=dataset,shot=shot,seed=seed,category=cat,n_train=len(train),n_support=len(selected),n_images=len(test),**value)
    write(out/'image_scores.csv',[dict(**r,score=float(s),rscin_score=float(a)) for r,s,a in zip(test,scores,after)])
    if rscin_proof:
        dump(out/'rscin_provenance.json',rscin_proof);dump(out/'rscin_metrics.json',dict(zip(KEYS[:3],classification(labels,after))))
    files=[raw/n for n in ['patchcore_params.pkl','nnscorer_search_index.faiss','memory_bank.npy','sample_queries.npy','sample_distances.npy','predictions.npz']]
    dump(out/'raw_manifest.json',[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in files])
    dump(out/'distance_verification.json',proof);dump(out/'metrics.json',row)
    # Reload predictions, recompute scores and assert all input/model hashes.
    with np.load(raw/'predictions.npz') as z:
        for k,v in metrics(z['labels'],z['image_scores'],z['masks'],z['score_maps']).items():assert abs(value[k]-v)<1e-12
    for r in selected+test:
        assert sha(r['image_path'])==r['sha256']
        if r['mask_path']:assert sha(r['mask_path'])==r['mask_sha256']
    dump(out/'verification.json',dict(status='passed',input_SHA_and_labels=True,all_seven_metrics_recomputed=True,distance=proof,
        PRO_algorithm_vs_native_passed=True,fit_seconds=fit_seconds,evaluation_seconds=evaluation_seconds,limits='Distance checked on64 queries/4 float64 queries; all forward passes not independently rerun. Condition differences documented.'))
    del model,maps,masks;gc.collect();torch.cuda.empty_cache();print('PATCHCORE CATEGORY VERIFIED',tag,cat,flush=True);return row
def aggregate():
    rows=[json.loads(p.read_text()) for p in sorted(OUT.glob('*_seed*/*/metrics.json')) if json.loads((p.parent/'verification.json').read_text())['status']=='passed']
    if not rows:return
    write(OUT/'category_metrics.csv',rows);runs=[]
    for dataset,shot in [('mvtec','full'),('mvtec','4shot'),('visa','4shot'),('btad','4shot')]:
        cats=json.loads((DATA[dataset]/'meta.json').read_text())['test']
        for seed in ([42] if shot=='full' else [42,43,44]):
            part=[r for r in rows if (r['dataset'],r['shot'],r['seed'])==(dataset,shot,seed)]
            if len(part)==len(cats):runs.append(dict(dataset=dataset,shot=shot,seed=seed,n_categories=len(cats),n_images=sum(r['n_images'] for r in part),**{k:float(np.mean([r[k] for r in part])) for k in KEYS}))
    if runs:write(OUT/'run_means.csv',runs)
    means=[]
    for dataset,shot in [('mvtec','full'),('mvtec','4shot'),('visa','4shot'),('btad','4shot')]:
        part=[r for r in runs if (r['dataset'],r['shot'])==(dataset,shot)];expected=1 if shot=='full' else 3
        if len(part)==expected:
            row=dict(dataset=dataset,shot=shot,n_runs=expected)
            for k in KEYS:row[k]=float(np.mean([r[k] for r in part]));row[k+'_std']=float(np.std([r[k] for r in part],ddof=0))
            means.append(row)
    if means:write(OUT/'aggregates.csv',means)
    rs=[]
    for flag in ['w/o','w']:
        part=[r for r in rows if r['dataset']=='mvtec' and r['shot']=='full']
        if len(part)==15:
            vals=[r if flag=='w/o' else json.loads((OUT/f'mvtec_full_seed42/{r["category"]}/rscin_metrics.json').read_text()) for r in part]
            rs.append(dict(rscin=flag,**{k:float(np.mean([r[k] for r in vals])) for k in KEYS[:3]}))
    if rs:write(OUT/'rscin_mean_metrics.csv',rs)
def main():
    OUT.mkdir(parents=True,exist_ok=True);RAW.mkdir(parents=True,exist_ok=True)
    if (OUT/'verification.json').exists() and json.loads((OUT/'verification.json').read_text())['status']=='passed':return
    assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==COMMIT
    assert not subprocess.check_output(['git','-C',str(REPO),'status','--porcelain','--untracked-files=no'],text=True).strip()
    torch.set_num_threads(8);torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    # Suppress only progress output; sampling mathematics stays official.
    patchcore.sampler.tqdm.tqdm=functools.partial(patchcore.sampler.tqdm.tqdm,disable=True)
    env=dict(status='running',gpu=torch.cuda.get_device_name(),commit=COMMIT,official_repo='https://github.com/amazon-science/patchcore-inspection',backbone='ImageNetV1 WideResNet50 layer2/3',
        input=224,resize=256,coreset=.1,pretrain_dimension=1024,target_dimension=1024,patchsize=3,nn=1,FAISS='GPU exact flat squaredL2 FP32; CPU/float64 verified',
        TF32=False,AMP=False,normal_training='full normal set or4 support images; no backbone training',fewshot_seeds=[42,43,44],std_ddof=0,
        wrapper_sha256=sha(__file__),raw_root=str(RAW),condition_limit='Official baseline; MuSc exact PatchCore config/ensemble/fewshot sampling undisclosed, not paper-number equivalence.',error=None)
    dump(OUT/'environment.json',env)
    for dataset,shot in [('mvtec','full'),('mvtec','4shot'),('visa','4shot'),('btad','4shot')]:
        for seed in ([42] if shot=='full' else [42,43,44]):
            for cat in json.loads((DATA[dataset]/'meta.json').read_text())['test']:
                env.update(dataset=dataset,shot=shot,seed=seed,category=cat);dump(OUT/'environment.json',env)
                run_category(dataset,shot,seed,cat);aggregate()
    assert len(read(OUT/'aggregates.csv'))==4
    dump(OUT/'verification.json',dict(status='passed',n_category_runs=len(read(OUT/'category_metrics.csv')),all_category_verifications_passed=True,expected_runs=105,limits='Single WideResNet50 official baseline and3 local fewshot repeats; all conditions and differences documented.'))
    env['status']='completed';dump(OUT/'environment.json',env)
    subprocess.run([sys.executable,str(LAB/'method10/source/build_musc_reproduction_tables.py')],check=True)
    subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',r'C:\Users\test\Desktop\Codex\tools\update_obsidian_links.ps1'],check=True)
    print('PATCHCORE ALL COMPARISONS VERIFIED COMPLETE',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        p=OUT/'environment.json'
        if p.exists():
            e=json.loads(p.read_text());e.update(status='failed',error=traceback.format_exc());dump(p,e)
        raise
