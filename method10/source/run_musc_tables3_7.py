"""Official MuSc modules, shared distance ablations, native efficiency measurement."""
import gc,importlib,importlib.metadata,json,os,random,subprocess,sys,time
from collections import defaultdict
import numpy as np
import torch
import yaml
from musc_tables3_7_common import *

OUT.mkdir(parents=True,exist_ok=True);RAW.mkdir(parents=True,exist_ok=True)
os.chdir(REPO);sys.path.insert(0,str(REPO));torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.manual_seed(42);np.random.seed(42);random.seed(42)
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==COMMIT
assert not subprocess.check_output(['git','diff','--name-only'],text=True).strip()
from models import musc as module
from models.modules._LNAMD import LNAMD
from models.modules._RsCIN import RsCIN
msm=importlib.import_module('models.modules._MSM')
module.tqdm=lambda x,*a,**kw:x;msm.tqdm=lambda x,*a,**kw:x
cfg=yaml.safe_load((REPO/'configs/musc.yaml').read_text())
cfg['device']='0';cfg['testing'].update(vis=False,save_excel=False,output_dir=str(RAW/'official'))
cfg['datasets'].update(dataset_name='mvtec_ad',data_path=DATA['mvtec_ad'],class_name='ALL',img_resize=518,divide_num=1)
cfg['models'].update(backbone_name='ViT-L-14-336',pretrained='openai',batch_size=4,feature_layers=[5,11,17,23],r_list=[1,3,5])
(OUT/'config.json').write_text(json.dumps(cfg,indent=2))
env=dict(status='starting',commit=COMMIT,gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,
    python=sys.version,seed=42,tf32=False,precision='native CUDA FP16 autocast; FP64 score aggregation; native RsCIN FP32',
    backbone='ViT-L-14-336',pretrained='openai',input_size=518,batch_size=4,feature_layers_0based=[5,11,17,23],
    checkpoint_sha256=sha('/home/test/.cache/clip/ViT-L-14-336px.pt'),
    datasets=DATA,tables=[3,4,5,6,7],training_executed=False,raw_path=str(RAW),
    source_sha256=sha(__file__),native_sources={p:sha(REPO/p) for p in ['models/musc.py','models/modules/_LNAMD.py','models/modules/_MSM.py','models/modules/_RsCIN.py','datasets/mvtec.py','datasets/visa.py']},
    efficiency='native full pipeline on VisA candle 200 images, warmup1/repeats3 per s, synchronized; report core/excluding RsCIN and total/including RsCIN separately; dataloading included, loading/metrics/saving excluded',
    sharing='extract full category once; share native fast cdist between six reductions; shared LNAMD embeddings across official stratified subsets; feature equivalence checked on first subset batch',
    packages={p:importlib.metadata.version(p) for p in ['torch','torchvision','numpy','scikit-learn','Pillow']})
def state():(OUT/'environment.json').write_text(json.dumps(env,indent=2))
state();model=module.MuSc(cfg,seed=42);model.clip_model.eval()
checks=[];all_rows=[]

def reductions(dist):
    k=int(dist.shape[1]*.3);assert k>0
    vals=torch.topk(dist.float(),k,largest=False,sorted=True).values
    reverse=torch.topk(vals,k,largest=True,sorted=True).values
    return dict(min=dist.float().min(1).values,max=dist.float().max(1).values,mean=dist.float().mean(1),
        **{'30_min':vals.min(1).values,'30_max':vals.max(1).values,'30_mean':reverse.mean(1)})

def scores_for(Z,all_strategies,proof_name):
    n,l,c=Z.shape;strategies=STRATEGIES if all_strategies else ['30_mean']
    result={s:torch.empty((n,l),dtype=torch.float64) for s in strategies}
    for i in range(n):
        ref=torch.cat((Z[:i],Z[i+1:]),0)
        d=torch.cdist(Z[i:i+1],ref.reshape(-1,c)).reshape(l,n-1,l).min(-1).values
        rs=reductions(d)
        for s in strategies:result[s][i]=rs[s].double().cpu()
        if i==0:
            native=msm.compute_scores_fast(Z,i,model.device,0,.3)
            error=float((native-rs['30_mean']).abs().max())
            assert torch.allclose(native,rs['30_mean'],rtol=1e-5,atol=1e-6),(proof_name,error)
            # CPU NumPy independently checks each reduction from the SAME native distance tensor.
            arr=d.float().cpu().numpy();sorted_dist=np.sort(arr,axis=1);k=int((n-1)*.3)
            refs={'min':arr.min(1),'max':arr.max(1),'mean':arr.mean(1),
                '30_min':sorted_dist[:,:k].min(1),'30_max':sorted_dist[:,:k].max(1),'30_mean':sorted_dist[:,:k].mean(1)}
            errors={s:float(np.max(np.abs(rs[s].cpu().numpy()-refs[s]))) for s in STRATEGIES}
            assert all(np.allclose(rs[s].cpu().numpy(),refs[s],rtol=1e-5,atol=1e-6) for s in STRATEGIES)
            checks.append(dict(scope=proof_name,n_images=n,native_MSM_max_abs=error,independent_reduction_max_abs=errors))
        del ref,d,rs
    return result

def category_run(ds,cat):
    model.dataset=ds;model.path=DATA[ds]
    folder=OUT/ds/cat;folder.mkdir(parents=True,exist_ok=True);rawfolder=RAW/ds/cat;rawfolder.mkdir(parents=True,exist_ok=True)
    if (folder/'verification.json').exists() and json.loads((folder/'verification.json').read_text())['status']=='passed':
        marker=json.loads((folder/'raw_manifest.json').read_text());assert sha(marker['path'])==marker['sha256']
        return read(folder/'metrics.csv')
    dataset=model.load_datasets(cat);loader=torch.utils.data.DataLoader(dataset,batch_size=4,shuffle=False,num_workers=0,pin_memory=True)
    paths=[];labels=[];masks=[];cls=[];tokens=[]
    with torch.no_grad():
        for info in loader:
            paths+=info['image_path'];labels+=info['is_anomaly'].tolist();masks.append(info['mask'].numpy().astype(np.int32))
            with torch.autocast('cuda',dtype=torch.float16):
                image=info['image'].float().to(model.device)
                feat,pt=model.clip_model.encode_image(image,model.features_list);feat/=feat.norm(dim=-1,keepdim=True)
            cls.append(feat.cpu().numpy());tokens.append([p.cpu() for p in pt])
    gt=np.concatenate(masks)[:,0].astype(np.uint8);y=np.array(labels);cls=np.concatenate(cls)
    n=len(paths);index={p:i for i,p in enumerate(paths)};assert len(index)==n
    groups={1:[list(range(n))]};subset_rows=[]
    for s in [2,3]:
        groups[s]=[]
        for j in range(s):
            sub=model.load_datasets(cat,divide_num=s,divide_iter=j)
            ids=[index[r[2]] for r in sub.data_to_iterate];groups[s].append(ids)
            subset_rows.extend(dict(s=s,subset=j,full_index=i,image_path=paths[i]) for i in ids)
        assert sorted(i for sub in groups[s] for i in sub)==list(range(n))
    # A subset's first batch can differ from the full pool's backbone batch layout.
    # Check cached vs independently re-encoded tokens; do not claim bitwise equality.
    sub=model.load_datasets(cat,divide_num=2,divide_iter=0)
    info=next(iter(torch.utils.data.DataLoader(sub,batch_size=4,shuffle=False,num_workers=0)))
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        _,independent=model.clip_model.encode_image(info['image'].float().to(model.device),model.features_list)
    for layer in range(4):
        cached=torch.stack([tokens[index[p]//4][layer][index[p]%4] for p in info['image_path']]).to(model.device)
        assert torch.allclose(cached,independent[layer],rtol=.01,atol=.005)
        checks.append(dict(scope=f'{ds}/{cat}/subset_feature_layer{layer}',max_abs=float((cached-independent[layer]).abs().max()),rtol=.01,atol=.005,bitwise_claim=False))
    del independent,cached,info
    write(folder/'subset_membership.csv',subset_rows)
    data={};split_scores={s:torch.zeros((n,1369),dtype=torch.float64) for s in [2,3]}
    with torch.no_grad():
        for r in [1,3,5]:
            ln=LNAMD(model.device,r=r,feature_dim=1024,feature_layer=model.features_list)
            layers=defaultdict(list)
            for pt in tokens:
                with torch.autocast('cuda',dtype=torch.float16):
                    f=ln._embed([p.to(model.device) for p in pt]);f/=f.norm(dim=-1,keepdim=True)
                assert torch.isfinite(f).all()
                for layer in range(4):layers[layer].append(f[:,:,layer,:])
            r_scores={strategy:[] for strategy in STRATEGIES}
            for layer in range(4):
                Z=torch.cat(layers[layer],0).to(model.device)
                scored=scores_for(Z,True,f'{ds}/{cat}/r{r}/layer{layer}/s1')
                for strategy in STRATEGIES:r_scores[strategy].append(scored[strategy])
                for s in [2,3]:
                    for j,ids in enumerate(groups[s]):
                        idx=torch.tensor(ids,device=model.device)
                        v=scores_for(Z[idx],False,f'{ds}/{cat}/r{r}/layer{layer}/s{s}/subset{j}')['30_mean']
                        split_scores[s][ids]+=v/12
                del Z,scored,layers[layer];torch.cuda.empty_cache()
            for strategy in STRATEGIES:data[f'r{r}_{strategy}']=torch.stack(r_scores[strategy]).mean(0).numpy()
            del r_scores,ln,layers;gc.collect()
    data['s1']=np.mean(np.stack([data[f'r{r}_30_mean'] for r in [1,3,5]]),0)
    for s in [2,3]:data[f's{s}']=split_scores[s].numpy()
    data.update(image_labels=y,masks=gt,class_tokens=cls)
    path=rawfolder/'patch_predictions.npz';np.savez_compressed(path,**data)
    manifest=dict(path=str(path),sha256=sha(path),map_reconstruction='FP64 patch scores -> torch bilinear 518 align_corners=True; masks native dataset cast int32 then uint8',config_sha256=sha(OUT/'config.json'))
    (folder/'raw_manifest.json').write_text(json.dumps(manifest,indent=2))
    imgrows=[]
    for row,label in zip(dataset.data_to_iterate,y):
        imgrows.append(dict(image_path=row[2],label=int(label),sha256=sha(row[2]),mask_path=row[3] or '',mask_sha256=sha(row[3]) if row[3] else ''))
    write(folder/'images.csv',imgrows)
    specs=[(3,','.join(map(str,rs))) for rs in RSETS]+[(4,s) for s in STRATEGIES]+[(5,'w/o'),(5,'w')]+[(7,f's{s}') for s in [1,2,3]]
    rows=[];score_rows=[];cache={}
    for table,condition in specs:
        patches=patch_for(data,table,condition)
        key=hashlib.sha256(patches.tobytes()).hexdigest()
        if key not in cache:
            maps=upsample(patches);before=maps.reshape(n,-1).max(1)
            after=RsCIN(before,cls,k_list=[1,2,3] if ds=='mvtec_ad' else [1,8,9])
            assert np.isfinite(maps).all()
            cache[key]=(before,after,float(roc_auc_score(gt.ravel(),maps.ravel())))
            del maps;gc.collect()
        before,after,pixel=cache[key]
        scores=before if table==5 and condition=='w/o' else after
        assert np.isfinite(scores).all()
        metric=image_metrics(y,scores)
        pixel=pixel if table!=5 else float('nan')
        s=int(condition[1:]) if table==7 else 1
        sizes=[len(ids) for ids in groups[s]]
        rows.append(dict(table=table,dataset=ds,category=cat,condition=condition,n_images=n,**metric,pixel_auroc=pixel,subset_min=min(sizes),subset_max=max(sizes)))
        score_rows.extend(dict(table=table,condition=condition,image_path=p,label=int(label),score_before=float(b),score_after=float(a),score_used=float(v)) for p,label,b,a,v in zip(paths,y,before,after,scores))
        gc.collect()
    write(folder/'metrics.csv',rows);write(folder/'image_scores.csv',score_rows)
    (folder/'module_checks.json').write_text(json.dumps([c for c in checks if c['scope'].startswith(ds+'/'+cat+'/')],indent=2))
    (folder/'verification.json').write_text(json.dumps(dict(status='passed',scope='module/formula equivalence and finite outputs; final independent metric verification follows',raw_sha256=manifest['sha256']),indent=2))
    del data,tokens,split_scores;gc.collect();torch.cuda.empty_cache()
    return rows

def benchmark():
    file=OUT/'timing_samples.csv'
    if file.exists() and (OUT/'benchmark_verification.json').exists():return
    model.dataset='visa';model.path=DATA['visa'];model.r_list=[1,3,5]
    assert len(model.load_datasets('candle'))==200
    original_rscin=module.RsCIN;original_metrics=module.compute_metrics
    timings=[]
    for s in [1,2,3]:
        model.divide_num=s
        for rep in range(-1,3):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();stops={}
            def timed_rscin(*a,**kw):
                torch.cuda.synchronize();stops['core']=time.perf_counter()
                result=original_rscin(*a,**kw);torch.cuda.synchronize();stops['total']=time.perf_counter();return result
            module.RsCIN=timed_rscin;module.compute_metrics=lambda *a:((0,0,0),(0,0,0,0))
            try:model.make_category_data('candle')
            finally:module.RsCIN=original_rscin;module.compute_metrics=original_metrics
            torch.cuda.synchronize()
            if rep>=0:timings.append(dict(s=s,repeat=rep,n_images=200,core_ms_per_image=(stops['core']-start)*1000/200,
                total_ms_per_image=(stops['total']-start)*1000/200,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                gpu_cost_decimal_MB=torch.cuda.max_memory_allocated()/1e6,gpu_cost_MiB=torch.cuda.max_memory_allocated()/1024**2))
            gc.collect();torch.cuda.empty_cache()
    write(file,timings)
    summary=[]
    for s in [1,2,3]:
        rs=[r for r in timings if r['s']==s]
        summary.append(dict(s=s,n_images=200,repeats=3,core_ms_per_image=float(np.mean([r['core_ms_per_image'] for r in rs])),
            total_ms_per_image=float(np.mean([r['total_ms_per_image'] for r in rs])),gpu_cost_MB=max(r['gpu_cost_decimal_MB'] for r in rs),gpu_cost_MiB=max(r['gpu_cost_MiB'] for r in rs)))
    write(OUT/'benchmark_summary.csv',summary)
    (OUT/'benchmark_verification.json').write_text(json.dumps(dict(status='passed',native_make_category_data=True,pool='VisA/candle',n_images=200,
        warmup=1,repeats=3,synchronized=True,peak_reset_each_repeat=True,MB_definition='allocated_bytes/1e6; reserved and MiB also stored',
        scope='core includes data loading/encoder/LNAMD/MSM/interpolation; excludes model load/RsCIN/metrics/saving; total also includes RsCIN',
        timing_samples_sha256=sha(file)),indent=2))

try:
    for ds,categories in [('mvtec_ad',module._CLASSNAMES_mvtec_ad),('visa',module._CLASSNAMES_visa)]:
        for cat in categories:
            env.update(status='running_accuracy',dataset=ds,category=cat);state()
            print('CATEGORY START',ds,cat,flush=True)
            all_rows+=category_run(ds,cat);write(OUT/'category_metrics.csv',all_rows)
            print('CATEGORY COMPLETE',ds,cat,flush=True)
    env['status']='benchmarking';state();benchmark()
    env['status']='verifying';state()
    subprocess.run([sys.executable,str(ROOT/'finish_musc_tables3_7.py')],check=True)
    env['status']='completed';state();print('MUSC TABLES3-7 COMPLETE',flush=True)
except Exception as e:
    env.update(status='failed',error=repr(e));state();raise
