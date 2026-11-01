"""Single-category full-resolution benchmark of official Simple3D code."""
import argparse, csv, json, os, platform, subprocess, sys, threading, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import psutil
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

HERE=Path(__file__).resolve().parent
REPO=Path('/mnt/c/Users/test/Desktop/Codex/Project/MiniShift-Simple3D')
OUT=HERE/'results'/'capsule_20261020'
OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(REPO))
from simple3d_compat import install, fast_interpolation
install()
import feature_extractors.features as features_module
features_module.interpolating_points_chunked=fast_interpolation
_organize=features_module.Features.unorganized_data_to_organized
def organize_on_device(self,pc,values):
    # Upstream returns CPU score maps then indexes them with CUDA indices.
    return [x.to(self.args.device) for x in _organize(self,pc,values)]
features_module.Features.unorganized_data_to_organized=organize_on_device
from feature_extractors.FPFH import FPFHFeatures
import data.MiniShiftAD as dataset_module

def sync(): torch.cuda.synchronize()
def dump(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def size(path):
    p=Path(path)
    if not p.exists(): return 0
    if p.is_file(): return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob('*') if f.is_file() and not f.is_symlink())
def stats(v):
    a=np.array(v,dtype=float)
    return dict(mean_ms=float(a.mean()*1000),median_ms=float(np.median(a)*1000),
                p95_ms=float(np.percentile(a,95)*1000),min_ms=float(a.min()*1000),
                max_ms=float(a.max()*1000),throughput_samples_s=float(1/a.mean()))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--max-test',type=int,default=0)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume-fit',action='store_true'); opts=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    args=SimpleNamespace(dataset='minishift',device='cuda:0',num_group=4096,
        group_size=128,max_nn=40,use_LFSA=True,use_MSND=True,num_MSND=2,
        feature='FPFH',level='ALL',vis_save=False)
    dataset_module.DATASETS_PATH='/home/test/data/MiniShift'
    train=dataset_module.MiniShiftADTrain('capsule')
    test=dataset_module.MiniShiftADTest('capsule',level='ALL')
    # Upstream str.replace('test','gt') also changes /home/test to /home/gt.
    root=Path(dataset_module.DATASETS_PATH)/'capsule'
    test.gt_paths=[0 if y==0 else str(root/'gt'/Path(p).relative_to(root/'test'))
                   for p,y in zip(test.pcd_paths,test.labels)]
    if not train.pcd_paths or not test.pcd_paths: raise RuntimeError('Dataset empty')
    files=[dict(split='train',path=p,label=0) for p in train.pcd_paths]
    files += [dict(split='test',path=p,label=int(y),gt=str(g)) for p,y,g in zip(test.pcd_paths,test.labels,test.gt_paths)]
    dump('manifest.json',files)
    env=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),platform=platform.platform(),
        upstream_commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
        args=vars(args),train_count=len(train),test_count=len(test),
        compatibility='NVRTC FPS (512 threads), exact scipy cKDTree KNN, nearest-three topk interpolation; missing unused data.quan stub',
        msnd_note='Official num_MSND=2 code concatenates scale-2 twice; preserved verbatim.')
    dump('environment.json',env)
    subprocess.run([sys.executable,'-m','pip','freeze'],stdout=(OUT/'pip_freeze.txt').open('w'))
    m=FPFHFeatures(args=args)
    # Validate FPS against scalar reference on a small, tie-free cloud.
    from simple3d_compat import furthest_point_sample
    rng=np.random.default_rng(123); x=rng.normal(size=(100,3)).astype('float32')
    ix=[0];md=np.full(100,1e10,dtype='float32')
    for _ in range(15):
        md=np.minimum(md,((x-x[ix[-1]])**2).sum(1));ix.append(int(md.argmax()))
    actual=furthest_point_sample(torch.tensor(x,device='cuda')[None],16).cpu().numpy()[0]
    assert np.array_equal(actual,ix),(actual,ix)
    print('FPS correctness check passed',flush=True)
    from simple3d_compat import KNN
    from feature_extractors.pointnet2_utils import interpolating_points_chunked
    ref=torch.tensor(x,device='cuda')[None]
    query=ref[:,::7,:].contiguous()
    _,idx=KNN(5)(ref,query)
    exact=torch.cdist(query,ref).topk(5,largest=False).indices
    assert torch.equal(idx,exact)
    vals=torch.arange(query.shape[1],device='cuda',dtype=torch.float32)[None,None]
    torch.testing.assert_close(fast_interpolation(ref.transpose(1,2),query.transpose(1,2),vals),
        interpolating_points_chunked(ref.transpose(1,2),query.transpose(1,2),vals))
    print('Exact KNN and interpolation reference checks passed',flush=True)
    pc,*_=train[0]; pc=torch.as_tensor(pc).unsqueeze(0)
    t=time.perf_counter(); m.get_features(pc); sync(); warmup=time.perf_counter()-t
    print(f'Warmup {warmup:.3f}s, input={tuple(pc.shape)}',flush=True)
    if opts.smoke:
        m.collect_features(pc); m.run_coreset(); m.init_para()
        pc,mask,label,path=test[0]
        m.predict(torch.as_tensor(pc)[None],mask,torch.tensor([label]),[path]);sync()
        print('SMOKE OK',m.image_preds,len(m.pixel_preds),flush=True);return

    peak=[0];stop=threading.Event();proc=psutil.Process()
    def monitor():
        while not stop.wait(.05): peak[0]=max(peak[0],proc.memory_info().rss)
    threading.Thread(target=monitor,daemon=True).start()
    torch.cuda.reset_peak_memory_stats()
    if opts.resume_fit:
        state=json.loads((OUT/'fit_state.json').read_text())
        trainrows=state['trainrows'];fit_s=state['fit_s'];coreset_s=state['coreset_s']
        m.patch_lib=torch.load(OUT/'memory_bank.pt',map_location='cpu',weights_only=True)['memory_bank']
    else:
        trainrows=[];fit_start=time.perf_counter()
        with torch.inference_mode():
            for i in range(len(train)):
                t=time.perf_counter();pc,*_=train[i];load=time.perf_counter()-t
                sync();t=time.perf_counter();m.collect_features(torch.as_tensor(pc)[None]);sync();feature=time.perf_counter()-t
                trainrows.append(dict(index=i,path=train.pcd_paths[i],points=len(pc),load_s=load,feature_s=feature))
                print(f'TRAIN {i+1}/{len(train)} load={load:.3f}s feature={feature:.3f}s',flush=True)
            t=time.perf_counter();m.run_coreset();sync();coreset_s=time.perf_counter()-t
        fit_s=time.perf_counter()-fit_start
        torch.save(dict(memory_bank=m.patch_lib.cpu(),args=vars(args),upstream_commit=env['upstream_commit']),OUT/'memory_bank.pt')
        state=dict(trainrows=trainrows,fit_s=fit_s,coreset_s=coreset_s,timing_origin='direct full-precision measurement')
        dump('fit_state.json',state)
    m.patch_lib=m.patch_lib.cuda()
    print(f'FIT COMPLETE {fit_s:.3f}s bank={tuple(m.patch_lib.shape)}',flush=True)
    testrows=[];allpred=[];allgt=[];imagepred=[];imagelabel=[]
    score_dir=OUT/'point_scores';score_dir.mkdir(exist_ok=True)
    n=len(test) if not opts.max_test else min(len(test),opts.max_test)
    eval_start=time.perf_counter()
    with torch.inference_mode():
        for i in range(n):
            t0=time.perf_counter();pc,mask,label,path=test[i];load=time.perf_counter()-t0
            pc=torch.as_tensor(pc)[None];label_t=torch.tensor([label])
            m.init_para();sync();t=time.perf_counter()
            f,raw,dense,centers=m.get_features(pc);sync();feat=time.perf_counter()-t
            t=time.perf_counter();m.compute_anomay_scores(f,mask,label_t,[path],raw,dense,centers);sync();score=time.perf_counter()-t
            pred=np.asarray(m.predictions[-1],dtype=np.float32).reshape(-1).copy()
            gt=np.asarray(m.gts[-1],dtype=np.uint8).reshape(-1).copy()
            assert np.isfinite(pred).all()
            image_score=float(m.image_preds[-1]); allpred.append(pred);allgt.append(gt)
            imagepred.append(image_score);imagelabel.append(label)
            t=time.perf_counter();np.savez_compressed(score_dir/f'{i:03d}.npz',scores=pred,labels=gt);save=time.perf_counter()-t
            testrows.append(dict(index=i,path=path,label=label,points=pc.shape[1],anomalous_points=int(gt.sum()),image_score=image_score,
                load_s=load,feature_s=feat,score_s=score,inference_s=feat+score,load_plus_inference_s=load+feat+score,save_s=save))
            print(f'TEST {i+1}/{n} label={label} points={pc.shape[1]} load={load:.3f}s feat={feat:.3f}s score={score:.3f}s',flush=True)
            # Release upstream Python per-point scalar lists each sample.
            m.init_para()
    eval_wall=time.perf_counter()-eval_start
    t=time.perf_counter();pred=np.concatenate(allpred);gt=np.concatenate(allgt);allpred.clear();allgt.clear()
    metrics=dict(object_auroc=float(roc_auc_score(imagelabel,imagepred)),
                 object_average_precision=float(average_precision_score(imagelabel,imagepred)),
                 point_auroc=float(roc_auc_score(gt,pred)),
                 point_average_precision=float(average_precision_score(gt,pred)),
                 point_count=len(gt),point_positive_count=int(gt.sum()),
                 au_pro=None,au_pro_reason='Official implementation does not compute AU-PRO for MiniShift; its zero is a placeholder.')
    metric_s=time.perf_counter()-t
    for name,rows in [('train_timings.csv',trainrows),('test_timings.csv',testrows)]:
        with (OUT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report=dict(category='capsule',level='ALL',full_category=n==len(test),train_count=len(train),test_count=n,
        metrics=metrics,warmup_s=warmup,fit_wall_s=fit_s,train_load_s=sum(r['load_s'] for r in trainrows),
        train_feature_s=sum(r['feature_s'] for r in trainrows),coreset_s=coreset_s,
        inference=stats([r['inference_s'] for r in testrows]),
        load_plus_inference=stats([r['load_plus_inference_s'] for r in testrows]),
        feature=stats([r['feature_s'] for r in testrows]),scoring=stats([r['score_s'] for r in testrows]),
        evaluation_wall_including_saving_s=eval_wall,metric_calculation_s=metric_s,
        peak_process_rss_bytes=peak[0],peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
        timing_scope='Warmup excluded. CPU geometry + transfers + GPU scoring included; per-sample loading and saving recorded separately. Official compute_anomay_scores includes CPU metric-collection overhead.',
        fit_timing_origin=state.get('timing_origin'),resumed_fit=opts.resume_fit,
        memory_scope='Current process measured phases only; fit memory excluded when resumed_fit is true.',
        caveat='Compatibility backends differ from original old CUDA extensions; local timings are not paper FPS reproduction.')
    dump('summary.json',report)
    roots=dict(dataset_including_zip='/home/test/data/MiniShift',environment='/home/test/simple3d-env',
               upstream_repository=str(REPO),method_artifacts=str(HERE.parent))
    storage={k:dict(path=p,bytes=size(p)) for k,p in roots.items()}
    storage['total_bytes']=sum(v['bytes'] for v in storage.values())
    storage['note']='Logical retained file bytes. Includes ZIP + extracted capsule data, dedicated environment, upstream checkout and method files. Excludes shared pip/conda caches, OS/driver, and this storage report itself.'
    dump('storage.json',storage);stop.set()
    print(json.dumps(report,indent=2),flush=True)
    print('STORAGE',json.dumps(storage,indent=2),flush=True)

if __name__=='__main__': main()
