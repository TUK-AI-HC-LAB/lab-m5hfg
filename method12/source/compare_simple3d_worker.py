"""Fresh category benchmark: shared minimally fixed baseline, optional GPU KD-tree.

Timing wrappers call the original operations. No compatibility operator substitution.
Nested stage measurements overlap and cannot be summed into inference wall time.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
REPO=Path('/mnt/c/Users/test/Desktop/Codex/Project/Simple3D-official-compatible-20261104')
ROOT=HERE/'results/official_vs_gpu_kdtree_20261104'
ASSETS=Path('/home/test/simple3d-comparison-20261104')
sys.path.insert(0,str(REPO))
import numpy as np
import torch
import open3d as o3d
import feature_extractors.features as features_module
import feature_extractors.FPFH as fpfh_module
from feature_extractors.FPFH import FPFHFeatures
from patchcore_runner import PatchCore
from pointnet2_ops import pointnet2_utils as ops
from sklearn.metrics import roc_auc_score,average_precision_score

base=SimpleNamespace(FPFHFeatures=FPFHFeatures,features_module=features_module)
CONTEXT=dict(phase='startup',index=-1)
PENDING=[]


def dump(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def append(path,rows):
    if not rows:return
    exists=path.exists()
    with path.open('a',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        if not exists:writer.writeheader()
        writer.writerows(rows)


def observed(name,fn,device,detail):
    context=CONTEXT.copy()
    if device.startswith('cuda'):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record();answer=fn();end.record();value=(start,end);clock='cuda_event'
    else:
        start=time.perf_counter();answer=fn();value=time.perf_counter()-start;clock='wall'
    PENDING.append(dict(**context,stage=name,device=device,detail=detail,clock=clock,value=value))
    return answer


def flush(out):
    torch.cuda.synchronize()
    rows=[]
    for item in PENDING:
        item=item.copy();v=item.pop('value')
        item['seconds']=v[0].elapsed_time(v[1])/1000 if isinstance(v,tuple) else v
        rows.append(item)
    append(out/'stage_timings.csv',rows);PENDING.clear()


def instrument():
    class Cloud:
        def __init__(self,*a,**kw):self.native=o3d.geometry.PointCloud(*a,**kw)
        def estimate_normals(self,param):
            return observed('normal_estimation_including_neighbor_search',lambda:self.native.estimate_normals(param),
                'cpu','legacy Open3D normals radius10000 k10')
    def fpfh(cloud,param):
        return observed('fpfh_scale_'+str(param.max_nn),
            lambda:o3d.pipelines.registration.compute_fpfh_feature(cloud.native,param),'cpu','official legacy CPU FPFH')
    fpfh_module.o3d=SimpleNamespace(utility=o3d.utility,
        geometry=SimpleNamespace(PointCloud=Cloud,KDTreeSearchParamHybrid=o3d.geometry.KDTreeSearchParamHybrid),
        pipelines=SimpleNamespace(registration=SimpleNamespace(compute_fpfh_feature=fpfh)))
    OriginalKNN=fpfh_module.KNN
    class TimedKNN:
        def __init__(self,*a,**kw):self.native=OriginalKNN(*a,**kw)
        def __call__(self,r,q):
            return observed('coordinate_knn',lambda:self.native(r,q),str(r.device),
                f'native KNN_CUDA ref={list(r.shape)} query={list(q.shape)}')
    fpfh_module.KNN=features_module.KNN=TimedKNN
    def wrap_module(module,name,stage,device):
        original=getattr(module,name)
        def call(*a,**kw):
            return observed(stage,lambda:original(*a,**kw),device(a),str([list(x.shape) for x in a if isinstance(x,torch.Tensor)]))
        setattr(module,name,call)
    wrap_module(ops,'furthest_point_sample','fps',lambda a:str(a[0].device))
    wrap_module(ops,'gather_operation','fps_gather',lambda a:str(a[0].device))
    wrap_module(features_module,'interpolating_points_chunked','point_score_interpolation',lambda a:str(a[0].device))
    wrap_module(features_module.Features,'unorganized_data_to_organized','scoremap_reorganization_cpu_gpu',lambda a:'cpu_gpu_wall')
    for name in ['cdist','min','mean','topk','cat']:
        original=getattr(torch,name)
        def make(original,name):
            def call(*a,**kw):
                tensor=a[0] if a and isinstance(a[0],torch.Tensor) else None
                if name=='cat':
                    tensor=a[0][0] if a and isinstance(a[0],(list,tuple)) and a[0] else None
                if tensor is None:return original(*a,**kw)
                stage={'cdist':'memory_bank_cdist','min':'nearest_neighbor_min','mean':'feature_or_score_mean',
                    'topk':'image_top80','cat':'feature_concatenation'}[name]
                if name in ('cdist','min','topk') and CONTEXT['phase']!='test':return original(*a,**kw)
                return observed(stage,lambda:original(*a,**kw),str(tensor.device),f'shape={list(tensor.shape)}')
            return call
        setattr(torch,name,make(original,name))
    projection=features_module.random_projection.SparseRandomProjection
    original=projection.fit_transform
    def project(self,x,*a,**kw):
        start=time.perf_counter()
        try:return observed('coreset_random_projection',lambda:original(self,x,*a,**kw),'cpu',str(list(x.shape)))
        except ValueError:
            PENDING.append(dict(**CONTEXT,stage='coreset_projection_fallback',device='cpu',detail='official fallback',
                clock='wall',value=time.perf_counter()-start));raise
    projection.fit_transform=project


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['official','gpu_kdtree'],required=True)
    parser.add_argument('--dataset',required=True);parser.add_argument('--category',required=True)
    parser.add_argument('--smoke',action='store_true');opts=parser.parse_args()
    out=(ROOT/'preflight' if opts.smoke else ROOT)/opts.mode/opts.dataset/opts.category
    out.mkdir(parents=True,exist_ok=True)
    if (out/'train_timings.csv').exists():raise RuntimeError('Existing run preserved')
    asset=(ASSETS/'preflight' if opts.smoke else ASSETS)/opts.mode/opts.dataset/opts.category
    asset.mkdir(parents=True,exist_ok=True)
    def progress(phase,index,total):
        CONTEXT.update(phase=phase,index=index)
        dump(out/'progress.json',dict(status='running',mode=opts.mode,dataset=opts.dataset,
            category=opts.category,phase=phase,index=index,total=total,updated_unix=time.time(),pid=os.getpid()))
    start_all=time.perf_counter()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    args=SimpleNamespace(dataset=opts.dataset,device='cuda:0',num_group=4096,group_size=128,
        max_nn=40,use_LFSA=True,use_MSND=True,num_MSND=2,feature='FPFH',level='ALL',vis_save=False)
    model=PatchCore(args=args)
    train=model.get_dataloader(opts.dataset,'train',opts.category,'ALL')
    test=model.get_dataloader(opts.dataset,'test',opts.category,'ALL')
    assert len(train)>0 and len(test)>0
    manifest=dict(train=train.dataset.pcd_paths,test=[dict(path=p,label=int(y)) for p,y in zip(test.dataset.pcd_paths,test.dataset.labels)])
    dump(out/'manifest.json',manifest)
    env=dict(args=vars(args),torch=torch.__version__,open3d=o3d.__version__,gpu=torch.cuda.get_device_name(),
        official_source=str(REPO),fresh_fit=True,smoke=opts.smoke,mode=opts.mode,
        source_sha256={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in REPO.rglob('*.py')},
        measurement='instrumented CUDA events plus host wall time; nested stages overlap; not pure uninstrumented latency')
    dump(out/'environment.json',env)
    instrument()
    if opts.mode=='gpu_kdtree':
        import simple3d_gpu_kdtree as backend
        backend.install(sys.modules[__name__],detail_tracing=True)
    # Both modes start from this identical minimally fixed source; KD-tree mode only replaces its backend.
    with torch.inference_mode():
        progress('warmup',0,1)
        pc,*_=next(iter(train));model.method.get_features(pc);torch.cuda.synchronize();flush(out)
        torch.cuda.reset_peak_memory_stats()
        ntrain=1 if opts.smoke else len(train)
        progress('fit',0,ntrain);fit_start=time.perf_counter()
        for i,(pc,_,_,path) in enumerate(train):
            progress('fit',i,ntrain);torch.cuda.synchronize();t=time.perf_counter()
            model.method.collect_features(pc);torch.cuda.synchronize()
            append(out/'train_timings.csv',[dict(index=i,path=path[0],points=pc.shape[1],feature_s=time.perf_counter()-t)])
            flush(out)
            if opts.smoke:break
        progress('coreset',0,1)
        observed('coreset_total',model.method.run_coreset,'cpu_gpu_wall','official 5% coreset');torch.cuda.synchronize()
        fit_s=time.perf_counter()-fit_start;flush(out)
        observed('memory_bank_save',lambda:torch.save(model.method.patch_lib.detach().cpu(),asset/'memory_bank.pt'),'cpu_gpu_wall','new bank')
        flush(out)
        rows=[];iterator=iter(test);ntest=2 if opts.smoke else len(test)
        for i in range(ntest):
            progress('test',i,ntest)
            pc,mask,label,path=observed('input_loading',lambda:next(iterator),'cpu_wall','official DataLoader with worker')
            model.method.init_para();torch.cuda.synchronize();t=time.perf_counter()
            feat,raw,dense,centers=model.method.get_features(pc);torch.cuda.synchronize();feature_s=time.perf_counter()-t
            t=time.perf_counter()
            model.method.compute_anomay_scores(feat,mask,label,path,raw,dense,centers)
            torch.cuda.synchronize();score_s=time.perf_counter()-t
            pred=np.asarray(model.method.predictions[-1],dtype=np.float32).reshape(-1).copy()
            gt=np.asarray(model.method.gts[-1],dtype=np.uint8).reshape(-1).copy()
            assert len(pred)==len(gt)==pc.shape[1] and np.isfinite(pred).all()
            row=dict(index=i,path=path[0],label=int(label.item()),points=pc.shape[1],
                object_score=float(model.method.image_preds[-1]),feature_s=feature_s,score_s=score_s,inference_s=feature_s+score_s)
            rows.append(row);append(out/'test_timings.csv',[row])
            progress('save',i,ntest)
            observed('point_scores_save',lambda:np.savez_compressed(asset/f'{i:05d}.npz',scores=pred,labels=gt),'cpu_wall','new raw scores')
            flush(out)
            del pred,gt,feat,raw,dense,centers;model.method.init_para()
            print('TEST',opts.mode,opts.dataset,opts.category,i+1,ntest,flush=True)
        peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
    metrics=[]
    if not opts.smoke:
        for level in (['easy','medium','hard','ALL'] if opts.dataset=='minishift' else ['ALL']):
            progress('metrics_'+level,0,1)
            chosen=[r for r in rows if r['label']==0 or level=='ALL' or '/'+level+'/' in r['path']]
            def calculate():
                pp=[];gg=[]
                for r in chosen:
                    with np.load(asset/f"{r['index']:05d}.npz") as z:pp.append(z['scores']);gg.append(z['labels'])
                p=np.concatenate(pp);g=np.concatenate(gg)
                return dict(object_auroc=float(roc_auc_score([r['label'] for r in chosen],[r['object_score'] for r in chosen])),
                    point_auroc=float(roc_auc_score(g,p)),point_ap=float(average_precision_score(g,p)))
            metric=observed('metrics_'+level,calculate,'cpu_wall','includes raw reload and CPU AUROC/AP')
            metrics.append(dict(mode=opts.mode,dataset=opts.dataset,category=opts.category,level=level,**metric))
            flush(out)
        append(out/'metrics.csv',metrics)
    values=np.asarray([r['inference_s'] for r in rows])*1000
    summary=dict(status='passed_smoke' if opts.smoke else 'complete',mode=opts.mode,dataset=opts.dataset,
        category=opts.category,train_count=ntrain,test_count=ntest,fit_wall_s=fit_s,
        inference_mean_ms=float(values.mean()),inference_median_ms=float(np.median(values)),
        inference_p95_ms=float(np.percentile(values,95)),peak_torch_allocated_bytes=peak,peak_torch_reserved_bytes=reserved,
        elapsed_wall_s=time.perf_counter()-start_all,metrics=metrics,raw_assets=str(asset))
    grouped={}
    with (out/'stage_timings.csv').open() as f:
        for r in csv.DictReader(f):grouped.setdefault(tuple(r[k] for k in ['phase','stage','device','detail','clock']),[]).append(float(r['seconds']))
    append(out/'stage_summary.csv',[dict(zip(['phase','stage','device','detail','clock'],k),calls=len(v),
        mean_ms=float(np.mean(v)*1000),total_s=sum(v)) for k,v in grouped.items()])
    dump(out/'summary.json',summary);progress('complete',ntest,ntest)
    print('CATEGORY COMPLETE',opts.mode,opts.dataset,opts.category,flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        print(traceback.format_exc(),flush=True);raise
