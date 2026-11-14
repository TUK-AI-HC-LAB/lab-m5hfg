"""Compare the baseline and a STEP-inspired 3D indexing engine, not full AD."""
import argparse
import csv
import json
import statistics
from pathlib import Path

import cupy as cp
import numpy as np
import open3d as o3d
import torch
from cupyx.scipy.spatial import KDTree

from measure_simple3d_capsule_flashfps import PatchCore, SimpleNamespace
from measure_simple3d_capsule_morton128 import BASE, HERE, timed
from simple3d_gpu_kdtree import descriptors

K = 120
SEARCH = cp.RawKernel((HERE/'simple3d_step_index.cu').read_text(), 'search',
                      options=('--std=c++17', '--fmad=false'))


def indexing(x, p, iterations):
    # A separate implementation of Lloyd K-means, seeded once per cloud.
    center = x[cp.random.RandomState(42).choice(len(x), p, replace=False)].copy()
    labels = cp.empty(len(x), cp.int32)
    for it in range(iterations+1):
        ss = cp.sum(center*center, axis=1)
        for i in range(0, len(x), 32768):
            # Query's squared length is common to all candidate centers.
            dots = x[i:i+32768] @ center.T
            dots *= -2
            dots += ss
            labels[i:i+len(dots)] = cp.argmin(dots, axis=1)
        if it == iterations:
            break
        count = cp.bincount(labels, minlength=p)
        sums = cp.stack([cp.bincount(labels, weights=x[:,j].astype(cp.float64),
                                    minlength=p) for j in range(3)], axis=1)
        center = cp.where(count[:,None] > 0, sums/cp.maximum(count[:,None],1), center).astype(cp.float32)
    count = cp.bincount(labels, minlength=p)
    offsets = cp.concatenate((cp.zeros(1, dtype=cp.int32), cp.cumsum(count).astype(cp.int32)))
    d = x.astype(cp.float64)-center[labels].astype(cp.float64)
    radii = cp.zeros(p, cp.float64)
    cp.maximum.at(radii,labels,cp.sqrt(cp.sum(d*d, axis=1)))
    return center, radii, offsets, cp.argsort(labels).astype(cp.int32)


def step_search(x, index, p):
    out = cp.empty((len(x),K), cp.int32)
    SEARCH((len(x),), (128,), (x, *index, np.int32(p), np.int32(len(x)), out))
    return out


def run_operators(x,p,iterations,reverse=False):
    def base():
        tree, build = timed(lambda: KDTree(x))
        result, search = timed(lambda: tree.query(x,k=K,eps=0,p=2)[1].astype(cp.int32))
        return result,build,search
    def alternative():
        idx, build = timed(lambda: indexing(x,p,iterations))
        result, search = timed(lambda: step_search(x,idx,p))
        return result,build,search
    if reverse:
        s = alternative(); b = base()
    else:
        b = base(); s = alternative()
    return b,s


def squared(x, y):
    d=x-y
    return d[:,:,0]**2+d[:,:,1]**2+d[:,:,2]**2


def validate(x, a, b, points=None):
    assert bool(cp.all((b>=0)&(b<len(x)))), 'incomplete candidate output'
    differing = cp.flatnonzero(cp.any(a!=b,axis=1))
    set_rows=0; distance_rows=0; distance_max=0.
    for j in range(0,len(differing),8192):
        i=differing[j:j+8192]
        ai,bi=a[i],b[i]
        set_rows+=int(cp.count_nonzero(cp.any(cp.sort(ai,axis=1)!=cp.sort(bi,axis=1),axis=1)).item())
        q=x[i,None,:].astype(cp.float64)
        da=squared(q,x[ai].astype(cp.float64))
        db=squared(q,x[bi].astype(cp.float64))
        err=cp.abs(cp.sort(da,axis=1)-cp.sort(db,axis=1))
        distance_rows+=int(cp.count_nonzero(cp.any(err>1e-12,axis=1)).item())
        distance_max=max(distance_max,float(cp.max(err).item()))
    row=dict(order_diff=len(differing),set_diff=set_rows,
             distance_diff_over_1e_12=distance_rows,max_squared_distance_delta=distance_max)
    if points is not None:
        cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=10000,max_nn=10))
        n=cp.asarray(np.asarray(cloud.normals),dtype=cp.float32)
        for k in (40,80,120):
            err=cp.abs(descriptors(x,n,a,k)-descriptors(x,n,b,k))
            assert bool(cp.all(cp.isfinite(err)))
            row[f'fpfh{k}_diff_rows']=int(cp.count_nonzero(cp.any(err!=0,axis=1)).item())
            row[f'fpfh{k}_max_delta']=float(cp.max(err).item())
    return row


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tag',default='pilot')
    parser.add_argument('--clusters',type=int,default=512,choices=(64,128,256,512))
    parser.add_argument('--iterations',type=int,default=50)
    parser.add_argument('--samples',type=int,nargs='+',default=[0,8,50,90,129])
    cli=parser.parse_args()
    assert cli.tag.isalnum() and cli.iterations>=1
    cp.random.seed(42)
    torch.set_num_threads(4)
    # Exact comparison against exhaustive distances on a small independent cloud.
    x=cp.random.RandomState(81).random_sample((1024,3),dtype=cp.float32)
    ref=cp.argsort(squared(x[:,None].astype(cp.float64),x[None].astype(cp.float64)),axis=1)[:,:K]
    trial=step_search(x,indexing(x,64,5),64)
    assert bool(cp.all(ref==trial)), 'small exhaustive smoke mismatch'
    print('small exhaustive smoke passed: 1024 x 120',flush=True)
    args=SimpleNamespace(dataset='minishift',device='cuda:0',num_group=4096,group_size=128,
                         max_nn=40,use_LFSA=True,use_MSND=True,num_MSND=2,feature='FPFH',level='ALL',vis_save=False)
    dataset=PatchCore(args=args).get_dataloader('minishift','test','capsule','ALL').dataset
    expected=json.loads((BASE/'manifest.json').read_text())['test']
    assert expected==[dict(path=p,label=int(y)) for p,y in zip(dataset.pcd_paths,dataset.labels)]
    out=HERE/f'results/capsule_step_120nn_{cli.tag}_20261114'
    out.mkdir(parents=True,exist_ok=False)
    rows=[]
    with (out/'stages.csv').open('w',newline='',encoding='utf-8') as file:
        for s in cli.samples:
            xyz,_,_,path=dataset[s]
            assert path==expected[s]['path'] and xyz.shape==(500000,3)
            x=cp.asarray(xyz,dtype=cp.float32)
            # First full-size warm-up, not recorded. Separate from the smoke test.
            if not rows:
                run_operators(x,cli.clusters,min(2,cli.iterations))
            (a,bt,bq),(b,st,sq)=run_operators(x,cli.clusters,cli.iterations,bool(len(rows)%2))
            row=dict(sample=s,name=Path(path).name,label=expected[s]['label'],
                     kd_build_ms=bt,kd_query_ms=bq,kd_total_ms=bt+bq,
                     step_build_ms=st,step_query_ms=sq,step_total_ms=st+sq,
                     **validate(x,a,b,np.asarray(xyz)))
            if not rows:
                writer=csv.DictWriter(file,fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row);file.flush();rows.append(row)
            print(f'{len(rows)}/{len(cli.samples)} {json.dumps(row)}',flush=True)
            del a,b
    summary=dict(gpu=torch.cuda.get_device_name(), clusters=cli.clusters,
                 iterations=cli.iterations, samples=cli.samples)
    for k in rows[0]:
        if k.endswith('_ms'):
            summary[k]=statistics.mean(r[k] for r in rows)
        elif 'max' in k:
            summary[k]=max(r[k] for r in rows)
        elif 'diff' in k:
            summary[k]=sum(r[k] for r in rows)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
