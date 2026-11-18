"""Benchmark tie-safe Top-3 interpolation, keeping STEP at 512 clusters."""
import csv
import hashlib
import os
import statistics as st
import sys
import traceback
from pathlib import Path

import run_simple3d_capsule_flashfps as bench
import run_simple3d_capsule_step_flashfps as step

ROOT=bench.HERE/'results/capsule_interp_top3_20261117'
ASSETS=Path('/home/test/simple3d-capsule-interp-top3-20261117')
LEAF=Path('gpu_kdtree/minishift/capsule')
ORDERS=(('sort','top3'),('top3','sort'))


def select(dists, audit=None):
    import torch
    # A fourth candidate detects the only tie that can change set membership.
    # Also restore original ordering for ties inside the first three, since
    # addition in FP32 is order-sensitive. Compare after the original clamp.
    ds,idx=torch.topk(dists,4,dim=-1,largest=False,sorted=True)
    tied=(ds[...,:-1]==ds[...,1:]).any(dim=-1)
    values,ids=ds[...,:3].contiguous(),idx[...,:3].contiguous()
    if tied.any().item():
        td,ti=dists[tied].sort(dim=-1)
        values[tied],ids[tied]=td[:,:3],ti[:,:3]
    if audit is not None:
        expect,ref=dists.sort(dim=-1)
        audit.append((tied.numel(),int(tied.sum().item()),
                      int((ids!=ref[...,:3]).any(dim=-1).sum().item()),
                      int((values!=expect[...,:3]).any(dim=-1).sum().item())))
    return values,ids


def top3(xyz1,xyz2,points2,chunk_size=50000,audit=None):
    import torch
    from feature_extractors.pointnet2_utils import square_distance,index_points,interpolating_points_chunked
    B,_,N=xyz1.shape
    if xyz2.shape[-1]<4:
        return interpolating_points_chunked(xyz1,xyz2,points2,chunk_size)
    target,source,features=xyz1.permute(0,2,1),xyz2.permute(0,2,1),points2.permute(0,2,1)
    chunks=[]
    for start in range(0,N,chunk_size):
        distance=square_distance(target[:,start:start+chunk_size],source)
        ds,idx=select(distance,audit)
        reciprocal=1.0/(ds+1e-8)
        weight=reciprocal/torch.sum(reciprocal,dim=2,keepdim=True)
        chunks.append(torch.sum(index_points(features,idx)*weight.view(B,-1,3,1),dim=2).permute(0,2,1))
    return torch.cat(chunks,dim=2)


def preflight():
    import torch
    import measure_simple3d_capsule_flashfps as w
    from feature_extractors.pointnet2_utils import interpolating_points_chunked as official
    from simple3d_shapenet_hybrid_fps import install
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    from pointnet2_ops import pointnet2_utils
    sample,_=install(pointnet2_utils.furthest_point_sample,'hybrid')
    g=torch.Generator(device='cuda:0').manual_seed(42)
    args=w.SimpleNamespace(dataset='minishift',device='cuda:0',num_group=4096,group_size=128,
        max_nn=40,use_LFSA=True,use_MSND=True,num_MSND=2,feature='FPFH',level='ALL',vis_save=False)
    ds=w.PatchCore(args=args).get_dataloader('minishift','test','capsule','ALL').dataset
    output=[]
    def measure(fn):
        import time
        torch.cuda.synchronize()
        now=time.perf_counter()
        a=fn()
        torch.cuda.synchronize()
        return a,1000*(time.perf_counter()-now)
    def test(name,xyz,c,v):
        audit=[]
        b=top3(xyz,c,v,audit=audit)
        a=official(xyz,c,v)
        tmp=[]
        for i in range(4):
            ref=lambda:official(xyz,c,v)
            new=lambda:top3(xyz,c,v)
            if i%2:
                _,tm_new=measure(new); _,tm_ref=measure(ref)
            else:
                _,tm_ref=measure(ref); _,tm_new=measure(new)
            tmp.append((tm_ref,tm_new))
        r=dict(sample=name,centers=c.shape[-1],queries=xyz.shape[-1],
            flagged=sum(q[1] for q in audit),id_mismatch=sum(q[2] for q in audit),
            distance_mismatch=sum(q[3] for q in audit),score_mismatch=int((a!=b).sum().item()),
            max_abs=float((a-b).abs().max().item()),sort_ms=st.median(t[0] for t in tmp),
            top3_ms=st.median(t[1] for t in tmp))
        output.append(r)
        bench.writecsv(ROOT/'preflight.csv',output)
        print(r,flush=True)
        assert r['id_mismatch']==r['distance_mismatch']==r['score_mismatch']==0, r
    with torch.inference_mode():
        # Distances 1, 4, 9, 9: tie only between 3rd and 4th, scores differ.
        x=torch.zeros((1,3,8),device='cuda:0')
        c=torch.zeros((1,3,8),device='cuda:0')
        c[0,0]=torch.tensor([1.,2.,3.,-3.,4.,5.,6.,7.],device='cuda:0')
        test('synthetic_3rd_4th_tie',x,c,torch.arange(8,device='cuda:0',dtype=torch.float32)[None,None])
        c=torch.zeros((1,3,4096),device='cuda:0')
        test('synthetic_all_ties',x,c,torch.arange(4096,device='cuda:0',dtype=torch.float32)[None,None])
        for i in (0,50,129):
            pc,_,_,_=ds[i]
            xyz=torch.as_tensor(pc,device='cuda:0',dtype=torch.float32)[None].contiguous()
            assert xyz.shape==(1,500000,3)
            for count in (4096,1024):
                centers=xyz[:,sample(xyz,count)[0].long()]
                test(str(i),xyz.transpose(1,2),centers.transpose(1,2),
                     torch.rand((1,1,count),generator=g,device='cuda:0'))


def worker(pair,variant):
    import measure_simple3d_capsule_flashfps as w
    step.ROOT,step.ASSETS=ROOT/f'pair{pair}'/variant,ASSETS/f'pair{pair}'/variant
    # Only replace the interpolation lookup used by score computation. Let
    # the existing instrumentation wrap it exactly as for the baseline.
    if variant=='top3':
        w.features_module.interpolating_points_chunked=top3
    step.worker_main('step')  # 512 clusters in both arms


def csv_rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def compare(pair):
    import numpy as np
    import torch
    p=[ROOT/f'pair{pair}'/v/'step'/LEAF for v in ('sort','top3')]
    a,b=[bench.load(x/'summary.json') for x in p]
    for f in p:
        assert bench.load(f/'manifest.json')==bench.load(bench.BASE/'manifest.json')
    assert bench.load(p[0]/'environment.json')==bench.load(p[1]/'environment.json')
    a1,b1=[Path(s['raw_assets']) for s in (a,b)]
    bank_equal=torch.equal(*(torch.load(x/'memory_bank.pt',map_location='cpu',weights_only=True) for x in (a1,b1)))
    detail=[]
    for x,y in zip(*(csv_rows(f/'test_timings.csv') for f in p),strict=True):
        assert all(x[k]==y[k] for k in ('index','path','points','label'))
        i=int(x['index'])
        with np.load(a1/f'{i:05d}.npz') as z,np.load(b1/f'{i:05d}.npz') as t:
            assert np.array_equal(z['labels'],t['labels'])
            u,v=z['scores'],t['scores']
            assert u.shape==v.shape==z['labels'].shape
            d=np.abs(u-v)
        detail.append(dict(index=i,points=u.size,changed=int(np.count_nonzero(d)),max_abs=float(d.max()),
            object_delta=abs(float(x['object_score'])-float(y['object_score'])),
            delta_ms=1000*(float(y['inference_s'])-float(x['inference_s']))))
    assert len(detail)==130
    bench.writecsv(ROOT/f'pair{pair}'/'paired_samples.csv',detail)
    timings={}
    for count in (4096,1024):
        timings[count]=[]
        for f in p:
            t=[r for r in csv_rows(f/'stage_summary.csv') if r['phase']=='test' and
               r['stage']=='point_score_interpolation' and f'[1, 3, {count}]' in r['detail']]
            assert sum(int(r['calls']) for r in t)==130
            timings[count].append(1000*sum(float(r['total_s']) for r in t)/130)
    gpu=[next(r['whole_gpu_used_mib_max'] for r in bench.load(f/'resource_summary.json') if r['phase']=='test') for f in p]
    return dict(pair=pair,order=ORDERS[pair-1],sort=a,top3=b,interp=timings,gpu_peak=gpu,bank_equal=bank_equal,
        metrics_equal=a['metrics']==b['metrics'],point_changes=sum(r['changed'] for r in detail),
        points=sum(r['points'] for r in detail),max_point_delta=max(r['max_abs'] for r in detail),
        max_object_delta=max(r['object_delta'] for r in detail))


def report(results):
    lines=['# W47 Simple3D 캡슐 — 전체 정렬 vs 동점 보존 Top-3 보간','',
        'STEP 512군집, FlashFPS 및 나머지 설정은 유지. 두 군의 보간은 같은 거리·가중치 수식을 사용함. Top-3판은 일반 행은 가장 가까운 3개만 사용하고, 네 번째 후보까지 확인해 동점이 있으면 그 행만 기존 정렬로 되돌림.',
        '같은 RTX 5080에서 전체 정렬→Top-3, Top-3→전체 정렬로 실행. 매번 정상 54개 새 뱅크, 동일 테스트 130개. 아래 추론 시간은 특징 추출+점수 계산으로 입출력·평가 시간은 제외함.',
        '', '| 쌍 | 비교 (ms/샘플) | 전체 정렬 | Top-3 | 차이 (Top-3−전체 정렬) |','|---|---|---:|---:|---:|']
    for r in results:
        a,b=r['sort'],r['top3']; q=r['interp']
        for n,u,v in [('집계 점수 보간',*q[4096]),('평활화 점수 보간',*q[1024]),
                      ('두 보간 합계',q[4096][0]+q[1024][0],q[4096][1]+q[1024][1]),
                      ('전체 평균 추론',a['inference_mean_ms'],b['inference_mean_ms']),
                      ('전체 p95',a['inference_p95_ms'],b['inference_p95_ms'])]:
            lines.append(f"| {r['pair']} | {n} | {u:.3f} | {v:.3f} | {v-u:+.3f} |")
    lines+=['','## 점수와 성능','']
    for r in results:
        lines.append(f"- 쌍 {r['pair']}: 뱅크 일치 {r['bank_equal']}, 난이도별 AUROC/AP 일치 {r['metrics_equal']}, "
            f"점수 다른 점 {r['point_changes']:,}/{r['points']:,}, 물체 점수 최대 오차 {r['max_object_delta']:.8g}.")
        for v in ('sort','top3'):
            m=next(m for m in r[v]['metrics'] if m['level']=='ALL')
            lines.append(f"  - {v}: 물체 AUROC {100*m['object_auroc']:.6f}%, 포인트 AUROC {100*m['point_auroc']:.6f}%, 포인트 AP {100*m['point_ap']:.6f}%.")
    lines+=['', f'- [동점·실제 좌표 보간 예비 검사](../source/results/{ROOT.name}/preflight.csv) · [전체 결과](../source/results/{ROOT.name}/verification.json) · [실행 코드](../source/run_simple3d_capsule_interp_top3.py)',
        '- 1개 장비·1개 카테고리, 2쌍 실행임. 다른 점군과 동점이 많은 센서 데이터에서 같은 속도를 보장하지 않음.']
    report=bench.HERE.parent/'markdown/W47_Simple3D_capsule_interpolation_top3.md'
    tmp=report.with_suffix('.md.tmp')
    tmp.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    bench.replace_with_retry(tmp,report)


def main():
    if '--mode' in sys.argv:
        worker(int(os.environ['SIMPLE3D_INTERP_PAIR']),os.environ['SIMPLE3D_INTERP_VARIANT'])
        return
    if (ROOT/'progress.json').exists():
        raise RuntimeError('Existing run preserved')
    original=bench.state
    state={'completed_jobs':0,'total_jobs':4}
    def update(**kw):
        state.update(kw)
        prior=bench.ROOT
        bench.ROOT=ROOT
        try: original(**state)
        finally: bench.ROOT=prior
    bench.state=update
    sources=[Path(__file__),*[bench.HERE/n for n in ('measure_simple3d_capsule_flashfps.py',
        'simple3d_step_index.cu','simple3d_gpu_kdtree.py','simple3d_gpu_kdtree.cu',
        'run_simple3d_capsule_step_flashfps.py','simple3d_shapenet_hybrid_fps.py')]]
    def hashes():
        return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    try:
        snapshot=hashes()
        update(status='preflight')
        bench.atomic(ROOT/'plan.json',dict(orders=ORDERS,source_sha256=snapshot))
        preflight()
        results=[]
        for i,order in enumerate(ORDERS,1):
            for v in order:
                os.environ.update(SIMPLE3D_INTERP_PAIR=str(i),SIMPLE3D_INTERP_VARIANT=v)
                bench.ROOT=ROOT/f'pair{i}'/v/'step'
                update(status='starting',pair=i,variant=v)
                s=bench.run(bench.ITEM,worker=Path(__file__).resolve())
                assert (s['train_count'],s['test_count'])==(54,130)
                update(completed_jobs=state['completed_jobs']+1)
            results.append(compare(i))
            assert snapshot==hashes()
            bench.atomic(ROOT/'verification.json',dict(pairs=results,source_sha256=snapshot))
            report(results)
        update(status='complete_verified',phase='complete')
    except BaseException:
        update(status='failed',error=traceback.format_exc())
        raise


if __name__=='__main__':
    main()
