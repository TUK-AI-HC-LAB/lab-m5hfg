"""Paired benchmark: original vs Morton-ordered 128-NN queries."""
import csv
import json
import statistics as stats
import time
from pathlib import Path

import cupy as cp
import torch
from cupyx.scipy.spatial import KDTree
from pointnet2_ops import pointnet2_utils
from simple3d_shapenet_hybrid_fps import install
from measure_simple3d_capsule_flashfps import PatchCore, SimpleNamespace
from fpfh_only_publish import replace_with_retry

HERE = Path(__file__).resolve().parent
BASE = HERE/'results/capsule_flashfps_fused_mean_20261113/fused/gpu_kdtree/minishift/capsule'
OUT = HERE/'results/capsule_flashfps_morton128_20261114'
REPORT = HERE.parent/'markdown/W47_Simple3D_capsule_FlashFPS_Morton_128NN.md'
REPEATS = 4


def dump(name,value):
    path=OUT/name
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    replace_with_retry(tmp,path)


def spread(n):
    n=(n | (n << 16)) & 0x030000ff
    n=(n | (n << 8)) & 0x0300f00f
    n=(n | (n << 4)) & 0x030c30c3
    return (n | (n << 2)) & 0x09249249


def order_for(q):
    lo=q.min(axis=0)
    span=q.max(axis=0)-lo
    xyz=((q-lo)/cp.maximum(span,cp.float32(1e-20))*1023).astype(cp.uint32)
    key=spread(xyz[:,0]) | (spread(xyz[:,1])<<1) | (spread(xyz[:,2])<<2)
    return cp.argsort(key)


def timed(fn):
    stream=cp.cuda.get_current_stream()
    stream.synchronize()
    t=time.perf_counter()
    out=fn()
    stream.synchronize()
    return out,(time.perf_counter()-t)*1000


def pair(tree,q,reverse):
    def original():
        return tree.query(q,k=128,eps=0,p=2)
    def ordered():
        t=order_for(q)
        d,i=tree.query(q[t],k=128,eps=0,p=2)
        dr,ir=cp.empty_like(d),cp.empty_like(i)
        dr[t],ir[t]=d,i
        return dr,ir
    if reverse:
        (db,ib),b=timed(ordered)
        (da,ia),a=timed(original)
    else:
        (da,ia),a=timed(original)
        (db,ib),b=timed(ordered)
    assert ia.shape==ib.shape==(4096,128)
    different=int(cp.count_nonzero(cp.any(ia!=ib,axis=1)).item())
    sets=int(cp.count_nonzero(cp.any(cp.sort(ia,axis=1)!=cp.sort(ib,axis=1),axis=1)).item()) if different else 0
    distance=float(cp.max(cp.abs(da-db)).item())
    return a,b,different,sets,distance


def main():
    if (OUT/'progress.json').exists():
        raise RuntimeError('Existing run preserved')
    OUT.mkdir(parents=True,exist_ok=True)
    dump('progress.json',dict(status='running',done=0,total=130))
    torch.set_num_threads(4)
    args=SimpleNamespace(dataset='minishift',device='cuda:0',num_group=4096,group_size=128,
                         max_nn=40,use_LFSA=True,use_MSND=True,num_MSND=2,feature='FPFH',level='ALL',vis_save=False)
    model=PatchCore(args=args)
    test=model.get_dataloader('minishift','test','capsule','ALL')
    expected=json.loads((BASE/'manifest.json').read_text())['test']
    actual=[dict(path=p,label=int(y)) for p,y in zip(test.dataset.pcd_paths,test.dataset.labels)]
    assert expected==actual and len(actual)==130
    sample,fps=install(pointnet2_utils.furthest_point_sample,'hybrid')
    start=time.perf_counter()
    records=[]
    csv_path=OUT/'queries.csv'
    try:
        with csv_path.open('w',newline='',encoding='utf-8') as file, torch.inference_mode():
            writer=None
            for i,(cloud,_,_,path) in enumerate(test):
                assert path[0]==expected[i]['path'] and cloud.shape==(1,500000,3)
                dense=cloud.to(device='cuda:0',dtype=torch.float32).contiguous()
                idx=sample(dense,4096).long()
                selected=dense[0,idx[0]].contiguous()
                torch.cuda.synchronize()
                tree=KDTree(cp.from_dlpack(dense[0]))
                q=cp.from_dlpack(selected)
                if i==0:
                    pair(tree,q,False)
                    pair(tree,q,True)
                for trial in range(REPEATS):
                    reverse=bool((i+trial)%2)
                    a,b,d,s,e=pair(tree,q,reverse)
                    row=dict(sample=i,trial=trial+1,first='morton' if reverse else 'original',
                             original_ms=a,morton_ms=b,delta_ms=b-a,different_rows=d,
                             different_sets=s,max_distance_delta=e)
                    if writer is None:
                        writer=csv.DictWriter(file,fieldnames=row)
                        writer.writeheader()
                    writer.writerow(row)
                    records.append(row)
                file.flush()
                if i%5==0 or i==129:
                    dump('progress.json',dict(status='running',done=i+1,total=130,elapsed_s=time.perf_counter()-start))
                del tree,q,idx,selected,dense
        assert len(records)==130*REPEATS
        def p95(values):
            values=sorted(values)
            return values[int((len(values)-1)*.95)]
        a,b,delta=[[r[k] for r in records] for k in ('original_ms','morton_ms','delta_ms')]
        wins=sum(x<0 for x in delta)
        differences=sum(r['different_rows'] for r in records)
        sets=sum(r['different_sets'] for r in records)
        error=max(r['max_distance_delta'] for r in records)
        stats_out=dict(status='verified',samples=130,repeats=REPEATS,gpu=torch.cuda.get_device_name(),
                       torch=torch.__version__,cupy=cp.__version__,fps=fps,
                       original_mean_ms=stats.mean(a),morton_mean_ms=stats.mean(b),
                       original_median_ms=stats.median(a),morton_median_ms=stats.median(b),
                       original_p95_ms=p95(a),morton_p95_ms=p95(b),
                       paired_median_delta_ms=stats.median(delta),morton_faster_pairs=wins,
                       different_rows=differences,different_sets=sets,max_distance_delta=error)
        dump('summary.json',stats_out)
        lines=['# W47 Simple3D 캡슐 — FlashFPS 대표점의 128-NN 공간순 조회','',
               '같은 FlashFPS 대표점 4,096개를 같은 GPU KD-tree에서 조회함. 유클리드 거리, 정확 조회(eps=0), k=128을 고정함.',
               'MiniShift 캡슐의 동일한 테스트 130개, 원본 50만 점/샘플. 샘플별 각 방식 4회, 먼저 실행하는 방식을 번갈아 바꿈.',
               '공간순 비용은 Morton 키 생성·정렬·조회·거리/인덱스 행 복원을 모두 포함함. KD-tree 구성과 FlashFPS는 공통 시점에 한 번 수행하고 계측 밖에 둠.',
               '조회 구간만 따로 실행함. FPFH, 특징 평균, 점수 보간, 전체 추론 및 AUROC는 이번에 재측정하지 않음.','',
               '| 지표 (ms) | 기존 순서 | 공간순 + 복원 |','|---|---:|---:|',
               f'| 평균 | {stats.mean(a):.3f} | {stats.mean(b):.3f} |',
               f'| 중앙값 | {stats.median(a):.3f} | {stats.median(b):.3f} |',
               f'| p95 | {p95(a):.3f} | {p95(b):.3f} |','',
               f'- 공간순이 더 빠른 비교: {wins}/{len(records)}회. 쌍별 차이(공간순−기존)의 중앙값: {stats.median(delta):+.3f}ms.',
               f'- 원래 대표점 순서로 복원한 뒤 비교한 {len(records)*4096:,}개 행에서 순서 불일치 {differences:,}개, 이웃 집합 불일치 {sets:,}개. 거리 최대 절대 차이 {error:.8g}.',
               '- [샘플별 계측](../source/results/'+OUT.name+'/queries.csv) · [검증 JSON](../source/results/'+OUT.name+'/summary.json) · [실행 코드](../source/measure_simple3d_capsule_morton128.py)']
        tmp=REPORT.with_suffix('.md.tmp')
        tmp.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        replace_with_retry(tmp,REPORT)
        dump('progress.json',dict(status='complete_verified',done=130,total=130,elapsed_s=time.perf_counter()-start))
    except Exception as exc:
        dump('progress.json',dict(status='failed',error=repr(exc),measured_pairs=len(records)))
        raise


if __name__=='__main__':
    main()
