"""Run the capsule STEP + FlashFPS recipe across MiniShift and Anomaly-ShapeNet."""
import csv
import hashlib
import math
import sys
import traceback
from pathlib import Path

import run_simple3d_capsule_flashfps as bench
import run_simple3d_capsule_step_flashfps as capsule

ROOT = bench.HERE/'results/all_step_flashfps_20261114_r2'
ASSETS = Path('/home/test/simple3d-all-step-flashfps-20261114-r2')
REPORT = bench.HERE.parent/'markdown/W47_Simple3D_all_STEP_FlashFPS_comparison.md'
HISTORY = {'minishift': 'minishift_hybrid_fps_20261112',
           'shapenet': 'shapenet_hybrid_fps_20261111'}
TOTAL = 52


def baseline(dataset, category):
    return ROOT.parent/HISTORY[dataset]/'hybrid/gpu_kdtree'/dataset/category


def target(dataset, category):
    return ROOT/'step/gpu_kdtree'/dataset/category


def records(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def validate(old, new):
    a,b = [bench.load(f/'summary.json') for f in (old,new)]
    assert a['status']==b['status']=='complete'
    assert (a['train_count'],a['test_count'])==(b['train_count'],b['test_count'])
    assert bench.load(old/'manifest.json')==bench.load(new/'manifest.json')
    assert bench.load(old/'environment.json')==bench.load(new/'environment.json')
    assert len(records(new/'train_timings.csv'))==b['train_count']
    assert len(records(new/'test_timings.csv'))==b['test_count']
    raw=Path(b['raw_assets'])
    assert (raw/'memory_bank.pt').is_file() and len(list(raw.glob('*.npz')))==b['test_count']
    resource=next(r for r in bench.load(new/'resource_summary.json') if r['phase']=='test')
    assert resource['samples'] > 0
    for v in (a,b):
        for r in v['metrics']:
            assert all(math.isfinite(r[k]) and 0<=r[k]<=1 for k in ('object_auroc','point_auroc','point_ap'))
    for phase,n in (('fit',b['train_count']),('test',b['test_count'])):
        s=[r for r in records(new/'stage_summary.csv') if r['phase']==phase]
        assert all(sum(int(r['calls']) for r in s if r['stage']==k)==n for k in (
            'step_cluster_index_build','step_query_k120','gpu_kdtree_all_points_k120'))
        for name,detail in [('fps','output_count=4096'),('coordinate_knn_gpu_kdtree','k=128;')]+(
                [('fps','output_count=1024'),('coordinate_knn_gpu_kdtree','k=12;')] if phase=='test' else []):
            assert sum(int(r['calls']) for r in s if r['stage']==name and detail in r['detail'])==n
    return a,b


def publish(completed):
    import statistics as st
    detail=[]; stage_rows=[]; resources=[]; metrics=[]
    for dataset,category in completed:
        folders=baseline(dataset,category),target(dataset,category)
        pair=[bench.load(f/'summary.json') for f in folders]
        row=dict(dataset=dataset,category=category,train=pair[1]['train_count'],test=pair[1]['test_count'])
        for v,summary,folder in zip(('baseline','step'),pair,folders,strict=True):
            stages=records(folder/'stage_summary.csv')
            raw=Path(summary['raw_assets'])
            overall=next(m for m in summary['metrics'] if m['level']=='ALL')
            tests=summary['test_count']
            resource=next(r for r in bench.load(folder/'resource_summary.json') if r['phase']=='test')
            data={k:overall[k]*100 for k in ('object_auroc','point_auroc','point_ap')}
            data.update(inference_ms=summary['inference_mean_ms'],p95_ms=summary['inference_p95_ms'],
                        fit_s=summary['fit_wall_s'], torch_gib=summary['peak_torch_allocated_bytes']/2**30,
                        gpu_peak_mib=resource['whole_gpu_used_mib_max'],gpu_util_pct=resource['gpu_util_percent_mean'],
                        fpfh120nn_ms=sum(1000*float(r['total_s'])/tests for r in stages
                                        if r['phase']=='test' and r['stage']=='gpu_kdtree_all_points_k120'),
                        saved_mib=sum(f.stat().st_size for f in raw.iterdir() if f.is_file())/2**20
                            if raw.is_dir() else None)
            for k,value in data.items(): row[v+'_'+k]=value
            stage_rows.extend(dict(dataset=dataset,category=category,variant=v,**r) for r in stages)
            for r in records(folder/'resource_summary.csv'):
                resources.append(dict(dataset=dataset,category=category,variant=v,**r))
            for r in summary['metrics']:
                metrics.append(dict(variant=v,**r))
        for k in data:
            a,b=row['baseline_'+k],row['step_'+k]
            row['delta_'+k]=b-a if a is not None and b is not None else None
        detail.append(row)
    for name,table in [('category_comparison',detail),('stage_comparison',stage_rows),
                       ('resource_comparison',resources),('metrics_comparison',metrics)]:
        bench.writecsv(ROOT/(name+'.csv'),table)
    lines=['# W47 Simple3D — 전체 카테고리 FlashFPS + STEP 비교','',
           f'**검증 완료: {len(detail)}/{TOTAL}개 카테고리.**',
           'MiniShift 12개와 Anomaly-ShapeNet 40개. FlashFPS 및 집계/평활화 KD-tree는 유지하고 FPFH용 120-NN만 STEP 기반으로 교체함. 카테고리마다 정상 뱅크를 새로 구성함.',
           '기준판은 같은 입력·모델 소스·설정으로 측정한 기존 FlashFPS 결과임. 재실행한 동시 비교가 아니므로 시간·장치 자원 차이에는 실행 시점의 영향이 있을 수 있음. 추론 시간은 입력·저장·평가를 제외함.',
           '', '| 데이터셋 | 카테고리 | 평균 추론 (ms, 기존 → STEP) | FPFH 120-NN (ms) | 물체 AUROC (%) | 포인트 AUROC (%) | 포인트 AP (%) |',
           '|---|---:|---:|---:|---:|---:|---:|']
    def cell(row,k):
        a,b=row['baseline_'+k],row['step_'+k]
        return f'{a:.2f} → {b:.2f} ({b-a:+.2f})'
    keys=('inference_ms','fpfh120nn_ms','object_auroc','point_auroc','point_ap')
    for dataset in HISTORY:
        group=[r for r in detail if r['dataset']==dataset]
        if not group: continue
        avg={v+'_'+k:st.mean(r[v+'_'+k] for r in group) for v in ('baseline','step') for k in keys}
        lines.append(f'| {dataset} | 평균 ({len(group)}개) | '+ ' | '.join(cell(avg,k) for k in keys)+' |')
        for r in group:
            lines.append(f"| {dataset} | {r['category']} | "+' | '.join(cell(r,k) for k in keys)+' |')
    lines.extend(['','차이는 STEP − 기존. 위 평균은 완료 카테고리의 동일 가중 평균. 지표 차이는 %p. GPU 전체 VRAM 표본은 다른 프로세스도 포함함.',''])
    for file,title in [('category_comparison','카테고리별 세부 비교'),('stage_comparison','단계별 시간'),('resource_comparison','자원 사용량'),('metrics_comparison','성능 (난이도별 포함)')]:
        lines.append(f'- [{title}](../source/results/{ROOT.name}/{file}.csv)')
    temp=REPORT.with_suffix('.md.tmp')
    temp.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    bench.replace_with_retry(temp,REPORT)


def main():
    if '--mode' in sys.argv:
        capsule.ROOT, capsule.ASSETS = ROOT, ASSETS
        capsule.worker_main('step')
        return
    if (ROOT/'progress.json').exists():
        raise RuntimeError('Existing run preserved')
    state={}; completed=[]; failed=[]
    original_state=bench.state
    def update(**kwargs):
        state.update(kwargs)
        prior=bench.ROOT
        bench.ROOT=ROOT
        try: original_state(**state)
        finally: bench.ROOT=prior
    bench.state=update
    planned=[]
    for dataset,history in HISTORY.items():
        names=bench.load(ROOT.parent/history/'plan.json')['categories']
        assert len(set(names))==len(names)==(12 if dataset=='minishift' else 40)
        planned.extend((dataset,cat) for cat in names)
    assert len(planned)==TOTAL
    sources=('simple3d_gpu_kdtree.py','simple3d_gpu_kdtree.cu','simple3d_step_index.cu',
             'measure_simple3d_capsule_step.py','measure_simple3d_capsule_flashfps.py',
             'simple3d_shapenet_hybrid_fps.py','run_simple3d_capsule_flashfps.py',
             'run_simple3d_capsule_step_flashfps.py',Path(__file__).name)
    initial={name:hashlib.sha256((bench.HERE/name).read_bytes()).hexdigest() for name in sources}
    try:
        update(status='checking',completed_jobs=0,total_jobs=TOTAL)
        for dataset,category in planned:
            assert bench.load(baseline(dataset,category)/'summary.json')['status']=='complete'
        bench.atomic(ROOT/'plan.json',dict(categories=planned,source_sha256=initial))
        bench.ROOT=ROOT/'step'
        for d,c in (('shapenet','ashtray0'),('minishift','capsule')):
            bench.run(dict(mode='gpu_kdtree',dataset=d,category=c),smoke=True,worker=Path(__file__).resolve())
        for d,c in sorted(planned, key=lambda job: job[0]!='shapenet'):
            try:
                bench.run(dict(mode='gpu_kdtree',dataset=d,category=c),worker=Path(__file__).resolve())
                validate(baseline(d,c),target(d,c))
            except Exception:
                failed.append(dict(dataset=d,category=c,error=traceback.format_exc()))
                bench.atomic(ROOT/'failures.json',failed)
            else:
                completed.append((d,c))
                bench.atomic(ROOT/'completed.json',completed)
                publish(completed)
            update(completed_jobs=len(completed),failed_jobs=len(failed))
        assert initial=={n:hashlib.sha256((bench.HERE/n).read_bytes()).hexdigest() for n in sources}
        update(status='complete_verified' if not failed else 'partial_failures',phase='complete')
    except BaseException:
        update(status='failed',error=traceback.format_exc())
        raise


if __name__=='__main__':
    main()
