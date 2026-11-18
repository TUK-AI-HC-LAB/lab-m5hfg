"""Fresh end-to-end pairs: same FlashFPS/STEP512/selective Top3, bounds sort only."""
import os
import sys
import traceback
from pathlib import Path
import run_simple3d_capsule_flashfps as bench
import run_simple3d_capsule_step_flashfps as step_run
import run_simple3d_capsule_interp_top3 as interpolation
from run_simple3d_all_step_flashfps import baseline

ROOT=bench.HERE/'results/step_radix_e2e_20261118'
ASSETS=Path('/home/test/simple3d-step-radix-e2e-20261118')
REPORT=bench.HERE.parent/'markdown/W48_Simple3D_STEP_radix_end_to_end.md'
TARGETS=(('minishift','capsule',54,130,('baseline','radix')),
         ('shapenet','vase9',4,37,('radix','baseline')))


def worker():
    import measure_simple3d_capsule_flashfps as w
    from pilot_simple3d_step_radix import kernel
    from pilot_simple3d_step_no_boundsort import search
    import measure_simple3d_capsule_step as operators
    d=sys.argv[sys.argv.index('--dataset')+1];c=sys.argv[sys.argv.index('--category')+1]
    v=os.environ['SIMPLE3D_RADIX_VARIANT']
    assert v in ('baseline','radix')
    step_run.ROOT=ROOT/c/v
    step_run.ASSETS=ASSETS/c/v
    original=w.features_module.interpolating_points_chunked
    routes=[]
    def selective(xyz1,xyz2,points2,chunk_size=50000):
        count=int(xyz2.shape[-1]);assert count in (4096,1024)
        routes.append(dict(**w.CONTEXT,source_centers=count,method='top3' if count==4096 else 'sort'))
        return (interpolation.top3 if count==4096 else original)(xyz1,xyz2,points2,chunk_size)
    w.features_module.interpolating_points_chunked=selective
    fn=None
    if v=='radix':
        fn=kernel()
        operators.step_search=lambda x,index,p:search(fn,x,index,p)
    step_run.worker_main('step')
    folder=step_run.ROOT/'step/gpu_kdtree'/d/c
    bench.writecsv(folder/'interpolation_paths.csv',routes)
    bench.atomic(folder/'variant.json',dict(variant=v,only_change='512 bound sorting: bitonic -> stable CUB radix',
        fixed='FlashFPS hybrid; STEP512/Lloyd50; selective Top3 at4096; full-sort1024; aggregation128; smoothing12',
        kernel_attributes=fn.attributes if fn is not None else None))


def compare(d,c,ntrain,ntest):
    import numpy as np
    import torch
    folders=[ROOT/c/v/'step/gpu_kdtree'/d/c for v in ('baseline','radix')]
    expected=bench.load(baseline(d,c)/'manifest.json')
    assert all(bench.load(f/'manifest.json')==expected for f in folders)
    assert bench.load(folders[0]/'environment.json')==bench.load(folders[1]/'environment.json')
    summaries=[bench.load(f/'summary.json') for f in folders]
    assert all((s['train_count'],s['test_count'])==(ntrain,ntest) for s in summaries)
    assets=[Path(s['raw_assets']) for s in summaries]
    bank_equal=torch.equal(*(torch.load(p/'memory_bank.pt',map_location='cpu',weights_only=True) for p in assets))
    samples=[]
    for x,y in zip(*(interpolation.csv_rows(f/'test_timings.csv') for f in folders),strict=True):
        assert all(x[k]==y[k] for k in ('index','path','points','label'))
        i=int(x['index'])
        with np.load(assets[0]/f'{i:05d}.npz') as a,np.load(assets[1]/f'{i:05d}.npz') as b:
            assert np.array_equal(a['labels'],b['labels'])
            assert a['scores'].shape==b['scores'].shape
            delta=np.abs(a['scores']-b['scores'])
        samples.append(dict(index=i,path=x['path'],points=delta.size,changed=int(np.count_nonzero(delta)),
            max_abs=float(delta.max()),object_delta=abs(float(x['object_score'])-float(y['object_score'])),
            delta_inference_ms=1000*(float(y['inference_s'])-float(x['inference_s']))))
    assert len(samples)==ntest
    bench.writecsv(ROOT/c/'paired_samples.csv',samples)
    return dict(dataset=d,category=c,baseline=summaries[0],radix=summaries[1],bank_equal=bank_equal,
        metrics_equal=summaries[0]['metrics']==summaries[1]['metrics'],points=sum(s['points'] for s in samples),
        point_changes=sum(s['changed'] for s in samples),max_point_delta=max(s['max_abs'] for s in samples),
        max_object_delta=max(s['object_delta'] for s in samples))


def publish(results):
    rows=[];stages=[];resources=[];metrics=[]
    lines=['# W48 Simple3D — STEP 군집 하한 정렬의 Radix 교체 검증','',
        f'완료한 비교: {len(results)}/2개 카테고리. 각 버전·카테고리 1회.','',
        '## 질문과 사전 가설','',
        '- 질문: 군집 방문 순서는 그대로 두고 512개 하한 정렬만 CUB stable radix로 교체하면 전체 추론이 빨라지는가?',
        '- 가설: 이웃 순서·뱅크·이상 점수가 같고 조회 시간이 감소함. 전체 시간에서는 감소가 작거나 다른 단계 변동에 가려질 수 있음.',
        '- 고정: FlashFPS hybrid, STEP512/Lloyd50, 4,096점 보간 Top-3, 1,024점 보간 전체 정렬, 집계128, 평활화12, seed0.',
        '- 변경: 독립 STEP 계열 구현의 군집 하한 정렬만 bitonic에서 NVIDIA CUB BlockRadixSort로 변경. 후보 병합 정렬 유지. 새 알고리즘의 최초 제안이라고 주장하지 않음.',
        '- 두 버전 모두 정상 데이터로 뱅크를 새로 구성함. capsule은 기존→Radix, vase9는 Radix→기존 순서. 입력/저장/지표 집계는 추론 시간 제외.',
        '- Δ = Radix−기존. ms/샘플, 음수가 빠름. 카테고리별 1회이므로 반복 분산이나 통계적 유의성은 미확인.','',
        '| 카테고리 | 평균 기존 (ms) | 평균 Radix (ms) | Δ (ms) | Δ (%) | p95 기존 (ms) | p95 Radix (ms) | 점수 불일치/전체 | 뱅크·지표 일치 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in results:
        d,c=r['dataset'],r['category'];a,b=r['baseline'],r['radix']
        row=dict(dataset=d,category=c,baseline_ms=a['inference_mean_ms'],radix_ms=b['inference_mean_ms'],
            delta_ms=b['inference_mean_ms']-a['inference_mean_ms'],delta_percent=100*(b['inference_mean_ms']/a['inference_mean_ms']-1),
            baseline_p95_ms=a['inference_p95_ms'],radix_p95_ms=b['inference_p95_ms'],
            point_changes=r['point_changes'],points=r['points'],bank_equal=r['bank_equal'],metrics_equal=r['metrics_equal'])
        rows.append(row)
        lines.append(f"| {c} | {row['baseline_ms']:.3f} | {row['radix_ms']:.3f} | {row['delta_ms']:+.3f} | {row['delta_percent']:+.2f} | {row['baseline_p95_ms']:.3f} | {row['radix_p95_ms']:.3f} | {row['point_changes']:,}/{row['points']:,} | {row['bank_equal']}/{row['metrics_equal']} |")
        for v in ('baseline','radix'):
            f=ROOT/c/v/'step/gpu_kdtree'/d/c
            stages.extend(dict(dataset=d,category=c,variant=v,**x) for x in interpolation.csv_rows(f/'stage_summary.csv'))
            resources.extend(dict(dataset=d,category=c,variant=v,**x) for x in bench.load(f/'resource_summary.json'))
            metrics.extend(dict(variant=v,**{k:val for k,val in x.items() if k!='mode'}) for x in r[v]['metrics'])
    for name,data in [('category_comparison',rows),('stage_comparison',stages),('resource_comparison',resources),('metrics_comparison',metrics)]:bench.writecsv(ROOT/(name+'.csv'),data)
    lines+=['','## 原본 근거'.replace('原본','원본'),'',
        '- 기준 commit: `66956cf11839902944dffc5082210cb859f9b5f8`. 새 파일은 미커밋이며 plan.json에 소스 해시 기록.',
        '- [실행 스크립트](../source/run_simple3d_step_radix_e2e.sh) · [코드](../source/run_simple3d_step_radix_e2e.py)',
        f'- [진행 상태](../source/results/{ROOT.name}/progress.json) · [계획·소스 해시](../source/results/{ROOT.name}/plan.json)',
        f'- [비교 CSV](../source/results/{ROOT.name}/category_comparison.csv) · [전체 단계별 시간](../source/results/{ROOT.name}/stage_comparison.csv)',
        f'- [전체·난이도별 AUROC/AP](../source/results/{ROOT.name}/metrics_comparison.csv) · [자원 계측](../source/results/{ROOT.name}/resource_comparison.csv)',
        f'- [검증 JSON](../source/results/{ROOT.name}/verification.json)',
        '- 각 물체 폴더의 paired_samples.csv에 샘플별 점수·시간 차이를 저장함. raw 뱅크와 점수는 WSL에만 저장.',
        '- 단계별 시간은 중첩되어 단순 합산하지 않음. GPU 전체 사용량에는 다른 프로세스 포함, Torch 할당에는 CuPy 제외.',
        '- 순서: 병목 집계 → 정렬 제거 후보 실패 → 작업량 진단 → 정렬 유지 Radix 연산 시험 → 현재 전체 파이프라인 검증. 이후 규모 변화·반복 측정·다른 모델 검증은 아직 미실시.']
    tmp=REPORT.with_suffix('.md.tmp');tmp.write_text('\n'.join(lines)+'\n',encoding='utf-8');bench.replace_with_retry(tmp,REPORT)


def main():
    if '--mode' in sys.argv:worker();return
    if (ROOT/'progress.json').exists():raise RuntimeError('Existing results preserved')
    sources=[Path(__file__),*[bench.HERE/n for n in ('pilot_simple3d_step_radix.py','pilot_simple3d_step_no_boundsort.py',
        'simple3d_step_index.cu','measure_simple3d_capsule_step.py','run_simple3d_capsule_step_flashfps.py',
        'run_simple3d_capsule_interp_top3.py','measure_simple3d_capsule_flashfps.py','simple3d_gpu_kdtree.py',
        'simple3d_gpu_kdtree.cu','simple3d_shapenet_hybrid_fps.py','run_simple3d_step_radix_e2e.sh')]]
    def hashes():return {str(p):bench.sha(p) for p in sources}
    snapshot=hashes()
    state=dict(completed_jobs=0,total_jobs=4,completed_pairs=0,total_pairs=2,pid=os.getpid())
    original_state=bench.state
    def update(**kw):
        state.update(kw);old=bench.ROOT;bench.ROOT=ROOT
        try:original_state(**state)
        finally:bench.ROOT=old
    bench.state=update
    update(status='starting')
    bench.atomic(ROOT/'plan.json',dict(targets=TARGETS,source_sha256=snapshot,
        only_change='STEP512 cluster bound sorting bitonic -> stable CUB radix',fresh_banks=True,assets=str(ASSETS)))
    results=[];publish(results)
    try:
        for d,c,ntrain,ntest,order in TARGETS:
            for v in order:
                assert hashes()==snapshot,'Sources changed'
                os.environ['SIMPLE3D_RADIX_VARIANT']=v
                bench.ROOT=ROOT/c/v/'step'
                update(status='starting',dataset=d,category=c,variant=v,order=order)
                s=bench.run(dict(mode='gpu_kdtree',dataset=d,category=c),worker=Path(__file__).resolve())
                assert (s['train_count'],s['test_count'])==(ntrain,ntest)
                route=interpolation.csv_rows(bench.ROOT/'gpu_kdtree'/d/c/'interpolation_paths.csv')
                route=[x for x in route if x['phase']=='test'];assert len(route)==2*ntest
                for i in range(ntest):assert {(x['source_centers'],x['method']) for x in route if int(x['index'])==i}=={('4096','top3'),('1024','sort')}
                update(completed_jobs=state['completed_jobs']+1)
            r=compare(d,c,ntrain,ntest);results.append(r)
            assert hashes()==snapshot
            bench.atomic(ROOT/'verification.json',dict(pairs=results,source_sha256=snapshot))
            publish(results);update(completed_pairs=len(results))
            assert r['bank_equal'] and r['metrics_equal'] and r['point_changes']==0 and r['max_object_delta']==0,'Exact output changed: inspect before expansion'
        update(status='complete_verified',phase='complete')
    except BaseException:
        update(status='failed',error=traceback.format_exc());raise


if __name__=='__main__':main()
