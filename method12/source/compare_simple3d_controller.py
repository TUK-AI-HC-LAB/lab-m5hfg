"""Detached sequential 104-job comparison, external resources and verified reports."""
import ast
import csv
import datetime
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import psutil

HERE=Path(__file__).resolve().parent
ROOT=HERE/'results/official_vs_gpu_kdtree_20261104'
REPO=Path('/mnt/c/Users/test/Desktop/Codex/Project/Simple3D-official-compatible-20261104')
REPORT=HERE.parent/'markdown/Simple3D_official_vs_GPU_KDTree_measurement_20261104.md'
STATE={}


def atomic(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2));temp.replace(path)


def writecsv(path,rows):
    if not rows:return
    temp=path.with_suffix('.csv.tmp')
    with temp.open('w',newline='') as file:
        w=csv.DictWriter(file,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    temp.replace(path)


def state(**v):
    STATE.update(v,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),heartbeat_unix=time.time())
    atomic(ROOT/'progress.json',STATE)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    paths=list(REPO.rglob('*.py'))+[HERE/n for n in ['compare_simple3d_worker.py','compare_simple3d_controller.py',
        'simple3d_gpu_kdtree.py','simple3d_gpu_kdtree.cu','run_simple3d_comparison.sh']]
    return {str(p):sha(p) for p in paths}


def plan():
    jobs=[]
    for ds,file,fn in [('shapenet','anomalyshape.py','shapenet3d_classes'),('minishift','MiniShiftAD.py','minishiftAD_classes')]:
        tree=ast.parse((REPO/'data'/file).read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==fn)
        names=ast.literal_eval(next(n.value for n in function.body if isinstance(n,ast.Return)))
        for category in names:
            # Alternate which method runs first to reduce consistent ordering bias.
            modes=['official','gpu_kdtree'] if len(jobs)%4==0 else ['gpu_kdtree','official']
            for mode in modes:jobs.append(dict(mode=mode,dataset=ds,category=category))
    assert len(jobs)==104
    return jobs


def run(item,smoke=False):
    folder=(ROOT/'preflight' if smoke else ROOT)/item['mode']/item['dataset']/item['category']
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'summary.json').exists() or (folder/'train_timings.csv').exists():
        raise RuntimeError('Existing outputs require inspection, not automatic reuse: '+str(folder))
    cmd=[sys.executable,'-u',str(HERE/'compare_simple3d_worker.py')]
    for key,val in item.items():cmd+=['--'+key,val]
    if smoke:cmd+=['--smoke']
    started=time.perf_counter();readings=[];phase='startup'
    with (folder/'run.log').open('w') as log,(folder/'resources.csv').open('w',newline='') as rf:
        writer=None
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=REPO)
        state(status='preflight' if smoke else 'running',**item,worker_pid=proc.pid,phase='startup',worker_elapsed_s=0)
        try:
            while proc.poll() is None:
                try:
                    p=json.loads((folder/'progress.json').read_text());phase=p['phase']
                except (OSError,ValueError):p={}
                rss=0;cpu_s=0
                try:
                    parent=psutil.Process(proc.pid)
                    for child in [parent]+parent.children(recursive=True):
                        try:
                            rss+=child.memory_info().rss;c=child.cpu_times();cpu_s+=c.user+c.system
                        except psutil.Error:pass
                except psutil.Error:pass
                try:
                    query=subprocess.check_output(['nvidia-smi',
                        '--query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu',
                        '--format=csv,noheader,nounits'],text=True,timeout=5)
                    for line in query.strip().splitlines():
                        q=[x.strip() for x in line.split(',')]
                        row=dict(**item,elapsed_s=time.perf_counter()-started,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            phase=phase,sample_index=p.get('index',-1),gpu_index=q[0],gpu_util_percent=q[1],
                            memory_controller_util_percent=q[2],whole_gpu_used_mib=q[3],power_w=q[4],temperature_c=q[5],
                            process_tree_rss_bytes=rss,live_process_tree_cpu_seconds=cpu_s,
                            host_available_ram_bytes=psutil.virtual_memory().available)
                        if writer is None:writer=csv.DictWriter(rf,fieldnames=list(row));writer.writeheader()
                        writer.writerow(row);rf.flush();readings.append(row)
                except Exception as exc:
                    with (folder/'resource_errors.log').open('a') as f:f.write(str(exc)+'\n')
                state(phase=phase,sample_index=p.get('index',-1),sample_total=p.get('total',0),worker_elapsed_s=time.perf_counter()-started)
                time.sleep(0.5)
            rc=proc.wait()
        except BaseException:
            proc.terminate();proc.wait();raise
    if rc:raise RuntimeError(f'Worker exit {rc}: {folder}/run.log')
    summary=json.loads((folder/'summary.json').read_text())
    assert summary['status']==('passed_smoke' if smoke else 'complete')
    if not readings:raise RuntimeError('No GPU resource samples')
    groups={}
    for row in readings:groups.setdefault(row['phase'],[]).append(row)
    resources=[]
    for phase,rows in groups.items():
        r=dict(phase=phase,samples=len(rows),sampling_target_interval_s=0.5)
        for key in ['gpu_util_percent','memory_controller_util_percent','whole_gpu_used_mib','power_w','temperature_c',
                    'process_tree_rss_bytes','host_available_ram_bytes']:
            values=[]
            for row in rows:
                try:values.append(float(row[key]))
                except (TypeError,ValueError):pass
            r[key+'_mean']=statistics.mean(values) if values else None
            r[key+'_max']=max(values) if values else None
        resources.append(r)
    writecsv(folder/'resource_summary.csv',resources)
    atomic(folder/'resource_summary.json',resources)
    return summary


def aggregate(jobs):
    summaries=[];metrics=[];stages=[];resources=[]
    for item in jobs:
        f=ROOT/item['mode']/item['dataset']/item['category']
        if not (f/'summary.json').exists():continue
        s=json.loads((f/'summary.json').read_text())
        summaries.append({k:v for k,v in s.items() if k not in ['metrics','raw_assets']})
        metrics+=s['metrics']
        for file,rows in [('stage_summary.csv',stages),('resource_summary.csv',resources)]:
            with (f/file).open() as stream:
                for r in csv.DictReader(stream):rows.append(dict(**item,**r))
    writecsv(ROOT/'category_summary.csv',summaries);writecsv(ROOT/'metrics.csv',metrics)
    writecsv(ROOT/'stage_summary_all.csv',stages);writecsv(ROOT/'resource_summary_all.csv',resources)
    text='''# Simple3D 公式호환판 vs GPU KD-tree — 동일 조건 전체 측정

## 측정 조건과 해석

- Anomaly-ShapeNet 40개 + MiniShift 12개, 두 버전 각각 총 52개. 총 104개 독립 category 프로세스.
- 공통 원본은 공식 commit 5951e097ed5f6c52c1e058490f8c64f9b6323501의 최소 실행 호환판. 원본 clean clone 별도 보존.
- 공식 호환판: CPU legacy Open3D 법선/FPFH, native GPU FPS/KNN_CUDA, full-sort 보간, GPU torch.cdist/min.
- GPU KD-tree판: 같은 호환판에서 CuPy GPU KD-tree 및 CUDA FPFH backend만 적용. 동일 CPU 점수맵/CPU 인덱스 수정 공유. 기존 GPU 점수맵 반환 우회와 구분.
- 각 범주마다 먼저 실행할 버전을 교대함. 두 GPU 작업을 동시에 돌리지 않음. 같은 데이터/설정/원본 해상도, 새 뱅크/점수 생성. 기존 결과 재사용 없음.
- Python3.11/Torch2.11 CUDA12.8/Open3D0.20/RTX5080. 저자 구형 바이너리 환경과 동일하지 않음.
- 이벤트 기반 CUDA 단계 + CPU wall 단계 계측. 전체 추론은 특징 추출+점수 처리 동기화 wall time. 입력 읽기·저장·AUROC/AP는 추론 합계에서 제외.
- 시간 계측 래퍼와 이벤트 기록 비용이 포함된 instrumented latency임. 순수 무계측 지연과 같다고 주장하지 않음. 중첩 단계 합산 금지.
- GPU 사용률/메모리 컨트롤러 사용률/전체 GPU 메모리/전력/온도 및 CPU 프로세스 트리 RSS를 약 0.5초+조회 지연 간격으로 관찰. 정확한 커널별 전력·사용률이 아닌 해당 시간 구간 표본임.
- 전체 GPU 메모리는 화면·다른 프로세스 포함. Torch peak는 CuPy 외부 할당 제외. 둘을 별도 기록. CPU 시간 카운터는 현재 살아 있는 작업 프로세스의 누계이며 종료된 loader의 누적 CPU 시간으로 해석하지 않음.
- raw resource sample의 phase/index와 stage CSV의 phase/index를 함께 사용. 샘플링보다 짧은 세부 단계에는 독립 GPU 사용률을 귀속할 수 없음.
- 공식 수치 재현 보장이 아닌 구현 변경에 따른 성능·비용 비교 실험임. 전체 성공 전 평균을 최종 평균으로 쓰지 않음.

## 실행과 근거

- sh: [run_simple3d_comparison.sh](../source/run_simple3d_comparison.sh)
- code: [worker](../source/compare_simple3d_worker.py), [controller](../source/compare_simple3d_controller.py)
- 원본 수정: [호환 처리 내역](Simple3D_official_compatibility_20261104.md)
- result: [category_summary.csv](../source/results/official_vs_gpu_kdtree_20261104/category_summary.csv), [metrics.csv](../source/results/official_vs_gpu_kdtree_20261104/metrics.csv)
- 단계: [stage_summary_all.csv](../source/results/official_vs_gpu_kdtree_20261104/stage_summary_all.csv)
- 자원: [resource_summary_all.csv](../source/results/official_vs_gpu_kdtree_20261104/resource_summary_all.csv)
- 각 mode/dataset/category 폴더에 sample별 시간, stage 호출별 시간, resource 시계열, manifest 및 environment 기록.

## 진행

'''
    text+=f"- 상태: {STATE.get('status')}; 완료 {len(summaries)}/104, 카테고리 쌍 {len(summaries)//2}/52.\n\n"
    text+='| Dataset | Category | Version | Object AUROC (%) | Point AUROC (%) | Bank build (s) | Inference mean (ms) | p95 (ms) | Torch peak allocated (GB) |\n|---|---|---|---:|---:|---:|---:|---:|---:|\n'
    for s in summaries:
        metric=next(m for m in metrics if m['mode']==s['mode'] and m['dataset']==s['dataset'] and m['category']==s['category'] and m['level']=='ALL')
        text+=f"| {s['dataset']} | {s['category']} | {s['mode']} | {metric['object_auroc']*100:.2f} | {metric['point_auroc']*100:.2f} | {s['fit_wall_s']:.3f} | {s['inference_mean_ms']:.3f} | {s['inference_p95_ms']:.3f} | {s['peak_torch_allocated_bytes']/1e9:.3f} |\n"
    REPORT.write_text(text)
    return summaries


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'progress.json').exists():raise RuntimeError('Existing controller run preserved')
    jobs=plan();hashes=source_hashes();atomic(ROOT/'source_lock.json',hashes);atomic(ROOT/'plan.json',jobs)
    state(status='preflight',pid=os.getpid(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        completed_jobs=0,target_jobs=104,failures=[])
    aggregate(jobs)
    try:
        for dataset,cat in [('shapenet','ashtray0'),('minishift','capsule')]:
            for mode in ['official','gpu_kdtree']:run(dict(mode=mode,dataset=dataset,category=cat),True)
        for i,item in enumerate(jobs):
            assert source_hashes()==hashes,'Source changed during measurement'
            run(item)
            # The second method for each category must see exactly the same ordered files and labels.
            if i%2==1:
                a=ROOT/'official'/item['dataset']/item['category']/'manifest.json'
                b=ROOT/'gpu_kdtree'/item['dataset']/item['category']/'manifest.json'
                assert json.loads(a.read_text())==json.loads(b.read_text()),'Input manifest mismatch'
            state(status='running',completed_jobs=i+1)
            aggregate(jobs)
        summaries=aggregate(jobs)
        assert len(summaries)==104 and source_hashes()==hashes
        with (ROOT/'metrics.csv').open() as f:assert len(list(csv.DictReader(f)))==176
        files=[]
        for path in ROOT.rglob('*'):
            if path.suffix in ['.csv','.json'] and path.name not in ['progress.json','completion_manifest.json']:
                files.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
        atomic(ROOT/'completion_manifest.json',dict(status='verified',completed_jobs=104,files=files))
        state(status='complete_verified',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase='complete')
        aggregate(jobs);os.sync();print('ALL 104 JOBS VERIFIED',flush=True)
    except BaseException:
        state(status='failed',error=traceback.format_exc());aggregate(jobs);raise


if __name__=='__main__':main()
