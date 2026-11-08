"""Single capsule G FPFH140 run with the existing W46 resource monitor."""
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
from fpfh_only_publish import replace_with_retry

HERE=Path(__file__).resolve().parent
ROOT=HERE/'results/capsule_gpu_kdtree_fpfh60_100_140_20261108'
REPO=Path('/mnt/c/Users/test/Desktop/Codex/Project/Simple3D-official-compatible-20261104')
REPORT=HERE.parent/'markdown/W46_Simple3D_capsule_GPU_KDTree_60_100_140_vs_20_40_60.md'
STATE={}


def atomic(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2));replace_with_retry(temp,path)


def writecsv(path,rows):
    if not rows:return
    temp=path.with_suffix('.csv.tmp')
    with temp.open('w',newline='') as file:
        w=csv.DictWriter(file,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    replace_with_retry(temp,path)


def state(**v):
    STATE.update(v,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),heartbeat_unix=time.time())
    atomic(ROOT/'progress.json',STATE)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    paths=list(REPO.rglob('*.py'))+[HERE/n for n in ['measure_simple3d_capsule_fpfh60_100_140.py','run_simple3d_capsule_fpfh60_100_140.py',
        'simple3d_gpu_kdtree.py','simple3d_gpu_kdtree_fpfh60_100_140.py','simple3d_gpu_kdtree.cu','run_simple3d_capsule_fpfh60_100_140.sh']]
    return {str(p):sha(p) for p in paths}


def run(item,smoke=False):
    folder=(ROOT/'preflight' if smoke else ROOT)/item['mode']/item['dataset']/item['category']
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'summary.json').exists() or (folder/'train_timings.csv').exists():
        raise RuntimeError('Existing outputs require inspection, not automatic reuse: '+str(folder))
    cmd=[sys.executable,'-u',str(HERE/'measure_simple3d_capsule_fpfh60_100_140.py')]
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



BASE=HERE/'results/capsule_gpu_kdtree_fpfh60_20261108/gpu_kdtree/minishift/capsule'
ITEM=dict(mode='gpu_kdtree',dataset='minishift',category='capsule')

def load(path):
    return json.loads(path.read_text())

def report(old,new,dst):
    a=load(old/'summary.json'); b=load(new/'summary.json')
    lines=['# W46 Simple3D — 캡슐 GPU KD-tree FPFH 60·100·140 비교','',
           '- 기존 전체 GPU KD-tree판 G 기준. **FPFH 이웃 스케일만 20·40·60 → 60·100·140**으로 변경함.',
           '- 원본 점군 50만 점, 중심점 4,096개, 평활화 중심점 1,024개·이웃 12개, 특징 집계 이웃 16개 및 나머지 설정 유지함. 특징 연결은 60·100·100·140으로 132차원 유지함.',
           '- 동일한 정상 54개에서 각 설정에 맞춰 뱅크를 구성함. 테스트 130개 동일. FPFH60은 이전 실행 결과라 실행 시점 차이가 있음.',
           '- 차이는 FPFH140−FPFH60. 지표 차이는 %p, 시간은 원래 단위. 추론은 특징 추출과 점수 계산 합계이며 입출력·지표 계산 제외함.','',
           '| 항목 | 기존 FPFH60 | 새 FPFH140 | 차이 |','|---|---:|---:|---:|']
    for k,title,scale in [('fit_wall_s','메모리뱅크 구성 (s)',1),('inference_mean_ms','평균 추론 (ms)',1),
                         ('inference_median_ms','중앙값 (ms)',1),('inference_p95_ms','p95 (ms)',1),
                         ('peak_torch_allocated_bytes','최대 Torch 할당 (GiB)',1/2**30)]:
        x,y=a[k]*scale,b[k]*scale; lines.append(f'| {title} | {x:.2f} | {y:.2f} | {y-x:+.2f} |')
    lines+=['','## 성능','',
            '| 난이도 | 지표 | FPFH60 (%) | FPFH140 (%) | 차이 (%p) |','|---|---|---:|---:|---:|']
    for aa,bb in zip(a['metrics'],b['metrics']):
        assert aa['level']==bb['level']
        for k,title in [('object_auroc','물체 AUROC'),('point_auroc','포인트 AUROC'),('point_ap','포인트 AP')]:
            x,y=100*aa[k],100*bb[k];lines.append(f"| {aa['level']} | {title} | {x:.2f} | {y:.2f} | {y-x:+.2f} |")
    lines+=['','## 단계별 추론 시간','',
            '단계별 테스트 총 시간을 샘플 수로 나눈 ms/샘플. GPU event와 동기화 wall time은 계측 범위가 중첩될 수 있어 단순 합산하지 않음.','',
            '| 단계 | FPFH60 (ms) | FPFH140 (ms) | 차이 (ms) |','|---|---:|---:|---:|']
    def stages(folder):
        sums={}
        with (folder/'stage_summary.csv').open() as f:
            for r in csv.DictReader(f):
                if r['phase']!='test':continue
                name=r['stage']
                if name.startswith('gpu_kdtree_all_points_k'):name='FPFH 전체 점 이웃 조회 (60→140)'
                scales=(20,40,60) if folder==old else (60,100,140)
                for label,k in zip(('small','medium','large'),scales):
                    if name==f'fpfh_gpu_kdtree_scale_{k}':name='FPFH '+label
                if name=='coordinate_knn_gpu_kdtree':
                    name+=' / smooth' if 'k=12;' in r['detail'] else ' / feature'
                sums[name]=sums.get(name,0)+float(r['total_s'])*1000/130
        return sums
    x,y=stages(old),stages(new)
    for name in sorted(x.keys()|y.keys()):
        u,v=x.get(name),y.get(name)
        fmt=lambda n:'—' if n is None else f'{n:.3f}'
        lines.append(f'| {name} | {fmt(u)} | {fmt(v)} | {v-u:+.3f} |' if u is not None and v is not None else f'| {name} | {fmt(u)} | {fmt(v)} | — |')
    lines+=['','## 자원 사용량 (테스트 구간)','',
            'GPU 전체 VRAM은 다른 작업·화면 출력 포함. Torch 할당은 CuPy 메모리 제외.','',
            '| 항목 | FPFH60 | FPFH140 | 차이 |','|---|---:|---:|---:|']
    x=next(r for r in load(old/'resource_summary.json') if r['phase']=='test')
    y=next(r for r in load(new/'resource_summary.json') if r['phase']=='test')
    for k,title,scale in [('gpu_util_percent_mean','GPU 사용률 평균 (%; 차이는 %p)',1),
                         ('whole_gpu_used_mib_max','전체 VRAM 최대 (MiB)',1),
                         ('power_w_mean','GPU 전력 평균 (W)',1),
                         ('process_tree_rss_bytes_max','CPU RSS 최대 (GiB)',1/2**30)]:
        u,v=x[k],y[k]
        if u is not None and v is not None:lines.append(f'| {title} | {u*scale:.2f} | {v*scale:.2f} | {(v-u)*scale:+.2f} |')
    lines+=['','## 원본 기록','',
            '- [FPFH140 결과 폴더](../source/results/capsule_gpu_kdtree_fpfh60_100_140_20261108/gpu_kdtree/minishift/capsule/)',
            '- [기존 FPFH60 결과](../source/results/capsule_gpu_kdtree_fpfh60_20261108/gpu_kdtree/minishift/capsule/summary.json)',
            '- [실행 코드](../source/measure_simple3d_capsule_fpfh60_100_140.py), [실행 스크립트](../source/run_simple3d_capsule_fpfh60_100_140.sh).']
    dst.write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    if (ROOT/'progress.json').exists():raise RuntimeError('Existing run preserved')
    lock=source_hashes()
    state(status='preflight',target_jobs=1,completed_jobs=0,pid=os.getpid())
    try:
        run(ITEM,True)
        run(ITEM)
        new=ROOT/'gpu_kdtree/minishift/capsule'
        assert lock==source_hashes(),'Sources changed during run'
        assert load(new/'manifest.json')==load(BASE/'manifest.json'),'Input changed'
        a=load(BASE/'environment.json'); b=load(new/'environment.json')
        assert a['args']['group_size']==b['args']['group_size']==16
        assert a['args']['max_nn']==20 and b['args']['max_nn']==60
        assert b['fpfh_scales']==[60,100,140]
        assert dict(a['args'],max_nn=60)==b['args'],'Other options changed'
        assert a['source_sha256']==b['source_sha256'],'Official source changed'
        s=load(new/'summary.json'); assert s['train_count']==54 and s['test_count']==130
        with (new/'stage_timings.csv').open() as f:
            rows=[r for r in csv.DictReader(f) if r['phase']=='test' and r['stage']=='coordinate_knn_gpu_kdtree']
        assert len(rows)==260
        assert sum('k=16;' in r['detail'] for r in rows)==130
        assert sum('k=12;' in r['detail'] for r in rows)==130
        with (new/'stage_timings.csv').open() as f:
            stages=[r['stage'] for r in csv.DictReader(f) if r['phase']=='test']
        for key in ('gpu_kdtree_all_points_k140','fpfh_gpu_kdtree_scale_60','fpfh_gpu_kdtree_scale_100','fpfh_gpu_kdtree_scale_140'):
            assert stages.count(key)==130,key
        assert 'gpu_kdtree_all_points_k60' not in stages
        assert 'gpu_kdtree_all_points_k120' not in stages
        report(BASE,new,REPORT)
        atomic(ROOT/'verification.json',dict(status='verified',only_model_change='fpfh_scales',
             old=[20,40,60],new=[60,100,140],source_sha256=lock,feature_k16_calls=130,smoothing_k12_calls=130))
        state(status='complete_verified',phase='complete',completed_jobs=1)
    except BaseException:
        state(status='failed',error=traceback.format_exc())
        raise

if __name__=='__main__':
    main()
