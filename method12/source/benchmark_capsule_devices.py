"""Fresh capsule benchmark, native device placement, auditable stage traces.

No input/feature/score cache is reused. CUDA events observe native operations
without an extra synchronize between each kernel. Outer inference timing is
synchronized wall time. Upstream numerical operations and parameters stay intact.
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
import subprocess
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

os.environ['SIMPLE3D_NATIVE_OPS'] = '1'
import benchmark_simple3d as base
import numpy as np
import psutil
import torch
import open3d as o3d
import feature_extractors.FPFH as fpfh_module
from knn_cuda import KNN as NativeKNN
from pointnet2_ops import pointnet2_utils as native_ops
from feature_extractors.pointnet2_utils import interpolating_points_chunked
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results/capsule_devices_20261101'
ASSETS = Path('/home/test/simple3d-capsule-devices-20261101')
REPORT = HERE.parent / 'markdown/Simple3D_capsule_device_benchmark_20261101.md'
CONTEXT = dict(phase='warmup', index=-1)
PENDING = []
AUDIT = {}


def dump(name, value):
    p = OUT / name
    q = p.with_suffix(p.suffix + '.tmp')
    q.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    q.replace(p)


def append_csv(name, rows):
    if not rows:
        return
    p = OUT / name
    exists = p.exists()
    with p.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if not exists:
            w.writeheader()
        w.writerows(rows)


def observed(name, fn, device, detail):
    """Record device plus CPU wall duration or CUDA-stream event duration."""
    dump('active_stage.json', dict(**CONTEXT, stage=name, device=device, detail=detail,
                                  timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    if device.startswith('cuda'):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        elapsed = (start, end)
        clock = 'cuda_event'
    else:
        start = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - start
        clock = 'cpu_wall'
    PENDING.append((name, device, detail, clock, elapsed))
    key = name + '|' + device + '|' + detail
    AUDIT.setdefault(key, dict(stage=name, device=device, detail=detail, calls=0))['calls'] += 1
    return result


def flush_stages():
    base.sync()
    rows = []
    for name, device, detail, clock, value in PENDING:
        seconds = value[0].elapsed_time(value[1]) / 1000 if clock == 'cuda_event' else value
        rows.append(dict(**CONTEXT, stage=name, device=device, detail=detail,
                         clock=clock, seconds=seconds))
    PENDING.clear()
    append_csv('stage_timings.csv', rows)


def instrument():
    # A proxy is local to the FPFH module: dataset Open3D operations are untouched.
    class Cloud:
        def __init__(self, *args):
            self.native = o3d.geometry.PointCloud(*args)

        def estimate_normals(self, params):
            return observed('normal_estimation_including_neighbor_search',
                            lambda: self.native.estimate_normals(params), 'cpu',
                            f'Open3D legacy; radius={params.radius}; max_nn={params.max_nn}')

    def fpfh(cloud, params):
        return observed(f'fpfh_scale_{params.max_nn}',
                        lambda: o3d.pipelines.registration.compute_fpfh_feature(cloud.native, params),
                        'cpu', f'Open3D legacy; radius={params.radius}; max_nn={params.max_nn}')

    fpfh_module.o3d = SimpleNamespace(
        utility=o3d.utility,
        geometry=SimpleNamespace(PointCloud=Cloud, KDTreeSearchParamHybrid=o3d.geometry.KDTreeSearchParamHybrid),
        pipelines=SimpleNamespace(registration=SimpleNamespace(compute_fpfh_feature=fpfh)))

    class AuditedKNN:
        def __init__(self, k, transpose_mode=True):
            self.k = k
            self.native = NativeKNN(k=k, transpose_mode=transpose_mode)

        def __call__(self, reference, query):
            assert reference.is_cuda and query.is_cuda
            return observed('coordinate_knn', lambda: self.native(reference, query),
                            str(reference.device), f'native KNN_CUDA; k={self.k}; ref={list(reference.shape)}; query={list(query.shape)}')

    fpfh_module.KNN = AuditedKNN
    base.features_module.KNN = AuditedKNN
    original_fps, original_gather = native_ops.furthest_point_sample, native_ops.gather_operation

    def fps(x, count):
        assert x.is_cuda
        return observed('fps', lambda: original_fps(x, count), str(x.device),
                        f'native PointNet2; centers={count}; input={list(x.shape)}')

    def gather(x, indices):
        assert x.is_cuda and indices.is_cuda
        return observed('fps_gather', lambda: original_gather(x, indices), str(x.device),
                        f'native PointNet2; count={indices.shape[-1]}')

    native_ops.furthest_point_sample, native_ops.gather_operation = fps, gather

    def interpolation(a, b, values, chunk_size=50000):
        assert a.is_cuda and b.is_cuda and values.is_cuda
        return observed('point_score_interpolation',
                        lambda: interpolating_points_chunked(a, b, values, chunk_size=chunk_size),
                        str(a.device), f'official full-sort; dense={a.shape[-1]}; centers={b.shape[-1]}; chunk={chunk_size}')

    # benchmark_simple3d imports a compatibility interpolator; restore the original.
    base.features_module.interpolating_points_chunked = interpolation
    original_cdist, original_min, original_mean = torch.cdist, torch.min, torch.mean

    def cdist(a, b, *args, **kwargs):
        if CONTEXT['phase'] != 'test':
            return original_cdist(a, b, *args, **kwargs)
        assert a.is_cuda and b.is_cuda
        return observed('memory_bank_cdist', lambda: original_cdist(a, b, *args, **kwargs),
                        str(a.device), f'query={list(a.shape)}; bank={list(b.shape)}')

    def minimum(a, *args, **kwargs):
        if CONTEXT['phase'] != 'test' or not isinstance(a, torch.Tensor) or a.ndim != 2:
            return original_min(a, *args, **kwargs)
        return observed('memory_bank_nearest_neighbor_min', lambda: original_min(a, *args, **kwargs),
                        str(a.device), f'distances={list(a.shape)}')

    def mean(a, *args, **kwargs):
        if not isinstance(a, torch.Tensor) or a.ndim != 4:
            return original_mean(a, *args, **kwargs)
        assert a.is_cuda
        return observed('lfsa_mean' if a.shape[-1] > 1 else 'score_smoothing_mean',
                        lambda: original_mean(a, *args, **kwargs), str(a.device), f'input={list(a.shape)}')

    torch.cdist, torch.min, torch.mean = cdist, minimum, mean
    projection = base.features_module.random_projection.SparseRandomProjection
    original_project = projection.fit_transform

    def project(self, x, *args, **kwargs):
        assert not isinstance(x, torch.Tensor) or not x.is_cuda
        start = time.perf_counter()
        try:
            return observed('coreset_random_projection', lambda: original_project(self, x, *args, **kwargs),
                            'cpu', f'sklearn; input={list(x.shape)}')
        except ValueError:
            detail = f'sklearn projection rejected; official no-projection fallback; input={list(x.shape)}'
            PENDING.append(('coreset_random_projection_fallback', 'cpu', detail,
                            'cpu_wall', time.perf_counter()-start))
            AUDIT['projection_fallback'] = dict(stage='coreset_random_projection_fallback',
                                               device='cpu', detail=detail, calls=1)
            raise

    projection.fit_transform = project


def inventory():
    # Only capsule and prerequisites for THIS run; not all-category data/results.
    roots = dict(capsule_extracted=Path('/home/test/data/MiniShift/capsule'),
                 capsule_archive=Path('/home/test/data/MiniShift/capsule.zip'),
                 base_environment=Path('/home/test/simple3d-env'),
                 native_environment=Path('/home/test/simple3d-native-env'),
                 native_cuda_toolkit=Path('/home/test/simple3d-native-toolkit'),
                 native_extension_sources=Path('/home/test/simple3d-native-src'),
                 upstream_checkout=base.REPO, current_raw_assets=ASSETS,
                 current_measurement_files=OUT)
    seen = set()
    result = {}
    for key, root in roots.items():
        count = 0
        files = [root] if root.is_file() else root.rglob('*') if root.exists() else []
        for f in files:
            if not f.is_file() or f.is_symlink():
                continue
            st = f.stat()
            identity = (st.st_dev, st.st_ino)
            if identity not in seen:
                count += st.st_size
                seen.add(identity)
        result[key] = dict(path=str(root), exists=root.exists(), unique_logical_bytes=count)
    result['total_bytes'] = sum(v['unique_logical_bytes'] for v in result.values())
    result['definition'] = ('Logical bytes, hardlinks deduplicated, symlink targets counted at base roots only. '
                            'Includes capsule ZIP if retained, base/native environments and native toolkit/sources. '
                            'Excludes other categories, previous results, OS/driver, shared global caches and this inventory itself.')
    dump('storage.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--paper-cuda-fpfh', action='store_true',
                        help='Use paper-stated CUDA FPFH rather than legacy CPU API')
    opts = parser.parse_args()
    global OUT, ASSETS
    if opts.paper_cuda_fpfh:
        dtype = os.environ.get('SIMPLE3D_O3D_DTYPE', 'float32')
        OUT = OUT.with_name(OUT.name + '_paper_cuda_' + dtype)
        ASSETS = ASSETS.with_name(ASSETS.name + '-paper-cuda-' + dtype)
    if opts.smoke:
        OUT = OUT.with_name(OUT.name + '_smoke')
        ASSETS = ASSETS.with_name(ASSETS.name + '_smoke')
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'summary.json').exists() or (OUT / 'train_timings.csv').exists():
        raise RuntimeError('Output already contains a run; refuse to overwrite or reuse it.')
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / 'scores').mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base.dataset_module.DATASETS_PATH = '/home/test/data/MiniShift'
    train = base.dataset_module.MiniShiftADTrain('capsule')
    test = base.dataset_module.MiniShiftADTest('capsule', level='ALL')
    root = Path(base.dataset_module.DATASETS_PATH) / 'capsule'
    test.gt_paths = [0 if y == 0 else str(root / 'gt' / Path(p).relative_to(root / 'test'))
                     for p, y in zip(test.pcd_paths, test.labels)]
    assert len(train) == 54 and len(test) == 130
    args = SimpleNamespace(dataset='minishift', device='cuda:0', num_group=4096,
                           group_size=128, max_nn=40, use_LFSA=True, use_MSND=True,
                           num_MSND=2, feature='FPFH', level='ALL', vis_save=False)
    dump('environment.json', dict(args=vars(args), gpu=torch.cuda.get_device_name(),
         torch=torch.__version__, cuda=torch.version.cuda, open3d=o3d.__version__,
         python=base.sys.version, upstream_commit=subprocess.check_output(
             ['git', '-C', str(base.REPO), 'rev-parse', 'HEAD'], text=True).strip(),
         runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         train_count=len(train), test_count=len(test), pid=os.getpid(), smoke=opts.smoke,
         native_fps_module=native_ops.__file__, native_knn_module=base.sys.modules['knn_cuda'].__file__,
         scales_computed=[40, 80, 120], scales_concatenated=[40, 80, 80, 120],
         compatibility='native CUDA12.8 sm_120 PointNet2 and source-preserving KNN fork; upstream path/device fixes retained',
         fresh_fit=True, reused_test_features=False, paper_cuda_fpfh=opts.paper_cuda_fpfh,
         fpfh_device='CUDA:0' if opts.paper_cuda_fpfh else 'CPU:0',
         fpfh_backend_note=('Open3D tensor CUDA, ' + os.environ.get('SIMPLE3D_O3D_DTYPE', 'float32') + ', max_nn queries with verified nonbinding radius bounds; not bitwise legacy CPU parity') if opts.paper_cuda_fpfh else 'Open3D legacy CPU'))
    dump('manifest.json', dict(train=train.pcd_paths, test=[dict(path=p, label=int(y), gt=g)
         for p, y, g in zip(test.pcd_paths, test.labels, test.gt_paths)]))
    instrument()
    if opts.paper_cuda_fpfh:
        import simple3d_paper_cuda_fpfh
        import sys
        simple3d_paper_cuda_fpfh.install(sys.modules[__name__])
    model = base.FPFHFeatures(args=args)  # official constructor sets seed=0
    proc = psutil.Process()
    stop = threading.Event()
    peak_rss = [proc.memory_info().rss]
    baseline_gpu_used = torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]
    gpu_device_peak_used = [baseline_gpu_used]

    def monitor():
        while not stop.wait(.1):
            peak_rss[0] = max(peak_rss[0], proc.memory_info().rss)
            free, total = torch.cuda.mem_get_info()
            gpu_device_peak_used[0] = max(gpu_device_peak_used[0], total-free)

    threading.Thread(target=monitor, daemon=True).start()
    started = time.perf_counter()

    def progress(phase, done, total):
        dump('progress.json', dict(status='running', pid=os.getpid(), phase=phase,
             sample_completed=done, sample_total=total, elapsed_s=time.perf_counter()-started,
             updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))

    with torch.inference_mode():
        progress('warmup', 0, 1)
        pc, *_ = train[0]
        warm_start = time.perf_counter()
        model.get_features(torch.as_tensor(pc)[None])
        base.sync()
        warmup_s = time.perf_counter()-warm_start
        flush_stages()
        torch.cuda.reset_peak_memory_stats()
        fit_start = time.perf_counter()
        training = []
        ntrain = 1 if opts.smoke else len(train)
        for i in range(ntrain):
            CONTEXT.update(phase='fit', index=i)
            t = time.perf_counter()
            pc, *_ = train[i]
            load = time.perf_counter()-t
            assert pc.shape == (500000, 3)
            base.sync()
            t = time.perf_counter()
            model.collect_features(torch.as_tensor(pc)[None])
            base.sync()
            feature_s = time.perf_counter()-t
            flush_stages()
            row = dict(index=i, path=train.pcd_paths[i], points=len(pc), load_s=load, feature_s=feature_s)
            training.append(row)
            append_csv('train_timings.csv', [row])
            progress('fit', i+1, ntrain)
            print(f'TRAIN {i+1}/{ntrain} feature={feature_s:.3f}s', flush=True)
        CONTEXT.update(phase='coreset', index=-1)
        base.sync()
        t = time.perf_counter()
        model.run_coreset()
        base.sync()
        coreset_s = time.perf_counter()-t
        fit_wall_s = time.perf_counter()-fit_start
        flush_stages()
        assert model.patch_lib.is_cuda and model.patch_lib.shape[1] == 132
        torch.save(dict(memory_bank=model.patch_lib.detach().cpu(), args=vars(args)), ASSETS/'memory_bank.pt')
        dump('fit_state.json', dict(fit_wall_s=fit_wall_s, coreset_s=coreset_s,
             warmup_s=warmup_s, bank_shape=list(model.patch_lib.shape), effective_seed=0))
        print(f'FIT COMPLETE {fit_wall_s:.3f}s bank={list(model.patch_lib.shape)}', flush=True)
        times, image_scores, image_labels = [], [], []
        ntest = 2 if opts.smoke else len(test)
        for i in range(ntest):
            CONTEXT.update(phase='test', index=i)
            t = time.perf_counter()
            pc, mask, label, path = test[i]
            load = time.perf_counter()-t
            assert pc.shape == (500000, 3)
            model.init_para()
            base.sync()
            t = time.perf_counter()
            f, raw, dense, centers = model.get_features(torch.as_tensor(pc)[None])
            base.sync()
            feature_s = time.perf_counter()-t
            assert f.device.type == 'cuda' and f.shape == (4096, 132)
            t = time.perf_counter()
            model.compute_anomay_scores(f, mask, torch.tensor([label]), [path], raw, dense, centers)
            base.sync()
            score_s = time.perf_counter()-t
            flush_stages()
            pred = np.asarray(model.predictions[-1], dtype=np.float32).reshape(-1).copy()
            gt = np.asarray(model.gts[-1], dtype=np.uint8).reshape(-1).copy()
            assert len(pred) == len(gt) == 500000 and np.isfinite(pred).all()
            image_score = float(model.image_preds[-1])
            image_scores.append(image_score)
            image_labels.append(int(label))
            t = time.perf_counter()
            np.savez_compressed(ASSETS/'scores'/f'{i:03d}.npz', scores=pred, labels=gt)
            save_s = time.perf_counter()-t
            row = dict(index=i, path=path, label=int(label), points=len(pc),
                       anomalous_points=int(gt.sum()), image_score=image_score, load_s=load,
                       feature_s=feature_s, score_s=score_s, inference_s=feature_s+score_s, save_s=save_s)
            times.append(row)
            append_csv('test_timings.csv', [row])
            progress('test', i+1, ntest)
            print(f'TEST {i+1}/{ntest} feature={feature_s:.3f}s score={score_s:.3f}s', flush=True)
            del f, raw, dense, centers, pred, gt
            model.init_para()
        peak_alloc = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
    dump('device_audit.json', list(AUDIT.values()))
    if opts.smoke:
        dump('smoke.json', dict(status='passed', test_samples=ntest,
             bank_shape=list(model.patch_lib.shape), peak_gpu_allocated_bytes=peak_alloc))
        print('SMOKE PASSED', flush=True)
        stop.set()
        return
    progress('metrics', 0, 1)
    preds, labels = [], []
    for i in range(ntest):
        with np.load(ASSETS/'scores'/f'{i:03d}.npz') as z:
            preds.append(z['scores'])
            labels.append(z['labels'])
    t = time.perf_counter()
    all_pred, all_gt = np.concatenate(preds), np.concatenate(labels)
    preds.clear()
    labels.clear()
    metrics = dict(object_auroc=float(roc_auc_score(image_labels, image_scores)),
                   point_auroc=float(roc_auc_score(all_gt, all_pred)),
                   point_count=int(len(all_gt)), anomaly_point_count=int(all_gt.sum()))
    metric_s = time.perf_counter()-t
    storage = inventory()
    summary = dict(status='complete', category='capsule', level='ALL', train_count=ntrain,
        test_count=ntest, full_category=True, metrics=metrics, fit_wall_s=fit_wall_s,
        train_load_s=sum(r['load_s'] for r in training), train_feature_s=sum(r['feature_s'] for r in training),
        coreset_s=coreset_s, warmup_s=warmup_s, inference=base.stats([r['inference_s'] for r in times]),
        load_plus_inference=base.stats([r['load_s']+r['inference_s'] for r in times]),
        peak_gpu_allocated_bytes=peak_alloc, peak_gpu_reserved_bytes=peak_reserved,
        peak_gpu_device_used_bytes=gpu_device_peak_used[0], gpu_device_baseline_used_bytes=baseline_gpu_used,
        gpu_device_peak_scope='100ms sampled whole GPU used memory including Open3D, desktop, driver and other programs. Not model-exclusive or exact unsampled peak.',
        peak_process_rss_bytes=peak_rss[0], metric_s=metric_s, storage_total_bytes=storage['total_bytes'],
        raw_assets=str(ASSETS), timing_scope='Fresh full fit; warmup excluded; inference includes CPU normals/FPFH, transfers, GPU geometry/NN/interpolation and official CPU score collection; excludes input loading, output persistence and AUROC. Stage CUDA events add small observer overhead.',
        memory_scope='PyTorch allocator peaks over complete fresh fit + evaluation, warmup excluded. Does not include desktop/driver/non-PyTorch allocations. Reserved is separately reported.',
        stage_clock_note='CPU wall times and CUDA event durations have different clocks; do not add them as an exact wall-time decomposition.',
        fps_claim='Local native-rebuilt RTX5080 measurement, not the original old CUDA/Torch environment.')
    dump('summary.json', summary)
    main_row = dict(object_auroc_pct=metrics['object_auroc']*100, point_auroc_pct=metrics['point_auroc']*100,
        fit_wall_s=fit_wall_s, inference_mean_ms=summary['inference']['mean_ms'],
        peak_gpu_allocated_gb=peak_alloc/1e9, peak_gpu_device_used_gb=gpu_device_peak_used[0]/1e9,
        storage_total_gb=storage['total_bytes']/1e9)
    append_csv('comparison.csv', [main_row])
    with (OUT/'stage_timings.csv').open() as f:
        stage_rows = list(csv.DictReader(f))
    aggregated = []
    for key in sorted({(r['phase'], r['stage'], r['device'], r['detail'], r['clock']) for r in stage_rows}):
        selected = [float(r['seconds']) for r in stage_rows if tuple(r[k] for k in ['phase','stage','device','detail','clock']) == key]
        aggregated.append(dict(zip(['phase','stage','device','detail','clock'], key), calls=len(selected),
                               mean_ms=float(np.mean(selected)*1000), total_s=sum(selected)))
    append_csv('stage_summary.csv', aggregated)
    text = '''# Simple3D capsule — 단계별 실행 장치 확인과 새 측정

## Paper Metadata

| Item | Content |
|---|---|
| Title | Towards High-Resolution 3D Anomaly Detection: A Scalable Dataset and Real-Time Framework for Subtle Industrial Defects |
| Authors | Yuqi Cheng, Yihan Sun, Hui Zhang, Weiming Shen, Yunkang Cao |
| Conference / Journal | AAAI |
| Year | 2026 |
| Paper link | https://doi.org/10.1609/aaai.v40i5.37328 |
| GitHub / Official code | https://github.com/hustCYQ/MiniShift-Simple3D |
| Reason for investigation | capsule 범주에서 native 실행 장치를 확인하고 성능·시간·메모리·저장 용량을 새로 측정 |

## 범위와 결과

정상 54개로 새 메모리뱅크 구성, 테스트 130개 전체 평가. 각 점군 500,000점 유지. 기존 메모리뱅크·특징·점수 재사용 없음. 이전 결과는 보존함.

| 물체 AUROC (%) | 포인트 AUROC (%) | 학습·메모리뱅크 구성 시간 | 평균 추론 (ms/샘플) | 최대 GPU 메모리 (GB) | 총 저장 용량 (GB) |
|---:|---:|---:|---:|---:|---:|
'''
    text += f"| {main_row['object_auroc_pct']:.4f} | {main_row['point_auroc_pct']:.4f} | {fit_wall_s:.3f}초 ({fit_wall_s/60:.2f}분) | {main_row['inference_mean_ms']:.3f} | {main_row['peak_gpu_device_used_gb']:.4f} | {main_row['storage_total_gb']:.3f} |\n"
    text += '''
## 실행 장치

| 단계 | 실제 장치 / 방식 |
|---|---|
| 로딩·중심 정렬 | CPU, 공식 로더 |
| 법선·법선용 좌표 이웃 탐색 | CPU, Open3D legacy KD-tree, radius=10000, max_nn=10 |
| FPFH 스케일별 이웃 탐색·계산 | CPU, Open3D legacy, radius=1000000, max_nn=40/80/120 |
| 특징 결합·장치 전송 | CPU 결과 → GPU, 결합 차원 132 |
| 특징 집계용 FPS / gather | GPU, native PointNet2, 중심 4096개 |
| 특징 집계용 좌표 이웃 탐색 | GPU, native KNN_CUDA, k=128 |
| LFSA 평균 집계 | GPU |
| coreset 임시 투영 | CPU, sklearn. 투영 불가 시 원본 코드의 예외 경로 유지 |
| coreset farthest-first 선택 | CPU 준비 후 GPU float16 거리 갱신·선택, 공식 5% |
| 메모리뱅크 NN | GPU, torch.cdist + min |
| 포인트 점수 1차 보간 | GPU, 공식 전체 정렬, chunk=50000, 최근접 3개 가중합 |
| 점수 평활화용 FPS·이웃·평균 | GPU, 중심 1024개, k=12 |
| 포인트 점수 2차 보간 | GPU, 공식 전체 정렬, chunk=50000 |
| 점수맵 재배열 | 공식 CPU 왕복 후 CUDA 인덱스 오류 방지를 위한 GPU 복귀 |
| 최종 물체 점수·평가 | 공식 상위 80점 평균은 GPU, 정답 비교·AUROC는 CPU |

법선 내부 이웃 탐색과 FPFH 내부 이웃 탐색은 Open3D 함수에 포함되어 별도 커널 시간으로 분리하지 않음. device_audit.json과 stage_timings.csv에 호출 시 확인한 장치·입출력 크기·횟수를 남김.

## 측정 해석

- GPU를 쓰기 위해 CPU FPFH를 다른 알고리즘으로 교체하지 않음. 좌표 kNN은 CPU KD-tree 대체가 아니라 native CUDA 경로 사용. 보간도 topk 대체가 아닌 공식 full-sort 사용.
- 공식 MSND 코드는 40/80/120 스케일을 계산하지만 특징은 40/80/80/120으로 결합함. 이 중복을 그대로 유지하여 차원은 132임.
- Simple3D에는 신경망 optimizer 학습이 없음. 표의 학습 시간은 전체 정상 로딩·특징 추출·5% coreset 선택까지의 새 메모리뱅크 구성 시간임. warmup·뱅크 파일 저장 제외, 진행 기록 오버헤드는 포함함.
- 평균 추론은 입력이 로딩된 뒤 법선/FPFH 계산부터 최종 점수 수집까지 포함함. 데이터 읽기·결과 압축 저장·AUROC 계산은 제외함. CUDA 동기화된 wall time이며 단계별 CUDA event 관찰에 따른 소량의 오버헤드가 있음.
- 최대 GPU 메모리는 새 fit와 평가 전체의 PyTorch 최대 allocated, GB=10^9 bytes임. 드라이버·화면·다른 프로그램 메모리는 포함하지 않음. reserved 값과 시스템 관측치는 별도로 기록하므로 서로 혼동하지 않음.
- 총 저장 용량은 capsule 데이터(남아 있는 ZIP 포함), 기본·native 환경, native toolkit·소스, 공식 checkout, 이번 원시 결과의 논리 파일 크기임. 중복 hardlink는 한 번만 계산함. 다른 카테고리·과거 결과·OS/드라이버·공용 캐시는 제외함. 물리 디스크 증가량과는 다름.
- 구형 Torch/CUDA 바이너리를 그대로 실행한 것이 아니라 RTX5080용으로 재빌드한 로컬 측정임. 데이터 경로 오류·점수맵 장치 혼용 우회 유지. 과거 호환 백엔드 결과나 논문의 FPS와 동일 조건으로 단정하지 않음.

## 근거와 실행

- 실행: `bash method12/source/run_capsule_devices.sh`
- [실행 코드](../source/benchmark_capsule_devices.py)
- [설정·소스 해시](../source/results/capsule_devices_20261101/environment.json)
- [전체 샘플 목록](../source/results/capsule_devices_20261101/manifest.json)
- [실제 장치 호출 기록](../source/results/capsule_devices_20261101/device_audit.json)
- [단계별 원시 시간](../source/results/capsule_devices_20261101/stage_timings.csv)
- [단계별 시간 집계](../source/results/capsule_devices_20261101/stage_summary.csv)
- [학습 시간](../source/results/capsule_devices_20261101/train_timings.csv)
- [추론 시간과 물체 점수](../source/results/capsule_devices_20261101/test_timings.csv)
- [요약 JSON](../source/results/capsule_devices_20261101/summary.json)
- [표 원시 CSV](../source/results/capsule_devices_20261101/comparison.csv)
- [저장 용량 상세](../source/results/capsule_devices_20261101/storage.json)
- 원시 포인트 점수·정답 및 메모리뱅크는 summary.json의 raw_assets 경로에 보관함. 데이터셋·가상환경·메모리뱅크를 GitHub에 업로드하지 않음.
'''
    text = text.replace('최대 GPU 메모리는 새 fit와 평가 전체의 PyTorch 최대 allocated, GB=10^9 bytes임. 드라이버·화면·다른 프로그램 메모리는 포함하지 않음. reserved 값과 시스템 관측치는 별도로 기록하므로 서로 혼동하지 않음.',
        '최대 GPU 메모리는 Open3D 할당도 포함하기 위해 100ms 주기로 관측한 장치 전체 사용량의 최대값임. 드라이버·화면·다른 프로그램도 포함하므로 모델 전용 메모리로 해석하면 안 됨. PyTorch allocated/reserved 값은 summary.json에 별도 기록함. GB=10^9 bytes이며 샘플링 사이 순간 최대값은 놓칠 수 있음.')
    text += f'\nPyTorch 최대 allocated: {peak_alloc/1e9:.4f} GB, reserved: {peak_reserved/1e9:.4f} GB. 장치 초기 사용량: {baseline_gpu_used/1e9:.4f} GB.\n'
    if opts.paper_cuda_fpfh:
        text = text.replace('capsule_devices_20261101/', OUT.name + '/')
        text = text.replace('CPU, Open3D legacy KD-tree, radius=10000, max_nn=10',
                            'GPU, Open3D tensor CUDA, max_nn=10; 반경이 전체 점군을 포함함을 검증')
        text = text.replace('CPU, Open3D legacy, radius=1000000, max_nn=40/80/120',
                            'GPU, Open3D tensor CUDA, max_nn=40/80/120; CPU fallback 금지')
        text = text.replace('CPU 결과 → GPU, 결합 차원 132', 'CUDA tensor에서 DLPack으로 GPU 직접 공유, 결합 차원 132')
        text = text.replace('GPU를 쓰기 위해 CPU FPFH를 다른 알고리즘으로 교체하지 않음.',
                            '논문의 CUDA 가속 FPFH 설명에 맞춰 legacy CPU API를 Open3D tensor CUDA FPFH로 교체함. GPU 법선·FPFH 수치가 legacy CPU 결과와 비트 단위 동일하다고 주장하지 않음.')
        text += '\n논문이 명시하지 않은 세부 장치 배치는 공개 코드로 보완함. Open3D tensor는 모든 점이 기존 반경 안에 있음을 확인한 뒤 max_nn만 사용하여 같은 최근접 이웃 목적을 계산함. GPU 수치·동률 처리 차이가 있을 수 있음. 공식 코드의 중복 스케일과 물체 top-80 집계는 장치 변경 외의 별도 차이로 남겨 두므로 논문 모든 수식과 완전히 동일하다는 주장은 하지 않음.\n'
    REPORT.write_text(text, encoding='utf-8')
    stop.set()
    dump('progress.json', dict(status='complete', phase='complete', pid=os.getpid(),
        elapsed_s=time.perf_counter()-started, summary='summary.json'))
    print(json.dumps(main_row, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        OUT.mkdir(parents=True, exist_ok=True)
        dump('failure.json', dict(status='failed', error=traceback.format_exc()))
        raise
