"""Run pinned official APRIL-GAN with exact disk-backed arrays and evidence."""
import argparse
import functools
import shutil
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--repo', default='/home/test/VAND-APRIL-GAN')
parser.add_argument('--data', default='/home/test/data/mvtec')
parser.add_argument('--output', default='/home/test/aprilgan_results/mvtec_zero_shot_20261006')
parser.add_argument('--dataset', choices=['mvtec','visa','btad'], default='mvtec')
parser.add_argument('--mode', choices=['zero_shot','few_shot'], default='few_shot')
parser.add_argument('--tag', required=True)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--backbone', choices=['ViT-L-14-336', 'ViT-B-16-plus-240'], default='ViT-L-14-336')
parser.add_argument('--checkpoint')
opts = parser.parse_args()
repo, data, output = Path(opts.repo), Path(opts.data), Path(opts.output)
evidence = Path(__file__).resolve().parent / 'results' / opts.tag
sys.dont_write_bytecode = True
if (output / 'environment.json').exists():
    raise SystemExit('Evidence already exists; select a fresh output path.')
output.mkdir(parents=True, exist_ok=True)
evidence.mkdir(parents=True, exist_ok=True)
os.chdir(repo)
sys.path.insert(0, str(repo))
import test as official

# Preserve existing metadata. Validate its exact test file/label set against
# current files rather than rewriting the shared dataset for this run.
if not (data / 'meta.json').exists():
    from data.mvtec import MVTecSolver
    MVTecSolver(root=str(data)).run()
meta = json.loads((data / 'meta.json').read_text())
categories = list(meta['test'])
expected_count = 1725 if opts.dataset == 'mvtec' else 2162 if opts.dataset == 'visa' else sum(map(len,meta['test'].values()))
assert sum(map(len,meta['test'].values())) == expected_count
audit = []
for category in categories:
    entries = meta['test'][category]
    actual = {row['img_path']: int(row['anomaly']) for row in entries}
    assert len(actual) == len(entries)
    for row in entries:
        assert (data / row['img_path']).is_file()
    for row in entries:
        if row['anomaly']:
            assert (data / row['mask_path']).is_file()
    audit.append(dict(category=category, n_images=len(entries), n_normal=sum(v == 0 for v in actual.values()),
                      n_anomaly=sum(v == 1 for v in actual.values()), paths_and_labels_validated=True))
with (evidence / 'dataset_audit.csv').open('w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=list(audit[0])); writer.writeheader(); writer.writerows(audit)

small = opts.backbone == 'ViT-B-16-plus-240'
layers = [3, 6, 9, 12] if small else [6, 12, 18, 24]
args = SimpleNamespace(data_path=str(data), save_path=str(output / 'official'),
                       checkpoint_path=opts.checkpoint or str(repo / ('exps/pretrained/mvtec_pretrained.pth' if opts.dataset == 'visa' else 'exps/pretrained/visa_pretrained.pth')),
                       config_path=str(repo / 'open_clip/model_configs' / (opts.backbone + '.json')),
                       dataset='visa' if opts.dataset == 'btad' else opts.dataset, model=opts.backbone, pretrained='laion400m_e31' if small else 'openai',
                       features_list=layers, few_shot_features=layers, image_size=518,
                       mode=opts.mode, k_shot=4 if opts.mode == 'few_shot' else 0, seed=opts.seed)
official.setup_seed(args.seed)
# Match the existing MuSc experiment's explicitly disabled TF32 condition.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
environment = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                   thread_environment={name:os.environ.get(name) for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},
                   instrumentation='captures local raw predictions and raw CLIP features; line-traced timing is ancillary, not isolated benchmark',
                   official_repo='https://github.com/ByChelsea/VAND-APRIL-GAN', python=sys.version,
                   platform=platform.platform(), gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                   packages={name: importlib.metadata.version(name) for name in
                             ['torch', 'torchvision', 'numpy', 'timm', 'scikit-learn', 'scikit-image', 'Pillow']},
                   args=vars(args), checkpoint_training_dataset='MVTec AD' if opts.dataset == 'visa' else 'VisA', target_dataset=opts.dataset,
                   checkpoint_sha256=hashlib.sha256(Path(args.checkpoint_path).read_bytes()).hexdigest(),
                   metadata_sha256=hashlib.sha256((data / 'meta.json').read_bytes()).hexdigest(),
                   tf32=False, target_training=False, source_edits=False, status='running',
                   command=sys.argv, raw_output=str(output))
environment['wrapper_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
environment['wrapper_source']=str(Path(__file__).resolve())
environment['projection_checkpoint_source']='own auxiliary VisA training' if small else 'official public checkpoint'
def write_environment():
    encoded = json.dumps(environment, indent=2)
    (output / 'environment.json').write_text(encoded)
    (evidence / 'environment.json').write_text(encoded)
write_environment()
(evidence / 'config.json').write_text(json.dumps(vars(args), indent=2))

# Capture exact references, projected image features, and inference-only timing.
import dataset as dataset_module
import few_shot as few_module
active = {}
dataset_entries = []
references = []
image_features = {}
def captured_class(base):
    class Captured(base):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.capture_mode = kw.get('mode','test')
            if self.capture_mode == 'test': dataset_entries.extend(self.data_all)
            else: references.extend(self.data_all)
        def __getitem__(self, index):
            active.update(mode=self.capture_mode, entry=self.data_all[index])
            return super().__getitem__(index)
    return Captured
for name in ['MVTecDataset','VisaDataset']:
    cls=captured_class(getattr(dataset_module,name))
    setattr(dataset_module,name,cls);setattr(official,name,cls);setattr(few_module,name,cls)
create = official.open_clip.create_model_and_transforms
def capture_model(*a, **kw):
    model, train_transform, transform = create(*a, **kw)
    encode=model.encode_image
    def capture_encode(*a, **kw):
        result=encode(*a, **kw)
        if active.get('mode')=='test':
            path=active['entry']['img_path']
            if path not in image_features: image_features[path]=result[0].detach().float().cpu().numpy().copy()
        return result
    model.encode_image=capture_encode
    return model,train_transform,transform
of_create=official.open_clip.create_model_and_transforms
of_file=Path(official.__file__).resolve()
lines=of_file.read_text().splitlines()
start_line=next(i+1 for i,l in enumerate(lines) if l.strip()=="image = items['img'].to(device)")
end_line=next(i+1 for i,l in enumerate(lines) if l.strip()=="results['anomaly_maps'].append(anomaly_map)")+1
inference_times=[]
clock_start=None
streamed=False
cache=output/'inference_cache'
class DiskArrayList:
    """Keep exact upstream arrays on disk so category metrics fit WSL RAM."""
    def __init__(self,name,tensor=False):
        self.name=name;self.tensor=tensor;self.paths=[]
        (cache/name).mkdir(parents=True,exist_ok=True)
    def append(self,value):
        path=cache/self.name/f'{len(self.paths):05d}.npy'
        np.save(path,value.detach().cpu().numpy() if self.tensor else value,allow_pickle=False)
        self.paths.append(path)
        if self.name=='maps':
            local=sys._getframe(1).f_locals
            entry=active['entry'];idx=len(self.paths)-1
            np.save(cache/'features'/f'{idx:05d}.npy',image_features[entry['img_path']])
            row=dict(index=idx,entry=entry,text_score=float(local['results']['pr_sp'][-1]))
            with (cache/'images.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    def __len__(self): return len(self.paths)
    def __getitem__(self,index):
        value=np.load(self.paths[index],allow_pickle=False)
        return torch.from_numpy(value) if self.tensor else value
def trace(frame,event,arg):
    global clock_start,streamed
    if frame.f_code.co_filename!=str(of_file) or frame.f_code.co_name!='test': return None
    if event=='line':
        if frame.f_lineno==start_line:
            if not streamed:
                (cache/'features').mkdir(parents=True,exist_ok=True)
                frame.f_locals['results']['imgs_masks']=DiskArrayList('masks',tensor=True)
                frame.f_locals['results']['anomaly_maps']=DiskArrayList('maps')
                streamed=True
            torch.cuda.synchronize();clock_start=time.perf_counter()
        elif clock_start is not None and frame.f_lineno>=end_line:
            torch.cuda.synchronize();inference_times.append(time.perf_counter()-clock_start);clock_start=None
    return trace
original_test=official.test
original_pro = official.cal_pro_score
rows = []
names = ['category', 'n_images', 'image_auroc', 'image_f1_max', 'image_ap',
         'pixel_auroc', 'pixel_f1_max', 'pixel_ap', 'aupro']
def save_rows():
    for destination in (output / 'category_metrics.csv', evidence / 'category_metrics.csv'):
        with destination.open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=names); writer.writeheader(); writer.writerows(rows)

def capture_pro(masks, maps, *a, **kw):
    # Official test() calls cal_pro_score after all its other category metrics.
    # Capture its exact arrays and unrounded values without recomputing scoring.
    local = sys._getframe(1).f_locals
    category = local['obj']
    labels, scores = local['gt_sp'], local['pr_sp']
    assert np.isfinite(scores).all() and np.isfinite(maps).all()
    folder = output / category
    folder.mkdir(exist_ok=True)
    np.savez_compressed(folder / 'raw_predictions.npz', image_labels=labels, image_scores=scores,
                        masks=masks, anomaly_maps=maps)
    images = [row for row in dataset_entries if row['cls_name'] == category]
    assert len(images) == len(scores) == len(meta['test'][category])
    with (folder / 'image_scores.csv').open('w', newline='') as file:
        writer = csv.writer(file); writer.writerow(['image_path', 'label', 'score', 'sha256'])
        for entry, label, score in zip(images, labels, scores):
            path = data / entry['img_path']
            writer.writerow([str(path), int(label), float(score), hashlib.sha256(path.read_bytes()).hexdigest()])
    features=np.concatenate([image_features[r['img_path']] for r in images])
    np.save(folder / 'clip_image_features.npy',features)
    started = time.monotonic()
    pro = original_pro(masks, maps, *a, **kw)
    row = dict(category=category, n_images=len(scores), image_auroc=float(local['auroc_sp']),
               image_f1_max=float(local['f1_sp']), image_ap=float(local['ap_sp']),
               pixel_auroc=float(local['auroc_px']), pixel_f1_max=float(local['f1_px']),
               pixel_ap=float(local['ap_px']), aupro=float(pro))
    rows.append(row); save_rows()
    category_evidence = evidence / category
    category_evidence.mkdir(exist_ok=True)
    (category_evidence / 'image_scores.csv').write_bytes((folder / 'image_scores.csv').read_bytes())
    print('CATEGORY COMPLETED', category, 'image AUROC', row['image_auroc'], 'pixel AUROC', row['pixel_auroc'],
          'AUPRO seconds', time.monotonic() - started, flush=True)
    return pro
official.cal_pro_score = capture_pro

started = time.monotonic()
torch.cuda.reset_peak_memory_stats()
try:
    official.open_clip.create_model_and_transforms=capture_model
    sys.settrace(trace)
    official.test(args)
    sys.settrace(None)
    assert len(inference_times)==expected_count,(len(inference_times),expected_count)
    with (evidence/'inference_times.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['image_index','inference_seconds']);w.writerows(enumerate(inference_times))
    (evidence/'references.json').write_text(json.dumps(references,indent=2))
    (evidence/'benchmark.json').write_text(json.dumps(dict(n_images=expected_count,mean_ms=1000*np.mean(inference_times),mean_ms_excluding_first=1000*np.mean(inference_times[1:]),peak_allocated_mb=torch.cuda.max_memory_allocated()/1024**2,peak_reserved_mb=torch.cuda.max_memory_reserved()/1024**2,memory_unit='MiB',timing_scope='H2D and official scoring plus local feature capture and disk-cache writes; excludes loader, visualization, metrics, model/text/reference setup',instrumentation='ancillary line trace with CUDA synchronization; not an isolated speed benchmark'),indent=2))
    assert len(rows) == len(categories) and {r['category'] for r in rows} == set(categories)
    assert sum(row['n_images'] for row in rows) == expected_count
    assert not subprocess.check_output(['git','diff','--name-only'],text=True).strip(), 'Tracked official source changed'
    macro = dict(category='macro_mean', n_images=expected_count)
    macro.update({key: float(np.mean([row[key] for row in rows])) for key in names[2:]})
    for destination in (output / 'mean_metrics.csv', evidence / 'mean_metrics.csv'):
        with destination.open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=names); writer.writeheader(); writer.writerow(macro)
    environment.update(status='completed', wall_seconds=time.monotonic()-started,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved())
    write_environment()
    print('ALL CATEGORIES COMPLETE', json.dumps(macro), flush=True)
except Exception as exc:
    sys.settrace(None)
    environment.update(status='failed', error=repr(exc), wall_seconds=time.monotonic()-started)
    write_environment()
    raise

subprocess.run([sys.executable, str(Path(__file__).resolve().parent / 'compact_aprilgan_cache.py'),
                '--tag', opts.tag], check=True)
