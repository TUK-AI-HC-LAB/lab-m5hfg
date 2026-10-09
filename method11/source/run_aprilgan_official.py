"""Run pinned official APRIL-GAN zero-shot with evidence capture, no scoring edits."""
import argparse
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
parser.add_argument('--seed', type=int, default=42)
opts = parser.parse_args()
repo, data, output = Path(opts.repo), Path(opts.data), Path(opts.output)
evidence = Path(__file__).resolve().parent / 'results/mvtec_zero_shot_20261006'
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
categories = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
              'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
assert set(meta['test']) == set(categories)
assert sum(len(rows) for rows in meta['test'].values()) == 1725
audit = []
for category in categories:
    entries = meta['test'][category]
    expected = {str(p.relative_to(data)): int(p.parent.name != 'good')
                for p in (data / category / 'test').glob('*/*.png')}
    actual = {row['img_path']: int(row['anomaly']) for row in entries}
    assert actual == expected, category
    for row in entries:
        if row['anomaly']:
            assert (data / row['mask_path']).is_file()
    audit.append(dict(category=category, n_images=len(entries), n_normal=sum(v == 0 for v in actual.values()),
                      n_anomaly=sum(v == 1 for v in actual.values()), paths_and_labels_validated=True))
with (evidence / 'dataset_audit.csv').open('w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=list(audit[0])); writer.writeheader(); writer.writerows(audit)

args = SimpleNamespace(data_path=str(data), save_path=str(output / 'official'),
                       checkpoint_path=str(repo / 'exps/pretrained/visa_pretrained.pth'),
                       config_path=str(repo / 'open_clip/model_configs/ViT-L-14-336.json'),
                       dataset='mvtec', model='ViT-L-14-336', pretrained='openai',
                       features_list=[6, 12, 18, 24], few_shot_features=[3, 6, 9], image_size=518,
                       mode='zero_shot', k_shot=0, seed=opts.seed)
official.setup_seed(args.seed)
# Match the existing MuSc experiment's explicitly disabled TF32 condition.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
environment = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                   official_repo='https://github.com/ByChelsea/VAND-APRIL-GAN', python=sys.version,
                   platform=platform.platform(), gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                   packages={name: importlib.metadata.version(name) for name in
                             ['torch', 'torchvision', 'numpy', 'timm', 'scikit-learn', 'scikit-image', 'Pillow']},
                   args=vars(args), checkpoint_training_dataset='VisA', target_dataset='MVTec AD',
                   checkpoint_sha256=hashlib.sha256(Path(args.checkpoint_path).read_bytes()).hexdigest(),
                   metadata_sha256=hashlib.sha256((data / 'meta.json').read_bytes()).hexdigest(),
                   tf32=False, target_training=False, source_edits=False, status='running',
                   command=sys.argv, raw_output=str(output))
def write_environment():
    encoded = json.dumps(environment, indent=2)
    (output / 'environment.json').write_text(encoded)
    (evidence / 'environment.json').write_text(encoded)
write_environment()
(evidence / 'config.json').write_text(json.dumps(vars(args), indent=2))

original_dataset = official.MVTecDataset
dataset_entries = []
class CapturedDataset(original_dataset):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        dataset_entries.extend(self.data_all)
official.MVTecDataset = CapturedDataset

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
    official.test(args)
    assert len(rows) == 15 and {r['category'] for r in rows} == set(categories)
    assert sum(row['n_images'] for row in rows) == 1725
    assert subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip() == ''
    macro = dict(category='macro_mean', n_images=1725)
    macro.update({key: float(np.mean([row[key] for row in rows])) for key in names[2:]})
    for destination in (output / 'mean_metrics.csv', evidence / 'mean_metrics.csv'):
        with destination.open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=names); writer.writeheader(); writer.writerow(macro)
    environment.update(status='completed', wall_seconds=time.monotonic()-started,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved())
    write_environment()
    print('ALL 15 COMPLETE', json.dumps(macro), flush=True)
except Exception as exc:
    environment.update(status='failed', error=repr(exc), wall_seconds=time.monotonic()-started)
    write_environment()
    raise
