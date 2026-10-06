"""Run pinned official MuSc with raw evidence capture; no scoring changes."""
import argparse, csv, hashlib, importlib.metadata, json, os, platform, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
import yaml

p = argparse.ArgumentParser()
p.add_argument('--repo', required=True, help='Path to the pinned official MuSc checkout.')
p.add_argument('--data', required=True, help='Path to the MVTec AD dataset root.')
p.add_argument('--dataset', default='mvtec_ad')
p.add_argument('--category', default='bottle')
p.add_argument('--output', required=True)
a = p.parse_args()
repo, output = Path(a.repo), Path(a.output)
if a.category.lower() == 'all':
    p.error('Use one category per output directory to preserve raw evidence.')
if (output / 'metrics.csv').exists():
    p.error('Completed output already exists; choose a fresh output directory.')
output.mkdir(parents=True, exist_ok=True)
os.chdir(repo)
sys.path.insert(0, str(repo))
from models import musc as module

cfg = yaml.safe_load((repo / 'configs/musc.yaml').read_text())
cfg['datasets'].update(dataset_name=a.dataset, data_path=a.data, class_name=a.category, img_resize=518, divide_num=1)
cfg['models'].update(backbone_name='ViT-L-14-336', pretrained='openai', batch_size=4, feature_layers=[5,11,17,23], r_list=[1,3,5])
cfg['device'] = '0'
cfg['testing'].update(output_dir=str(output / 'official'), vis=True, save_excel=True)
(output / 'config.json').write_text(json.dumps(cfg, indent=2))
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
packages = {}
for name in ['torch','torchvision','timm','numpy','scikit-learn','scikit-image','ftfy','opencv-python','openpyxl']:
    try: packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: packages[name] = None
metadata = dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                python=sys.version, platform=platform.platform(), packages=packages,
                gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda, seed=seed,
                tf32=False, command=sys.argv, scope='one complete category test pool')
(output / 'environment.json').write_text(json.dumps(metadata, indent=2))
original_metrics = module.compute_metrics
original_visualization = module.MuSc.visualization
def capture_metrics(gt_sp, pr_sp, gt_px, pr_px):
    # Save the exact arrays passed to upstream metrics, before computing them.
    np.savez_compressed(output / 'raw_predictions.npz', image_labels=gt_sp,
                        image_scores=pr_sp, masks=gt_px, anomaly_maps=pr_px)
    assert np.isfinite(pr_sp).all() and np.isfinite(pr_px).all()
    print('Raw predictions saved; computing official metrics', flush=True)
    result = original_metrics(gt_sp, pr_sp, gt_px, pr_px)
    names = ['image_auroc','image_f1_max','image_ap','pixel_auroc','pixel_f1_max','pixel_ap','aupro']
    values = [float(v) for group in result for v in group]
    with (output / 'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset','category','n_images']+names)
        writer.writeheader()
        writer.writerow(dict(dataset=a.dataset, category=a.category, n_images=len(gt_sp), **dict(zip(names,values))))
    return result
def capture_visualization(self, paths, labels, maps, category):
    data = np.load(output / 'raw_predictions.npz')
    with (output / 'image_scores.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path','label','score_before_rscin','score_after_rscin','sha256'])
        before = maps.reshape(len(paths), -1).max(-1)
        for i,path in enumerate(paths):
            writer.writerow([path,int(labels[i]),float(before[i]),float(data['image_scores'][i]),hashlib.sha256(Path(path).read_bytes()).hexdigest()])
    return original_visualization(self, paths, labels, maps, category)
module.compute_metrics = capture_metrics
module.MuSc.visualization = capture_visualization
start = time.perf_counter()
model = module.MuSc(cfg, seed=seed)
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
model.main()
torch.cuda.synchronize()
metadata.update(wall_seconds_including_model_loading_and_metrics=time.perf_counter()-start,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(), status='completed')
(output / 'environment.json').write_text(json.dumps(metadata, indent=2))
print('COMPLETED', output, flush=True)
