"""Build W41 comparison from measured CSVs; does not rerun or tune MuSc."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--framework', type=Path, required=True, help='Common-framework checkout used for the evaluation.')
parser.add_argument('--common-csv', type=Path, required=True, help='Common-framework bottle result CSV.')
args = parser.parse_args()
FRAMEWORK = args.framework
COMMON = args.common_csv
OUT = ROOT / 'results/w41_official_vs_common_20261006'
OUT.mkdir(parents=True, exist_ok=True)

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))

official = read(ROOT / 'results/bottle_paper_20261002/metrics.csv')[0]
common = read(COMMON)[0]
assert official['category'] == 'bottle' and int(official['n_images']) == 83
assert common['dataset_name'] == 'mvtec_bottle'
assert int(common['seed']) == 42
(OUT / 'common_results_musc.csv').write_bytes(COMMON.read_bytes())
with (OUT / 'comparison.csv').open('w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['category', 'n_images', 'metric', 'official', 'common', 'delta_percentage_points', 'comparison_status'])
    for key, common_key in [('image_auroc', 'auroc_mean'), ('pixel_auroc', 'pixel_auroc_mean')]:
        old, new = float(official[key]), float(common[common_key])
        writer.writerow(['bottle', 83, key, old, new, (new-old)*100, 'different_pipeline_descriptive_only'])
    for key in ('image_f1_max', 'image_ap', 'pixel_f1_max', 'pixel_ap', 'aupro'):
        writer.writerow(['bottle', 83, key, official[key], '', '', 'not_reported_by_common_evaluation'])
    writer.writerow(['bottle', 83, 'saliency_cr_f1', '', common['sal_f1_mean'], '', 'different_from_official_f1_max'])

categories = read(ROOT / 'results/mvtec_all_paper_20261002/category_metrics.csv')
with (OUT / 'category_coverage.csv').open('w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['category', 'n_images', 'official_image_auroc', 'official_pixel_auroc', 'common_image_auroc', 'common_pixel_auroc', 'common_status'])
    for row in categories:
        matched = row['category'] == 'bottle'
        writer.writerow([row['category'], row['n_images'], row['image_auroc'], row['pixel_auroc'],
                         common['auroc_mean'] if matched else '', common['pixel_auroc_mean'] if matched else '',
                         'completed' if matched else 'not_run'])

files = [FRAMEWORK / p for p in ('trainer/trainer_musc.py', 'configs/musc.yaml', 'component_registry.py',
                                'datasets/base.py', 'trainer/trainer.py', 'metrics_gpu.py')]
manifest = {
    'created': '2026-10-06', 'scope': 'bottle only; official 15-category results retained as coverage context',
    'official_commit': 'b76b93da8bd3096a99964a96ae29d46f197a0651',
    'common_commit': None, 'common_commit_note': 'downloaded codebase is not a Git checkout',
    'official_csv': str(ROOT / 'results/bottle_paper_20261002/metrics.csv'),
    'common_csv': str(COMMON), 'common_log': 'not included in repository',
    'common_raw_predictions': None,
    'common_command': 'python main.py --method musc --dataset mvtec --category bottle --data-path <MVTec-root> --results-path <fresh-output-root> --seed 42 --num-workers 1',
    'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files + [COMMON]},
    'caveat': 'Source hashes describe files at report generation, not an immutable snapshot of the earlier run. No causal attribution or statistical significance test.'
}
(OUT / 'provenance.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
print((OUT / 'comparison.csv').read_text())
