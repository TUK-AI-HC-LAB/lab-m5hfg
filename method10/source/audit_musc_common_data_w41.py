"""Verify common loader file/label sets against official image SHA-256 evidence."""
import argparse
import csv
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--framework', type=Path, required=True, help='Common-framework checkout.')
parser.add_argument('--data', type=Path, required=True, help='MVTec AD dataset root.')
args_cli = parser.parse_args()
sys.path.insert(0, str(args_cli.framework))
from main import resolve_args
from datasets.mvtec import MVTecDataset, DatasetSplit

args = resolve_args(['--method', 'musc', '--data-path', str(args_cli.data), '--seed', '42'])
rows = []
for category_dir in sorted((ROOT / 'results/mvtec_all_paper_20261002').iterdir()):
    path = category_dir / 'image_scores.csv'
    if not path.exists():
        continue
    with path.open() as file:
        official = list(csv.DictReader(file))
    dataset = MVTecDataset(source=args.data_path, classname=category_dir.name, resize=args.resize,
                           imagesize=args.imagesize, split=DatasetSplit.TEST, args=args, seed=42)
    common = {row[2]: int(row[1] != 'good') for row in dataset.data_to_iterate}
    expected = {row['image_path']: int(row['label']) for row in official}
    assert common == expected, category_dir.name
    verified = sum(hashlib.sha256(Path(row['image_path']).read_bytes()).hexdigest() == row['sha256'] for row in official)
    assert verified == len(official), category_dir.name
    rows.append(dict(category=category_dir.name, n_images=len(official), same_paths=True,
                     same_labels=True, matched_sha256=verified,
                     same_order=[row[2] for row in dataset.data_to_iterate] == [row['image_path'] for row in official]))
assert len(rows) == 15 and sum(r['n_images'] for r in rows) == 1725
out = ROOT / 'results/w41_official_vs_common_all_20261006/dataset_identity.csv'
with out.open('w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
print('PASS: 15 categories, 1725 matching image hashes and file/label sets')
print('Same order:', sum(r['same_order'] for r in rows), '/ 15')
