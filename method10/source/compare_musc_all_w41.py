"""Compare all measured official/common MuSc categories and unweighted macro means."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/w41_official_vs_common_all_20261006'

def read(path):
    with path.open(encoding='utf-8-sig') as file:
        return list(csv.DictReader(file))

official = read(ROOT / 'results/mvtec_all_paper_20261002/category_metrics.csv')
status = read(OUT / 'status.csv')
assert len(status) == len(official) == 15
assert all(row['status'] in ('completed', 'reused') for row in status)
comparisons, common_rows, remaining_metrics = [], [], []
for old in official:
    category = old['category']
    new = read(OUT / category / 'results_musc.csv')[0]
    assert new['dataset_name'] == 'mvtec_' + category and int(new['seed']) == 42
    common_rows.append(new)
    ia, pa = float(old['image_auroc']), float(old['pixel_auroc'])
    ci, cp = float(new['auroc_mean']), float(new['pixel_auroc_mean'])
    comparisons.append(dict(category=category, n_images=int(old['n_images']),
                            official_image_auroc=ia, common_image_auroc=ci,
                            image_delta_pp=100*(ci-ia), official_pixel_auroc=pa,
                            common_pixel_auroc=cp, pixel_delta_pp=100*(cp-pa)))
    remaining_metrics.append(dict(category=category, official_image_f1_max=old['image_f1_max'],
                                  official_image_ap=old['image_ap'], official_pixel_f1_max=old['pixel_f1_max'],
                                  official_pixel_ap=old['pixel_ap'], official_aupro=old['aupro'],
                                  common_saliency_cr_f1=new['sal_f1_mean'], common_pro_sentinel=new['pro_mean'],
                                  note='Different metrics: no F1-max vs saliency F1 delta'))

assert sum(row['n_images'] for row in comparisons) == 1725
def write(name, rows):
    with (OUT / name).open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

write('category_comparison.csv', comparisons)
write('common_category_metrics.csv', common_rows)
write('nonmatching_metrics.csv', remaining_metrics)
macro = dict(category='macro_mean', n_categories=15, n_images=1725)
for key in comparisons[0]:
    if key not in ('category', 'n_images'):
        macro[key] = sum(row[key] for row in comparisons) / 15
write('macro_comparison.csv', [macro])
tolerance_pp = 0.0001
macro['equal_tolerance_pp'] = tolerance_pp
for name, field in [('image', 'image_delta_pp'), ('pixel', 'pixel_delta_pp')]:
    values = [row[field] for row in comparisons]
    macro[name + '_higher_count'] = sum(v > tolerance_pp for v in values)
    macro[name + '_lower_count'] = sum(v < -tolerance_pp for v in values)
    macro[name + '_same_count'] = sum(abs(v) <= tolerance_pp for v in values)
    largest = max(comparisons, key=lambda row: abs(row[field]))
    macro[name + '_largest_absolute_change'] = {'category': largest['category'], 'delta_pp': largest[field]}
(OUT / 'summary.json').write_text(json.dumps(macro, indent=2))

table = ['| Category | Test 수 | 공식 Image AUROC (%) | 공통 Image AUROC (%) | Image Δ (%p) | 공식 Pixel AUROC (%) | 공통 Pixel AUROC (%) | Pixel Δ (%p) |',
         '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for row in comparisons + [macro]:
    table.append('| {} | {} | {:.4f} | {:.4f} | {:+.4f} | {:.4f} | {:.4f} | {:+.4f} |'.format(
        row['category'], row['n_images'], row['official_image_auroc']*100,
        row['common_image_auroc']*100, 0.0 if abs(row['image_delta_pp']) <= tolerance_pp else row['image_delta_pp'], row['official_pixel_auroc']*100,
        row['common_pixel_auroc']*100, 0.0 if abs(row['pixel_delta_pp']) <= tolerance_pp else row['pixel_delta_pp']))
(OUT / 'category_table.md').write_text('\n'.join(table)+'\n')
print('\n'.join(table))
print(json.dumps(macro, indent=2))
