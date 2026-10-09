"""Remove duplicate inference cache only after exact comparison to final artifacts."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument('--tag', required=True)
args = p.parse_args()
raw = Path('/home/test/aprilgan_results') / args.tag
evidence = ROOT / 'results' / args.tag
assert json.loads((evidence / 'environment.json').read_text())['status'] == 'completed'
cache = raw / 'inference_cache'
manifest = evidence / 'duplicate_cache_verification.json'
if manifest.exists() and json.loads(manifest.read_text())['status'] == 'passed':
    raise SystemExit(0)
entries = [json.loads(line) for line in (cache / 'images.jsonl').read_text().splitlines()]
verified = []
for category in dict.fromkeys(e['entry']['cls_name'] for e in entries):
    selected = [e for e in entries if e['entry']['cls_name'] == category]
    with np.load(raw / category / 'raw_predictions.npz') as z:
        for name, key in [('masks', 'masks'), ('maps', 'anomaly_maps')]:
            final = z[key]
            assert len(final) == len(selected)
            for i, entry in enumerate(selected):
                path = cache / name / f"{entry['index']:05d}.npy"
                original = np.load(path).reshape(final[i].shape)
                assert original.dtype == final.dtype and np.array_equal(original, final[i]), (category, name, i)
                verified.append(path)
            del final
    final_features = np.load(raw / category / 'clip_image_features.npy')
    for i, entry in enumerate(selected):
        path = cache / 'features' / f"{entry['index']:05d}.npy"
        original = np.load(path).reshape(final_features[i].shape)
        assert original.dtype == final_features.dtype and np.array_equal(original, final_features[i])
        verified.append(path)
    del final_features
released = sum(path.stat().st_size for path in verified)
manifest.write_text(json.dumps(dict(status='passed', n_images=len(entries),
    exact_dtype_shape_value_comparison=True, duplicate_files=len(verified), duplicate_bytes=released,
    retained='category raw_predictions.npz, clip_image_features.npy, images.jsonl; only duplicate .npy cache removed'), indent=2))
for path in verified:
    assert path.resolve().is_relative_to(cache.resolve()) and path.suffix == '.npy'
    path.unlink()
print('Verified and removed duplicate cache:', args.tag, released, 'bytes', flush=True)
