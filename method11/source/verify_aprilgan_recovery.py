"""Verify exact cached arrays against the preserved pre-streaming Visa attempt."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
RAW=Path('/home/test/aprilgan_results')
tag='visa_0shot_seed42_20261006'
state=ROOT/'results'/tag/'environment.json'
assert state.exists() and json.loads(state.read_text())['status'] in ['running','completed']
with (ROOT/'results'/(tag+'_interrupted1')/'category_metrics.csv').open() as f:
    old_rows=list(csv.DictReader(f))
assert len(old_rows)>0
with (ROOT/'results'/tag/'category_metrics.csv').open() as f:
    completed={r['category'] for r in csv.DictReader(f)}
selected=[row for row in old_rows if row['category'] in completed]
assert selected,'No completed overlapping category yet.'
records=[]
for row in selected:
    category=row['category']
    before=RAW/(tag+'_interrupted1')/category/'raw_predictions.npz'
    after=RAW/tag/category/'raw_predictions.npz'
    record=dict(category=category)
    with np.load(before) as old,np.load(after) as new:
        for key in ['image_labels','image_scores','masks','anomaly_maps']:
            a,b=old[key],new[key]
            identical=a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b)
            record[key+'_identical']=identical
            assert identical,(category,key,a.shape,b.shape)
            del a,b
    records.append(record)
destination=ROOT/'results'/'streaming_recovery_verification.json'
destination.write_text(json.dumps(dict(status='passed' if len(selected)==len(old_rows) else 'partial_passed',baseline='interrupted original in-memory Visa attempt',
    recovered='serial exact disk-backed arrays',categories_verified=len(records),expected_categories=len(old_rows),records=records),indent=2))
print('Recovery equivalence verified for',len(records),'categories.',flush=True)
