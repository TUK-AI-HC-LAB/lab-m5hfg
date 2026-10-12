"""User-authorized cleanup of completed CutPaste caches and trained weights only."""
import argparse,csv,hashlib,json
from datetime import datetime,timezone,timedelta
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

RAW=Path('/home/test/cutpaste_results/mvtec_scratch3way_seed42_20261008')
OUT=Path(__file__).resolve().parent/'results/mvtec_scratch3way_seed42_20261008'
NAMES={'train_features.npy','test_features.npy','density.npz','last.pt','density_mean.npy','density_precision.npy','density_shrinkage.npy'}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def no_active_cutpaste():
    names={'run_cutpaste.py','evaluate_cutpaste.py','finish_cutpaste.py','benchmark_acceleration.py','benchmark_data_pipeline.py'}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:args=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        assert not any(Path(a).name in names for a in args),f'Active CutPaste process {proc.name}'

def main(execute):
    no_active_cutpaste()
    assert RAW.resolve()==RAW and not RAW.is_symlink()
    assert json.loads((OUT/'verification.json').read_text())['status']=='passed'
    assert json.loads((OUT/'environment.json').read_text())['status']=='completed'
    rows=list(csv.DictReader((OUT/'category_metrics.csv').open()));assert len(rows)==15
    files=[p for p in RAW.rglob('*') if p.is_file()]
    assert all(not p.is_symlink() and p.resolve().is_relative_to(RAW) for p in files)
    assert all(p.name=='predictions.npz' or p.name in NAMES for p in files)
    removed=[];retained=[]
    # Validate the irreplaceable retained outputs before deleting any model or cache.
    for row in rows:
        cat=row['category'];path=RAW/cat/'predictions.npz'
        manifest=json.loads((OUT/cat/'raw_manifest.json').read_text())
        digest=next(a['sha256'] for a in manifest if a['path']==str(path))
        assert sha(path)==digest
        with np.load(path) as z:
            labels,scores,maps,masks=[z[k] for k in ['labels','image_scores','score_maps','masks']]
            assert len(labels)==int(row['n_images']) and maps.shape==masks.shape
            assert np.isfinite(scores).all() and np.isfinite(maps).all()
            image_auc=float(roc_auc_score(labels,scores));pixel_auc=float(roc_auc_score(masks.ravel(),maps.ravel()))
            assert abs(image_auc-float(row['image_auroc']))<1e-12
            assert abs(pixel_auc-float(row['pixel_auroc']))<1e-12
        retained.append(dict(category=cat,path=str(path),sha256=digest,bytes=path.stat().st_size,image_auroc=image_auc,pixel_auroc=pixel_auc))
        for artifact in manifest:
            p=Path(artifact['path'])
            if p.name in NAMES:
                assert p.resolve().is_relative_to(RAW) and not p.is_symlink()
                if p.exists():removed.append(dict(path=str(p),bytes=p.stat().st_size,previous_sha256=artifact['sha256']))
        print('PRESERVED RESULT VERIFIED',cat,flush=True)
    assert set(a['path'] for a in removed)=={str(p) for p in files if p.name in NAMES}
    report=dict(timestamp=datetime.now(timezone(timedelta(hours=9))).isoformat(),status='planned',
        scope=str(RAW),authorization='User requested delete everything except results needed for CutPaste paper reproduction.',
        removed=removed,retained=retained,deleted_bytes=sum(a['bytes'] for a in removed),remaining_raw_bytes=sum(a['bytes'] for a in retained),
        metrics_and_predictions_reverified=True,paper_code_config_logs_csv_json_preserved=True,
        limitation='Historical full verification remains unchanged. Full finish_cutpaste.py cannot run again without regenerating deleted features/densities and retraining deleted models. Retained predictions suffice to recompute image/pixel AUROC and figures. No other method/dataset/environment touched; no WSL shutdown or VHD compaction.')
    target=OUT/'storage_cleanup_20261010.json'
    target.write_text(json.dumps(report,indent=2))
    if execute:
        no_active_cutpaste()
        for item in removed:
            p=Path(item['path'])
            assert p.name in NAMES and p.resolve().is_relative_to(RAW) and not p.is_symlink()
            p.unlink()
        assert all(Path(a['path']).exists() for a in retained)
        assert all(not Path(a['path']).exists() for a in removed)
        report['status']='completed';target.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ['status','deleted_bytes','remaining_raw_bytes']},indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');main(parser.parse_args().execute)
