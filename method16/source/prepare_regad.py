"""Pinned RegAD runtime; compatibility and documented equivalent batched scoring."""
import difflib
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=Path('/home/test/RegAD')
ASSETS=Path('/home/test/regad_assets')
RAW=Path('/home/test/regad_results/mvtec_public_20261007')
OUT=ROOT/'results/mvtec_public_20261007'
RUNTIME=RAW/'runtime'
COMMIT='5e2c1f8c18d302b0354471567846fee3ed2ff063'
KEYS=['image_auroc','image_f1_max','image_ap','pixel_auroc','pixel_f1_max','pixel_ap','aupro']
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def prepare():
    OUT.mkdir(parents=True,exist_ok=True);RAW.mkdir(parents=True,exist_ok=True)
    diffs=[];manifest=[]
    for path in sorted(REPO.rglob('*.py')):
        rel=path.relative_to(REPO);before=path.read_text();after=before.replace('Image.ANTIALIAS','Image.Resampling.LANCZOS')
        if rel.as_posix()=='datasets/mvtec.py':
            # Literal 'test' also appears in /home/test; replace category phase only.
            after=after.replace("image_dir_one.replace('test', 'ground_truth')","image_dir_one.replace('/'+self.class_name+'/test/', '/'+self.class_name+'/ground_truth/')")
        if rel.as_posix()=='test.py':
            start=after.index('    dist_list = []')
            end=after.index('\n\n    # upsample',start)
            after=after[:start]+"    from accelerate_regad import mahalanobis_maps\n    dist_list = mahalanobis_maps(embedding_vectors, train_outputs[0], train_outputs[1], B, H, W)"+after[end:]
        target=RUNTIME/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(after)
        manifest.append(dict(path=str(rel),official_sha256=sha(path),runtime_sha256=sha(target)))
        if before!=after:diffs.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=str(rel),tofile='runtime/'+str(rel)))
    (OUT/'compatibility.patch').write_text(''.join(diffs));(OUT/'runtime_sources.json').write_text(json.dumps(manifest,indent=2))
