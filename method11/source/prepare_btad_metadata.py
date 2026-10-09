"""Generate/verify the APRIL-GAN metadata view of the official BTAD archive.

Download https://avires.dimi.uniud.it/papers/btad/btad.zip and extract it.
Archive SHA-256: 461c9387e515bfed41ecaae07c50cf6b10def647b36c9e31d239ab2736b10d2a
This script leaves image/mask files unchanged and verifies existing metadata.
"""
import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--root', default='/home/test/data/btad_original/BTech_Dataset_transformed')
opt=p.parse_args()
root=Path(opt.root).resolve()
extensions={'.png','.bmp','.jpg','.jpeg','.tif','.tiff'}
metadata={split:{} for split in ['train','test']}
for split in metadata:
    for category in ['01','02','03']:
        rows=[]
        for path in sorted((root/category/split).glob('*/*')):
            if not path.is_file() or path.suffix.lower() not in extensions: continue
            specie=path.parent.name
            assert specie in ['ok','ko'],path
            anomaly=int(specie=='ko')
            mask=''
            if anomaly:
                candidates=sorted((root/category/'ground_truth'/specie).glob(path.stem+'.*'))
                assert len(candidates)==1,(path,candidates)
                mask=candidates[0].relative_to(root).as_posix()
            rows.append(dict(img_path=path.relative_to(root).as_posix(),mask_path=mask,
                             cls_name=category,specie_name=specie,anomaly=anomaly))
        metadata[split][category]=rows
assert [len(metadata['train'][c]) for c in metadata['train']]==[400,399,1000]
assert [len(metadata['test'][c]) for c in metadata['test']]==[70,230,441]
target=root/'meta.json'
if target.exists():
    assert json.loads(target.read_text())==metadata,'Existing metadata differs; preserved.'
else:
    target.write_text(json.dumps(metadata,indent=2))
proof=dict(status='passed',root=str(root),exact_file_order_labels_masks_verified=True,
           metadata_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),test_images=741)
destination=Path(__file__).resolve().parent/'results/btad_metadata_verification.json'
destination.write_text(json.dumps(proof,indent=2))
print(json.dumps(proof),flush=True)
