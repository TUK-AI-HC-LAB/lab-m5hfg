import sys,json
sys.path.insert(0,'/mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method21/source')
from pyramidflow_common import *
from PIL import Image
items=[]
for r in manifest('02','test'):
 if r['label'] and not mask(r).any():
  a=np.array(Image.open(r['mask_path']).convert('L'))
  items.append(dict(image=r['image_path'],original_max=int(a.max()),original_positive_pixels=int((a>0).sum()),original_gt_sha256=r['mask_sha256']))
(OUT/'02/original_missing_mask_diagnosis.json').write_text(json.dumps(items,indent=2))
print(items)
