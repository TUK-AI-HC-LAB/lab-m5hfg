"""Validate all official support/checkpoint files before multi-category evaluation."""
import json
import torch
from prepare_regad import ASSETS,OUT,sha
names=['bottle','cable','capsule','carpet','grid','hazelnut','leather','metal_nut','pill','screw','tile','toothbrush','transistor','wood','zipper']
records=[];errors=[]
for shot in [4,2,8]:
    for category in names:
        support_path=ASSETS/'support_set'/category/f'{shot}_10.pt'
        try:
            x=torch.load(support_path,map_location='cpu',weights_only=True)
            assert len(x)==10 and all(t.shape==(shot,3,224,224) and torch.isfinite(t).all() for t in x)
            del x
        except Exception as error:
            errors.append(dict(shot=shot,category=category,path=str(support_path),error=repr(error),bytes=support_path.stat().st_size))
        checkpoint=ASSETS/'save_checkpoints'/str(shot)/category/f'{category}_{shot}_rotation_scale_model.pt'
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert set(state)=={'STN','ENC','PRED'}
        records.append(dict(shot=shot,category=category,support_sha256=sha(support_path),checkpoint_sha256=sha(checkpoint)))
        del state
(OUT/'public_assets_verification.json').write_text(json.dumps(dict(status='passed' if not errors else 'invalid_support_files',n_conditions=3,n_categories=15,files=records,errors=errors),indent=2))
print(json.dumps(errors) if errors else 'ALL45 PUBLIC SUPPORT/CHECKPOINT PAIRS VERIFIED')
