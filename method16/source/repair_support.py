"""Generate documented fixed normal support only for empty capsule/grid files."""
import json
import random
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from prepare_regad import ASSETS,OUT,sha
transform=transforms.Compose([transforms.Resize(224,Image.Resampling.LANCZOS),transforms.ToTensor()])
overrides=[]
OUT.mkdir(parents=True,exist_ok=True)
for category in ['capsule','grid']:
    paths=sorted((Path('/home/test/data/mvtec')/category/'train/good').glob('*.png'))
    for shot in [4,2,8]:
        original=ASSETS/'support_set'/category/f'{shot}_10.pt'
        if original.stat().st_size!=0:continue
        folder=ASSETS/'local_support_set'/category;folder.mkdir(parents=True,exist_ok=True)
        output=folder/f'{shot}_10.pt';rng=random.Random(668);supports=[];rounds=[]
        for round_id in range(10):
            selected=rng.sample(paths,shot)
            supports.append(torch.stack([transform(Image.open(p).convert('RGB')) for p in selected]))
            rounds.append(dict(round=round_id,images=[dict(path=str(p),sha256=sha(p)) for p in selected]))
        torch.save(supports,output)
        overrides.append(dict(category=category,shot=shot,original=str(original),original_bytes=0,path=str(output),sha256=sha(output),
            seed=668,source='normal training images only; locally sampled without replacement within each round',rounds=rounds))
(OUT/'support_overrides.json').write_text(json.dumps(overrides,indent=2))
print('6 EMPTY SUPPORT FILES REPLACED BY DOCUMENTED LOCAL SUPPORT')
