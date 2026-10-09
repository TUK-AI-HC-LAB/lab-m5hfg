"""Train missing small-backbone projection with official auxiliary-data trainer."""
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import torch

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/train_vit_b16_plus_240_20261006';OUT.mkdir(exist_ok=True)
raw=Path('/home/test/aprilgan_results/train_vit_b16_plus_240_20261006')
if (OUT/'environment.json').exists(): raise SystemExit('Existing training evidence preserved.')
repo=Path('/home/test/VAND-APRIL-GAN');os.chdir(repo);sys.path.insert(0,str(repo));sys.dont_write_bytecode=True
import train as official
original_source=Path(official.__file__).read_text()
old='patch_tokens[layer] /= patch_tokens[layer].norm(dim=-1, keepdim=True)'
new='patch_tokens[layer] = (patch_tokens[layer] / patch_tokens[layer].norm(dim=-1, keepdim=True)).to(patch_tokens[layer].dtype)'
assert original_source.count(old)==1
adapted=original_source.replace(old,new)
(OUT/'train_autograd_compatibility.py').write_text(adapted)
# Modern autograd rejects the upstream in-place division. Out-of-place division
# preserves forward values and gradient dependency; official checkout is intact.
exec(compile(adapted,str(OUT/'train_autograd_compatibility.py'),'exec'),official.__dict__)
args=SimpleNamespace(dataset='visa',train_data_path='/home/test/data/VisA_20220922',save_path=str(raw),
    config_path=str(repo/'open_clip/model_configs/ViT-B-16-plus-240.json'),model='ViT-B-16-plus-240',
    features_list=[3,6,9,12],pretrained='laion400m_e31',image_size=518,batch_size=8,
    epoch=15,learning_rate=0.001,aug_rate=0.2,print_freq=1,save_freq=1)
official.setup_seed(111);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
meta=json.loads(Path(args.train_data_path,'meta.json').read_text())
# Official train() uses dataset's default mode='test': auxiliary labeled test split.
assert sum(map(len,meta['test'].values()))==2162
env=dict(status='running',args=vars(args),seed=111,auxiliary_split='VisA test split with annotations; official trainer default',
    auxiliary_images=2162,target_dataset='MVTec AD',target_images_used_for_training=False,
    official_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
    protocol_status='official trainer and VisA 15-epoch settings extended to small backbone; original Table 13 checkpoint unavailable',
    inference_protocol_derived=True,
    compatibility_change=dict(old=old,new=new,reason='modern Torch autograd rejects upstream in-place normalization',
       adapted_source_sha256=hashlib.sha256(adapted.encode()).hexdigest()))
def save(): (OUT/'environment.json').write_text(json.dumps(env,indent=2))
save();started=time.monotonic()
try:
    official.train(args)
    weights=raw/'epoch_15.pth';assert weights.exists()
    state=torch.load(weights,map_location='cpu',weights_only=True)['trainable_linearlayer']
    assert all(torch.isfinite(v).all() for v in state.values()),'Non-finite trained checkpoint'
    epoch_logs=re.findall(r'epoch \[(\d+)/15\], loss:([^\s]+)',(raw/'log.txt').read_text())
    assert len(epoch_logs)==15 and all(float(loss)==float(loss) and abs(float(loss))<float('inf') for epoch,loss in epoch_logs)
    env.update(status='completed',wall_seconds=time.monotonic()-started,checkpoint=str(weights),
               checkpoint_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
               checkpoint_all_tensors_finite=True,epoch_losses=epoch_logs)
    save();print('TRAINING COMPLETE',json.dumps(env),flush=True)
except Exception as exc:
    env.update(status='failed',error=repr(exc),wall_seconds=time.monotonic()-started);save();raise
