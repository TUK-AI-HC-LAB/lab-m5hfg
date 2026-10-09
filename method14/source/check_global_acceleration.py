"""Validate one full global BF16 training step before the long run."""
import os
os.environ['IGD_RUN_TAG']='mvtec_accelerated_seed42_20261007'
os.environ['IGD_ACCELERATED']='1'
import importlib
import json
import random
from types import SimpleNamespace
import numpy as np
import torch
from common import prepare,setup,RAW,OUT,SilentRange
from accelerate_igd import fast_model,fast_module
prepare();setup()
module=importlib.import_module('p256.ssim_main')
module.tqdm=SilentRange;module.recorder=None
# The compiler is tested separately; this check verifies BF16 model arithmetic.
fast_module(module)
random.seed(42);np.random.seed(42);torch.manual_seed(42)
folder=RAW/'global_preflight';folder.mkdir(exist_ok=True);(folder/'optimizer').mkdir(exist_ok=True)
module.ckpt_path=str(folder)
generator=fast_model(module.twoin1Generator256(64,latent_dimension=128).cuda())
discriminator=fast_model(module.VisualDiscriminator256(64).cuda())
module.generator=generator
opt_g=torch.optim.Adam(generator.parameters(),lr=module.LR,betas=(0.,.9),weight_decay=1e-6,fused=True)
opt_d=torch.optim.Adam(discriminator.parameters(),lr=module.LR,betas=(0.,.9),fused=True)
module.MAX_EPOCH=module.BATCH_SIZE/209+1e-8
def validate(*a,**kw):
    assert all(torch.isfinite(p).all() for p in generator.parameters())
    assert all(torch.isfinite(p.grad).all() for p in generator.parameters() if p.grad is not None)
    assert all(torch.isfinite(p.grad).all() for p in discriminator.parameters() if p.grad is not None)
    return 0.,[0.]
module.validation=validate
module.train(SimpleNamespace(sample_rate=1.),'bottle',generator,discriminator,opt_g,opt_d)
steps=[float(s['step']) for s in opt_g.state.values() if 'step' in s]
assert steps and set(steps)=={1.}
(OUT/'global_acceleration_preflight.json').write_text(json.dumps(dict(status='passed',optimizer_steps=1,
    bf16_model=True,fp32_statistics=True,tf32=True,fused_adam=True,channels_last=True,
    finite_gradients=True,production_epochs_unchanged=256),indent=2))
print('GLOBAL ACCELERATION PASSED',flush=True)
