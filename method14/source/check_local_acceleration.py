"""Reproducible FP32/BF16 representative local-step acceleration check."""
import os
os.environ['IGD_RUN_TAG']='mvtec_accelerated_seed42_20261007'
os.environ['IGD_ACCELERATED']='1'
import gc
import json
import time
import torch
from common import prepare,setup,OUT
from accelerate_igd import fast_model
prepare();lib=setup()
from p32.ssim_module import twoin1Generator,VisualDiscriminator
results=[]
for accelerated in [False,True]:
    torch.backends.cuda.matmul.allow_tf32=accelerated;torch.backends.cudnn.allow_tf32=accelerated
    torch.backends.cudnn.benchmark=accelerated;torch.manual_seed(42)
    g=twoin1Generator(64,latent_dimension=128).cuda();d=VisualDiscriminator(64).cuda()
    if accelerated:fast_model(g);fast_model(d)
    og=torch.optim.Adam(g.parameters(),lr=1e-4,betas=(0.,.9),fused=accelerated)
    od=torch.optim.Adam(d.parameters(),lr=1e-4,betas=(0.,.9),fused=accelerated)
    x=torch.randn(450,3,32,32,device='cuda');center=torch.zeros(128,device='cuda');sigma=torch.tensor(10.,device='cuda')
    times=[];losses=[]
    for step in range(4):
        torch.cuda.synchronize();start=time.perf_counter();og.zero_grad(set_to_none=True)
        z=g.encoder(x);recon=g(x)
        rec=.85*(1-lib.ms_ssim(x,recon,data_range=4.7579,size_average=True,win_size=3,weights=[.0516,.3295,.3463,.2726]))+.15*torch.nn.functional.l1_loss(x,recon)/4.7579
        alpha=torch.rand(450,1,device='cuda')*.5;e2=alpha*z+(1-alpha)*z.flip(0)
        generated=g.generate(e2)
        loss=rec+(1-torch.exp(-((z-center)**2).sum(1)/sigma)).mean()+.1*(d(generated)**2).mean()
        assert torch.isfinite(loss);loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in g.parameters() if p.grad is not None)
        og.step();od.zero_grad(set_to_none=True)
        gen=g.generate(e2).detach();reconstruction=g(x).detach()
        dl=((d(gen)-alpha)**2).mean()+(d(reconstruction+.2*(x-reconstruction))**2).mean()
        dl.backward();assert all(torch.isfinite(p.grad).all() for p in d.parameters() if p.grad is not None)
        od.step();torch.cuda.synchronize();times.append(time.perf_counter()-start);losses.append(float(loss.detach()))
    results.append(dict(accelerated=accelerated,mean_step_seconds=sum(times[1:])/3,losses=losses,finite_gradients=True))
    del g,d,og,od,x,z,recon,generated,loss,rec,gen,reconstruction,dl,e2;gc.collect();torch.cuda.empty_cache()
(OUT/'acceleration_preflight.json').write_text(json.dumps(dict(status='passed',results=results,
    speedup=results[0]['mean_step_seconds']/results[1]['mean_step_seconds'],
    conditions='same random patch batch; warm-up excluded; finite gradients checked; not full-run ETA'),indent=2))
