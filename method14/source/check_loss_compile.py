"""One-time MS-SSIM compilation check; no runtime ETA tracking."""
import json
import os
import time
os.environ.setdefault('IGD_RUN_TAG','mvtec_accelerated_seed42_20261007')
os.environ['IGD_ACCELERATED']='1'
os.environ['TORCHINDUCTOR_COMPILE_THREADS']='2'
from common import setup,OUT
import torch
fn=setup().ms_ssim
x=torch.randn(450,3,32,32,device='cuda');y=torch.randn_like(x,requires_grad=True)
args=dict(data_range=4.7579,win_size=3,weights=[.0516,.3295,.3463,.2726],size_average=True)
reference=fn(x,y,**args)
try:
    compiled=torch.compile(fn,mode='default',dynamic=False)
    results=[]
    for label,f in [('eager',fn),('compiled',compiled)]:
        durations=[]
        for i in range(4):
            y.grad=None;torch.cuda.synchronize();start=time.perf_counter()
            z=f(x,y,**args);z.backward();torch.cuda.synchronize()
            assert torch.isfinite(z) and torch.isfinite(y.grad).all()
            assert torch.allclose(z,reference,rtol=1e-3,atol=1e-4)
            durations.append(time.perf_counter()-start)
        results.append(dict(mode=label,seconds=sum(durations[1:])/3))
    result=dict(status='passed',results=results,enabled=results[1]['seconds']<results[0]['seconds'],
        scope='representative local MS-SSIM forward/backward; compile warm-up excluded; not total ETA')
except Exception as error:
    result=dict(status='failed',enabled=False,error=repr(error))
(OUT/'compile_preflight.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
