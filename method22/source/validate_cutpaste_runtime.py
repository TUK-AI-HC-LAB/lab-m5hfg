"""Actual MVTec feature checks for GPU shrinkage and RF math, not metrics."""
import json,sys
import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from cutpaste_common import *
from evaluate_cutpaste import ProjectionNet,shrunk_precision,kernel,maps_from_scores
torch.set_num_threads(4);torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
model=ProjectionNet(pretrained=False,head_layers=[512,128],num_classes=3).cuda().eval()
rows=manifest('bottle','train')[:16];features=[]
with torch.inference_mode():
    for r in rows:
        x=NORM(image(r['image_path'])).unsqueeze(0).cuda()
        crops=torch.cat([x[:,:,:32,:32],x[:,:,112:144,112:144],x[:,:,-32:,-32:]])
        features.append(model(crops)[0].cpu().numpy())
x=torch.from_numpy(np.stack(features)).cuda().double().permute(1,0,2);mu=x.mean(1);z=x-mu[:,None]
cov=z.transpose(1,2)@z/len(rows);fourth=z.square().sum(-1).square().mean(-1)
inv,s=shrunk_precision(cov,fourth,len(rows));checks=[]
for i in range(3):
    ref=LedoitWolf().fit(x[i].cpu().numpy())
    assert abs(float(s[i].cpu())-ref.shrinkage_)<1e-8
    assert np.allclose(inv[i].cpu().numpy(),ref.precision_,rtol=1e-5,atol=1e-5)
    checks.append(dict(position=i,shrinkage_error=float(abs(float(s[i].cpu())-ref.shrinkage_)),precision_max_error=float(np.max(np.abs(inv[i].cpu().numpy()-ref.precision_)))))
scores=np.linspace(0,1,3249).reshape(1,3249);actual=maps_from_scores(scores)[0];independent=np.zeros((256,256),np.float64);k=kernel().numpy()[0,0]
for h in range(57):
    for w in range(57):independent[h*4:h*4+32,w*4:w*4+32]+=np.float32(scores[0,h*57+w])*k
assert np.allclose(actual,independent,rtol=1e-5,atol=1e-6)
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'runtime_verification.json').write_text(json.dumps(dict(status='passed',untrained_model_architecture_only=True,
    actual_MVTec_train_images=[dict(path=r['image_path'],sha256=r['sha256']) for r in rows],gpu_LedoitWolf_vs_sklearn=checks,
    RF_transposed_convolution_vs_independent_scatter_max_error=float(np.max(np.abs(actual-independent))),
    limitation='initial-feature numerical/API smoke, not learned performance or full precision equivalence'),indent=2))
print('CUTPASTE REAL FEATURE GDE AND RF CHECK PASSED',flush=True)
