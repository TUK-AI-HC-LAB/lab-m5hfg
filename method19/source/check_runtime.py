"""Small real-BTAD check of native encoder/NN APIs before long evaluation."""
import sys,json
import numpy as np
import torch
from psvdd_common import *
sys.path.insert(0,str(REPO))
from codes.networks import EncoderHier
from codes.inspection import infer
from codes.nearest_neighbor import search_NN
from codes.utils import distribute_scores
torch.set_num_threads(8);torch.manual_seed(42)
train=images(image_files('01','train')[:2]);test=images(image_files('01','test')[:1]);mean=train.astype(np.float32).mean(0)
enc=EncoderHier(64,64).cuda().eval();proof=[]
for k,stride,method,encoder in [(64,16,'kdt',enc),(32,4,'ngt',enc.enc)]:
    bank=infer((train.astype(np.float32)-mean)/255,encoder,k,stride).reshape(-1,64)
    queries=infer((test.astype(np.float32)-mean)/255,encoder,k,stride)[:,:1,:16,:]
    distances,indices=search_NN(queries,bank,NN=1,method=method)
    assert np.isfinite(distances).all() and ((indices>=0)&(indices<len(bank))).all()
    retrieved=bank[indices[...,0]]
    direct=np.linalg.norm(queries-retrieved,axis=-1)
    assert np.allclose(distances[...,0],direct,rtol=1e-4,atol=1e-6)
    if method=='kdt':
        brute=np.linalg.norm(queries.reshape(-1,64)[:,None]-bank[None],axis=-1).min(1)
        assert np.allclose(distances.ravel(),brute,rtol=1e-4,atol=1e-6)
    proof.append(dict(patch_size=k,stride=stride,method=method,n_bank=len(bank),n_queries=queries.shape[2],finite=True,
        retrieved_distance_max_abs=float(np.abs(distances[...,0]-direct).max()),exact_nearest_independently_checked=method=='kdt'))
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'native_runtime_verification.json').write_text(json.dumps(dict(status='passed',real_BTAD_images=True,checkpoint='untrained architecture smoke only, not performance evidence',checks=proof),indent=2))
print('NATIVE ENCODER/KDT/NGT REAL-IMAGE CHECK PASSED',flush=True)
