"""Same inverse covariance Mahalanobis formula, vectorized over images/positions."""
import json
import torch
from prepare_regad import OUT
def mahalanobis_maps(embedding,mean,cov,B,H,W):
    result=torch.empty(B,H*W,device='cpu')
    verified=False
    for start in range(0,H*W,16):
        end=min(start+16,H*W)
        inverse=torch.linalg.inv(cov[:,:,start:end].permute(2,0,1).contiguous())
        delta=(embedding[:,:,start:end]-mean[:,start:end].unsqueeze(0)).permute(2,0,1).contiguous()
        distances=(torch.bmm(delta,inverse)*delta).sum(2).sqrt().transpose(0,1)
        assert torch.isfinite(distances).all()
        if not verified and not (OUT/'mahalanobis_preflight.json').exists():
            reference=[];batched=[]
            for p in range(min(3,end-start)):
                native_inv=torch.linalg.inv(cov[:,:,start+p])
                for i in range(min(8,B)):
                    d=embedding[i,:,start+p]-mean[:,start+p]
                    reference.append(torch.dot(d,torch.matmul(native_inv,d)).sqrt())
                    batched.append(distances[i,p])
            reference=torch.stack(reference);batched=torch.stack(batched)
            max_error=float((reference-batched).abs().max())
            assert torch.allclose(reference,batched,rtol=1e-3,atol=1e-3)
            (OUT/'mahalanobis_preflight.json').write_text(json.dumps(dict(status='passed',n_comparisons=len(reference),
                max_absolute_error=max_error,rtol=1e-3,atol=1e-3,scope='real first support covariance/query features; limited positions/images; not bitwise/full-output equivalence'),indent=2))
            verified=True
        result[:,start:end]=distances.cpu()
    return result.reshape(B,H,W)
