"""Global/position-wise Gaussian scoring and paper RF upsampling."""
import argparse,gc,json,sys
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from cutpaste_common import *
sys.path.insert(0,str(REPO))
from model import ProjectionNet

def kernel():
    t=torch.arange(32,dtype=torch.float32)-15.5
    a=torch.exp(-.5*(t/8)**2);k=a[:,None]*a[None,:]
    return (k/k.sum()).reshape(1,1,32,32)
def maps_from_scores(scores):
    x=torch.from_numpy(np.asarray(scores,dtype=np.float32)).reshape(-1,1,57,57)
    return F.conv_transpose2d(x,kernel(),stride=4).numpy()[:,0]
def shrunk_precision(cov,fourth,n):
    p=cov.shape[-1];mu=cov.diagonal(dim1=-2,dim2=-1).sum(-1)/p
    square=cov.square().sum((-1,-2));delta=(square-p*mu.square())/p
    beta=(fourth-square)/(p*n)
    beta=beta.clamp_min(0);beta=torch.minimum(beta,delta.clamp_min(0))
    shrink=torch.where(delta>0,beta/delta,torch.zeros_like(delta))
    shrunk=(1-shrink[...,None,None])*cov
    shrunk=shrunk+torch.eye(p,device='cuda',dtype=torch.float64)* (shrink*mu)[...,None,None]
    return torch.linalg.pinv(shrunk,hermitian=True),shrink

def fit_patch(features,aligned,folder):
    n,p,d=features.shape
    mean=np.lib.format.open_memmap(folder/'density_mean.npy',mode='w+',dtype='float64',shape=(p if aligned else 1,d))
    precision=np.lib.format.open_memmap(folder/'density_precision.npy',mode='w+',dtype='float64',shape=(p if aligned else 1,d,d))
    shrink=np.lib.format.open_memmap(folder/'density_shrinkage.npy',mode='w+',dtype='float64',shape=(p if aligned else 1,))
    proofs=[]
    if aligned:
        for j in range(0,p,16):
            x=torch.from_numpy(np.array(features[:,j:j+16],copy=True)).cuda().double().permute(1,0,2)
            mu=x.mean(1);z=x-mu[:,None,:];cov=z.transpose(1,2)@z/n;fourth=z.square().sum(-1).square().mean(-1)
            inv,s=shrunk_precision(cov,fourth,n);mean[j:j+16]=mu.cpu().numpy();precision[j:j+16]=inv.cpu().numpy();shrink[j:j+16]=s.cpu().numpy()
        for j in [0,p//2,p-1]:
            ref=LedoitWolf().fit(np.asarray(features[:,j],dtype=np.float64))
            assert np.allclose(mean[j],ref.location_,rtol=1e-9,atol=1e-9)
            assert abs(shrink[j]-ref.shrinkage_)<1e-8
            assert np.allclose(precision[j],ref.precision_,rtol=1e-5,atol=1e-5)
            proofs.append(dict(position=j,shrinkage_error=float(abs(shrink[j]-ref.shrinkage_)),precision_max_error=float(np.max(np.abs(precision[j]-ref.precision_)))))
    else:
        flat=features.reshape(-1,d);count=len(flat);mu=np.zeros(d,np.float64)
        for i in range(0,count,8192):mu+=np.asarray(flat[i:i+8192],dtype=np.float64).sum(0)
        mu/=count;gpu_mu=torch.from_numpy(mu).cuda();cov=torch.zeros(d,d,device='cuda',dtype=torch.float64);fourth=torch.zeros((),device='cuda',dtype=torch.float64)
        for i in range(0,count,8192):
            z=torch.from_numpy(np.array(flat[i:i+8192],copy=True)).cuda().double()-gpu_mu
            cov+=z.T@z;fourth+=z.square().sum(-1).square().sum()
        inv,s=shrunk_precision(cov/count,fourth/count,count);mean[0]=mu;precision[0]=inv.cpu().numpy();shrink[0]=float(s.cpu())
        # Full global fit may contain >1 million patches. Validate the formula
        # against sklearn on actual subset features, without using a subset
        # for the reported full-data density.
        subset=np.asarray(flat[:min(1024,count)],dtype=np.float64);ref=LedoitWolf().fit(subset)
        z=torch.from_numpy(subset).cuda();m=z.mean(0);z=z-m;cov=z.T@z/len(z);fourth=z.square().sum(-1).square().mean()
        inv,s=shrunk_precision(cov,fourth,len(z));assert np.allclose(inv.cpu().numpy(),ref.precision_,rtol=1e-5,atol=1e-5)
        assert abs(float(s.cpu())-ref.shrinkage_)<1e-8
        proofs.append(dict(actual_training_subset=len(subset),full_reported_fit_samples=count,precision_max_error=float(np.max(np.abs(inv.cpu().numpy()-ref.precision_)))))
    mean.flush();precision.flush();shrink.flush()
    return mean,precision,dict(status='passed',aligned=aligned,sklearn_checks=proofs,precision='FP64',no_feature_subsampling_for_reported_density=True)

@torch.inference_mode()
def extract(rows,kind,checkpoint,folder,phase):
    model=ProjectionNet(pretrained=False,head_layers=[512,128],num_classes=3).cuda().to(memory_format=torch.channels_last)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);assert saved['step']==STEPS
    model.load_state_dict(saved['model']);del saved;model.eval()
    shape=(len(rows),512) if kind=='image' else (len(rows),57*57,512)
    features=np.lib.format.open_memmap(folder/f'{phase}_features.npy',mode='w+',dtype='float32',shape=shape)
    for i,r in enumerate(rows):
        x=NORM(image(r['image_path'])).unsqueeze(0).cuda()
        if kind=='patch':x=x.unfold(2,32,4).unfold(3,32,4).permute(0,2,3,1,4,5).reshape(-1,3,32,32)
        parts=[]
        for j in range(0,len(x),256):parts.append(model(x[j:j+256].contiguous(memory_format=torch.channels_last))[0].float().cpu().numpy())
        z=np.concatenate(parts);assert np.isfinite(z).all();features[i]=z[0] if kind=='image' else z
    features.flush();del model;gc.collect();torch.cuda.empty_cache();return features

def score_patch(features,mean,precision,aligned):
    scores=[]
    for x in features:
        values=[]
        for j in range(0,len(x),32):
            mu=torch.from_numpy(np.array(mean[j:j+32] if aligned else mean,copy=True)).cuda()
            inv=torch.from_numpy(np.array(precision[j:j+32] if aligned else precision,copy=True)).cuda()
            q=torch.from_numpy(np.array(x[j:j+32],copy=True)).cuda().double()-mu
            dist=torch.einsum('bi,bij,bj->b',q,inv.expand(len(q),-1,-1),q) if aligned else torch.einsum('bi,ij,bj->b',q,inv[0],q)
            values.append(dist.clamp_min(0).cpu().numpy())
        scores.append(np.concatenate(values))
    return np.stack(scores)

def main(cat):
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=True
    out=OUT/cat;raw=RAW/cat;train=read(out/'train_images.csv');test=read(out/'test_images.csv');labels=np.array([int(r['label']) for r in test])
    artifact={}
    for kind in ['image','patch']:
        folder=raw/kind;proof=json.loads((out/kind/'checkpoint.json').read_text());assert sha(proof['path'])==proof['sha256']
        tr=extract(train,kind,proof['path'],folder,'train');te=extract(test,kind,proof['path'],folder,'test')
        print('FEATURE EXTRACTION COMPLETE',cat,kind,flush=True)
        if kind=='image':
            fit=LedoitWolf().fit(np.asarray(tr,dtype=np.float64));z=np.asarray(te,dtype=np.float64)-fit.location_
            scores=np.maximum(np.einsum('bi,ij,bj->b',z,fit.precision_,z),0)
            np.savez_compressed(folder/'density.npz',mean=fit.location_,precision=fit.precision_,shrinkage=fit.shrinkage_)
        else:
            aligned=cat in ALIGNED;mu,inv,dp=fit_patch(tr,aligned,folder)
            (out/'density_verification.json').write_text(json.dumps(dp,indent=2));ps=score_patch(te,mu,inv,aligned)
            maps=maps_from_scores(ps);ms=masks(test)
        del tr,te;gc.collect();torch.cuda.empty_cache()
    np.savez_compressed(raw/'predictions.npz',labels=labels,image_scores=scores,patch_scores=ps,score_maps=maps,masks=ms)
    row=dict(dataset='mvtec',category=cat,n_train=len(train),n_images=len(test),image_auroc=float(roc_auc_score(labels,scores)),pixel_auroc=float(roc_auc_score(ms.ravel(),maps.ravel())))
    (out/'metrics.json').write_text(json.dumps(row,indent=2));write(out/'image_scores.csv',[dict(image_path=r['image_path'],label=r['label'],score=float(s)) for r,s in zip(test,scores)])
    files=[raw/'predictions.npz',raw/'image/density.npz']
    for kind in ['image','patch']:files += [raw/kind/'last.pt',raw/kind/'train_features.npy',raw/kind/'test_features.npy']
    files += [raw/'patch'/f'density_{name}.npy' for name in ['mean','precision','shrinkage']]
    (out/'raw_manifest.json').write_text(json.dumps([dict(path=str(p),sha256=sha(p)) for p in files],indent=2))
    rows=[json.loads((OUT/c/'metrics.json').read_text()) for c in CATS if (OUT/c/'metrics.json').exists()]
    write(OUT/'category_metrics.csv',rows);print('CATEGORY COMPLETE',cat,row,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--category',choices=CATS,required=True);main(p.parse_args().category)
