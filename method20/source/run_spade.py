"""Paper-aligned SPADE with audited public-code feature extraction.

No optimizer, coreset or test tuning. Paper K=50 and squared L2 are used;
layer1/2/3 pyramids are concatenated before dense nearest matching.
"""
import json,os,platform,random,subprocess,sys,time,traceback
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torchvision.models import wide_resnet50_2,Wide_ResNet50_2_Weights
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
import cv2
from spade_common import *

class Images(Dataset):
    def __init__(self,rows):self.rows=rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        with Image.open(self.rows[i]['image_path']) as im:return XFORM(im.convert('RGB'))

@torch.inference_mode()
def features(model,outputs,rows,folder,phase):
    loader=DataLoader(Images(rows),batch_size=32,num_workers=4,pin_memory=True,persistent_workers=True)
    arrays={};offset=0
    for x in loader:
        outputs.clear();model(x.cuda(non_blocking=True))
        assert len(outputs)==4
        for name,y in zip(['layer1','layer2','layer3','avgpool'],outputs):
            z=y.cpu().numpy()
            if name not in arrays:arrays[name]=np.lib.format.open_memmap(folder/f'{phase}_{name}.npy',mode='w+',dtype='float32',shape=(len(rows),*z.shape[1:]))
            arrays[name][offset:offset+len(x)]=z
        offset+=len(x)
    assert offset==len(rows)
    for a in arrays.values():a.flush()
    outputs.clear();torch.cuda.empty_cache()
    return arrays

@torch.inference_mode()
def pyramid(feats,idx):
    layers=[]
    for name in ['layer1','layer2','layer3']:
        a=torch.from_numpy(np.array(feats[name][idx],copy=True)).cuda()
        if a.ndim==3:a=a.unsqueeze(0)
        if a.shape[-1]!=56:a=F.interpolate(a,size=(56,56),mode='bilinear',align_corners=False)
        layers.append(a)
    return torch.cat(layers,1).permute(0,2,3,1).reshape(-1,1792).contiguous()

@torch.inference_mode()
def dense_min(q,g):
    # Exhaustive matching. Chunking changes memory use, not candidate set.
    gn=(g*g).sum(1);result=[]
    for i in range(0,len(q),512):
        x=q[i:i+512];xn=(x*x).sum(1);best=torch.full((len(x),),float('inf'),device='cuda')
        for j in range(0,len(g),32768):
            d=xn[:,None]+gn[j:j+32768][None,:]-2*(x@g[j:j+32768].T)
            best=torch.minimum(best,d.min(1).values)
        result.append(best.clamp_min(0))
    return torch.cat(result)

@torch.inference_mode()
def verify_dense(q,g,cat):
    # Actual first test image and its full K=50 gallery, not synthetic input.
    indices=[0,1568,3135];actual=dense_min(q[indices],g).cpu().numpy()
    reference=[];gd=g.cpu().numpy();qd=q[indices].cpu().numpy().astype(np.float64)
    for x in qd:
        best=float('inf')
        for j in range(0,len(gd),4096):
            delta=gd[j:j+4096].astype(np.float64)-x
            best=min(best,float(np.einsum('ij,ij->i',delta,delta).min()))
        reference.append(best)
    reference=np.array(reference)
    assert np.allclose(actual,reference,rtol=1e-4,atol=2e-3),(actual,reference)
    proof=dict(status='passed',category=cat,real_BTAD_first_test=True,query_locations=indices,gallery_size=len(g),channels=1792,
               exhaustive_float64_direct_difference=reference.tolist(),gpu_chunked_squared_l2=actual.tolist(),rtol=1e-4,atol=2e-3,
               max_absolute_difference=float(np.max(np.abs(actual-reference))),TF32=False,AMP=False)
    (OUT/cat/'distance_verification.json').write_text(json.dumps(proof,indent=2))
    print('REAL FULL GALLERY GPU DISTANCE CHECK PASSED',cat,flush=True)

def map256(d):
    up=F.interpolate(torch.from_numpy(d).reshape(1,1,56,56),size=(224,224),mode='bilinear',align_corners=False).numpy().squeeze()
    return gaussian_filter(cv2.resize(up,(256,256),interpolation=cv2.INTER_AREA),sigma=4)

def main():
    assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==COMMIT
    assert not subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip()
    OUT.mkdir(parents=True,exist_ok=True);RAW.mkdir(parents=True,exist_ok=True)
    if (OUT/'verification.json').exists() and json.loads((OUT/'verification.json').read_text())['status']=='passed':return
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');torch.backends.cudnn.benchmark=True
    weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1
    model=wide_resnet50_2(weights=weights).cuda().eval();outputs=[]
    for module in [model.layer1[-1],model.layer2[-1],model.layer3[-1],model.avgpool]:
        module.register_forward_hook(lambda m,x,y:outputs.append(y.detach()))
    checkpoint=Path(torch.hub.get_dir())/'checkpoints'/Path(weights.url).name
    env=dict(status='extracting',category='01',official_author_implementation=False,public_repo='https://github.com/byungjae89/SPADE-pytorch',commit=COMMIT,
             dataset=str(DATA),categories=['01','02','03'],n_train=[400,399,1000],n_test=[70,230,441],seed=42,training_epochs=0,
             backbone='torchvision wide_resnet50_2 IMAGENET1K_V1',weights_url=weights.url,weights_path=str(checkpoint),weights_sha256=sha(checkpoint),
             gpu=torch.cuda.get_device_name(),torch=torch.__version__,python=platform.python_version(),precision='FP32 no TF32 no AMP',
             input='Resize shorter side256 LANCZOS / CenterCrop224 / ImageNet normalization',
             K=50,kappa=1,global_score='mean K50 squared Euclidean distance of avgpool2048',
             pyramid='layer1/2/3 256/512/1024 channels; bilinear upsample to56; concatenate1792; exact dense squared L2',
             postprocess='56->224 bilinear align_corners=False ->256 cv2.INTER_AREA -> Gaussian sigma4',
             mask='Resize shorter side256 NEAREST / CenterCrop224 / >128 ->256 INTER_AREA ->>.5',
             protocol_ambiguity='paper does not specify feature alignment interpolation or mapping crop224 to evaluation256; choices recorded',
             no_test_tuning=True,batch_size=32,raw_path=str(RAW),wrapper_sha256=sha(__file__),common_sha256=sha(ROOT/'spade_common.py'),
             source_hashes={str(p.relative_to(REPO)):sha(p) for p in [REPO/'src/main.py',REPO/'src/datasets/mvtec.py']},error=None)
    def state(status,cat):
        env.update(status=status,category=cat);(OUT/'environment.json').write_text(json.dumps(env,indent=2))
    rows=[]
    for cat,ntr,nte in zip(['01','02','03'],[400,399,1000],[70,230,441]):
        out=OUT/cat;raw=RAW/cat;out.mkdir(exist_ok=True);raw.mkdir(exist_ok=True)
        train=manifest(cat,'train');test=manifest(cat,'test');assert len(train)==ntr and len(test)==nte
        write(out/'train_images.csv',train);write(out/'test_images.csv',test)
        state('extracting',cat);t0=time.perf_counter()
        tr=features(model,outputs,train,raw,'train');te=features(model,outputs,test,raw,'test')
        print('FEATURE EXTRACTION COMPLETE',cat,flush=True);state('matching',cat)
        train_global=torch.from_numpy(np.array(tr['avgpool']).reshape(ntr,-1)).cuda()
        test_global=torch.from_numpy(np.array(te['avgpool']).reshape(nte,-1)).cuda()
        # Small global matrices permit direct subtraction, preserving Eq2.
        distances=[]
        for x in test_global:distances.append(((x[None,:]-train_global)**2).sum(1))
        dist=torch.stack(distances);topvals,topidx=dist.topk(50,largest=False,dim=1)
        scores=topvals.mean(1).cpu().numpy();neighbors=topidx.cpu().numpy()
        mask=masks(test);labels=np.array([r['label'] for r in test]);maps=[];nn56=[]
        for i in range(nte):
            with torch.inference_mode():
                g=pyramid(tr,neighbors[i]);q=pyramid(te,i)
                if i==0:verify_dense(q,g,cat)
                d=dense_min(q,g).reshape(56,56).cpu().numpy()
            assert np.isfinite(d).all();nn56.append(d);maps.append(map256(d));del q,g
        prediction=raw/'predictions.npz'
        np.savez_compressed(prediction,labels=labels,masks=mask,image_scores=scores,score_maps=np.stack(maps),nn56=np.stack(nn56),neighbor_indices=neighbors)
        row=dict(dataset='btad',category=cat,n_train=ntr,n_images=nte,image_auroc=float(roc_auc_score(labels,scores)),
                 pixel_auroc=float(roc_auc_score(mask.ravel(),np.stack(maps).ravel())),seconds=time.perf_counter()-t0)
        (out/'metrics.json').write_text(json.dumps(row,indent=2));rows.append(row);write(OUT/'category_metrics.csv',rows)
        write(out/'image_scores.csv',[dict(image_path=r['image_path'],label=r['label'],score=float(s)) for r,s in zip(test,scores)])
        (out/'raw_manifest.json').write_text(json.dumps(dict(path=str(prediction),sha256=sha(prediction),
            global_features={phase:dict(path=str(raw/f'{phase}_avgpool.npy'),sha256=sha(raw/f'{phase}_avgpool.npy')) for phase in ['train','test']}),indent=2))
        print('CATEGORY COMPLETE',cat,row,flush=True)
        del tr,te,mask,maps,nn56,train_global,test_global,dist;torch.cuda.empty_cache()
    state('verifying','03')
    subprocess.run([sys.executable,str(ROOT/'finish_spade.py')],check=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        p=OUT/'environment.json'
        if p.exists():
            env=json.loads(p.read_text());env.update(status='failed',error=traceback.format_exc());p.write_text(json.dumps(env,indent=2))
        raise
