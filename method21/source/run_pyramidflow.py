"""Official PyramidFlow ResNet18, BTAD adapter, final-epoch evaluation."""
import copy,gc,json,math,os,platform,subprocess,sys,time,traceback
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from pyramidflow_common import *
sys.path.insert(0,str(DEPENDENCY));sys.path.insert(0,str(REPO))
from model import PyramidFlow,VolumeNorm
from util import BatchDiffLoss,fix_randseed

ORIGINAL_VN=VolumeNorm.forward
def detached_volume_norm(self,x):
    if self.training:
        sample_mean=torch.mean(x,dim=self.dims,keepdim=True)
        # Native running buffer otherwise retains every preceding autograd
        # graph. It never participates in training output, so detach only it.
        with torch.no_grad():self.running_mean=(1-self.momentum)*self.running_mean+self.momentum*sample_mean.detach()
        return x-sample_mean
    return x-self.running_mean

def verify_buffer_fix():
    a=VolumeNorm((0,1));b=VolumeNorm((0,1));x=torch.randn(2,3,8,8,requires_grad=True);y=x.detach().clone().requires_grad_()
    p=ORIGINAL_VN(a,x);q=detached_volume_norm(b,y)
    assert torch.equal(p,q) and torch.equal(a.running_mean,b.running_mean)
    p.square().sum().backward();q.square().sum().backward();assert torch.equal(x.grad,y.grad) and not b.running_mean.requires_grad
    return dict(status='passed',forward_exact=True,input_gradient_exact=True,running_mean_value_exact=True,detached_running_buffer=True)

class BTAD(torch.utils.data.Dataset):
    def __init__(self,rows):self.rows=rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        with Image.open(self.rows[i]['image_path']) as im:x=XFORM(im.convert('RGB'))
        return x,mask(self.rows[i])

def load_state(flow,state):
    # CVN running buffer shape is data-dependent (spatial), not 1x1.
    for name,module in flow.named_modules():
        if isinstance(module,VolumeNorm):module.running_mean=state[name+'.running_mean'].to(next(flow.parameters()).device).clone()
    flow.load_state_dict(state,strict=True)

def loss(flow,criterion,x):
    pyramid=flow(x);diff=criterion(pyramid)
    return torch.fft.fft2(flow.pyramid.compose_pyramid(diff).mean(1)).abs().mean()

def tf32(enabled):
    torch.backends.cuda.matmul.allow_tf32=enabled;torch.backends.cudnn.allow_tf32=enabled

def verify_gpu(flow,criterion,x,cat):
    state=copy.deepcopy(flow.state_dict());flow.train();tf32(False)
    ref=loss(flow,criterion,x);ref.backward()
    grad_ref=[p.grad.detach().clone() for p in flow.parameters() if p.grad is not None]
    flow.zero_grad(set_to_none=True);load_state(flow,state);tf32(True)
    fast=loss(flow,criterion,x);fast.backward()
    grads=[p.grad for p in flow.parameters() if p.grad is not None]
    assert torch.isfinite(fast) and all(torch.isfinite(g).all() for g in grads)
    grad_rel=float(torch.sqrt(sum((a-b).square().sum() for a,b in zip(grads,grad_ref)))/torch.sqrt(sum(g.square().sum() for g in grad_ref)).clamp_min(1e-12))
    loss_rel=float((fast-ref).abs()/ref.abs().clamp_min(1e-12))
    enable=grad_rel<=.02 and loss_rel<=.005
    proof=dict(status='passed',gpu=torch.cuda.get_device_name(),real_BTAD_batch=True,batch_size=2,input_size=1024,
        fp32_loss=float(ref.detach()),tf32_loss=float(fast.detach()),loss_relative_difference=loss_rel,gradient_relative_L2_difference=grad_rel,
        tf32_enabled=enable,thresholds=dict(loss=.005,gradient=.02),finite_gradients=True,
        allocated_mib=torch.cuda.memory_allocated()/1024**2)
    (OUT/cat/'first_gpu_step.json').write_text(json.dumps(proof,indent=2))
    flow.zero_grad(set_to_none=True);load_state(flow,state);tf32(enable)
    del grad_ref,grads,state,ref,fast;gc.collect();torch.cuda.empty_cache()
    print('REAL GPU LOSS/GRADIENT CHECK PASSED',cat,'TF32',enable,flush=True)
    return enable

def checkpoint(flow,optimizer,epoch,path):
    temp=path.with_suffix('.tmp.pt')
    torch.save(dict(model=flow.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch),temp);temp.replace(path)

@torch.no_grad()
def evaluate(flow,train,test,raw,out):
    flow.eval();val=torch.utils.data.DataLoader(BTAD(train),batch_size=1,num_workers=4,persistent_workers=True,pin_memory=True)
    template=[0]*4;count=0
    for x,_ in val:
        z=flow(x.cuda(non_blocking=True));template=[a+b for a,b in zip(template,z)];count+=1
    template=[x/count for x in template];del val
    torch.save([x.cpu() for x in template],raw/'template.pt')
    loader=torch.utils.data.DataLoader(BTAD(test),batch_size=4,num_workers=4,persistent_workers=True,pin_memory=True)
    maps=[];ms=[]
    for i,(x,m) in enumerate(loader):
        z=flow(x.cuda(non_blocking=True));d=[(a-b).abs() for a,b in zip(z,template)]
        am=flow.pyramid.compose_pyramid(d).mean(1)
        assert am.shape[-2:]==(256,256) and torch.isfinite(am).all()
        if i==0:torch.save(dict(latent=[a[:1].cpu() for a in z],template=[a.cpu() for a in template]),raw/'first_test_latent.pt')
        maps.append(am.cpu().numpy());ms.append(m.numpy()[:,0])
    maps=np.concatenate(maps);ms=np.concatenate(ms);labels=np.array([int(r['label']) for r in test]);scores=maps.max(axis=(1,2))
    missing=np.flatnonzero((labels==1)&(ms.max(axis=(1,2))==0))
    (out/'mask_resize_audit.json').write_text(json.dumps(dict(original_image_labels_retained=True,
        missing_after_native_resize=[test[int(i)]['image_path'] for i in missing],n_missing=len(missing),
        rule='Do not infer image labels from resized masks; preserve native pixel mask transform'),indent=2))
    np.savez_compressed(raw/'predictions.npz',labels=labels,masks=ms,score_maps=maps,image_scores=scores)
    return dict(image_auroc=float(roc_auc_score(labels,scores)),pixel_auroc=float(roc_auc_score(ms.ravel(),maps.ravel())))

def main():
    for repo,commit in [(REPO,COMMIT),(DEPENDENCY,DEP_COMMIT)]:
        assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==commit
    assert not subprocess.check_output(['git','-C',str(REPO),'status','--porcelain','--untracked-files=no'],text=True).strip()
    OUT.mkdir(parents=True,exist_ok=True);RAW.mkdir(parents=True,exist_ok=True)
    if (OUT/'verification.json').exists() and json.loads((OUT/'verification.json').read_text())['status']=='passed':return
    previous=OUT/'environment.json'
    if previous.exists() and not (OUT/'environment_before_mask_label_fix.json').exists():
        (OUT/'environment_before_mask_label_fix.json').write_bytes(previous.read_bytes())
    proof=verify_buffer_fix();VolumeNorm.forward=detached_volume_norm;torch.set_num_threads(8)
    weights=Path(torch.hub.get_dir())/'checkpoints/resnet18-f37072fd.pth'
    env=dict(status='preparing',official_repo='https://github.com/FourthM/PyramidFlow',commit=COMMIT,
        dependency_repo='https://github.com/twimclee/pyramidflow-moai',dependency_commit=DEP_COMMIT,dependency_sha256=sha(DEPENDENCY/'autoFlow.py'),
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,python=platform.python_version(),seed=0,dataset=str(DATA),
        categories=['01','02','03'],epochs=15,batch_size=2,encoder='resnet18 ImageNet V1 first stage frozen under no_grad; native BN training behavior retained',
        weights_path=str(weights),weights_sha256=sha(weights),input_size=1024,mask_size=256,num_layers=4,num_stacks=4,channels=64,kernel_size=7,
        volume_normalization='CVN dims(0,1), native auto fallback for category names not in SVN list; exact BTAD-specific config unavailable',
        optimizer='Adam lr2e-4 eps1e-4 weight_decay1e-5 betas(0.5,0.9), fused',gradient_clip=1.,loss='native BatchDiffLoss p2 -> native compose -> channel mean -> FFT2 abs mean',
        augmentation='none, as paper BTAD protocol',input='Resize fixed square1024 bilinear / ImageNet normalization',
        mask='Resize fixed square256 bilinear then native ToTensor round int',
        memory_saving=False,precision='FP32, TF32 enabled only after recorded loss/gradient check; no AMP for FFT',
        buffer_fix_verification=proof,checkpoint_selection='last epoch15; native demo saves best test pixel/pro checkpoints, report final instead',
        evaluation='all normal train batch1 latent mean template; test batch4 abs latent differences compose -> channel mean, no smoothing; image max',
        intermediate_test_evaluation=False,workers=4,no_test_tuning=True,raw_path=str(RAW),
        wrapper_sha256=sha(__file__),common_sha256=sha(ROOT/'pyramidflow_common.py'),
        resume_policy='Reuse finished epoch15 checkpoints with 15-row training history and saved GPU proof. No partial-epoch resume.',
        source_hashes={p.name:sha(p) for p in [REPO/'model.py',REPO/'util.py',REPO/'train.py']},error=None)
    rows=[]
    def state(status,cat,epoch=0):
        env.update(status=status,category=cat,epoch=epoch);(OUT/'environment.json').write_text(json.dumps(env,indent=2))
    for cat,ntr,nte in zip(['01','02','03'],[400,399,1000],[70,230,441]):
        out=OUT/cat;raw=RAW/cat;out.mkdir(exist_ok=True);raw.mkdir(exist_ok=True)
        train=manifest(cat,'train');test=manifest(cat,'test');assert len(train)==ntr and len(test)==nte
        write(out/'train_images.csv',train);write(out/'test_images.csv',test)
        ld=fix_randseed(0)
        # cuDNN algorithm selection and non-memory-saving native backend
        # speed training; data/seed/model/batch remain fixed.
        torch.use_deterministic_algorithms(False);torch.backends.cudnn.deterministic=False;torch.backends.cudnn.benchmark=True
        # Native NumPy masks may be float64; explicitly convert the entire
        # model, including fixed LU masks/buffers, to the chosen FP32 dtype.
        flow=PyramidFlow(18,16,4,4,7,(0,1),False).float().cuda();criterion=BatchDiffLoss(2,p=2)
        optimizer=torch.optim.Adam(flow.parameters(),lr=2e-4,eps=1e-4,weight_decay=1e-5,betas=(.5,.9),fused=True)
        loader=torch.utils.data.DataLoader(BTAD(train),batch_size=2,shuffle=True,num_workers=4,persistent_workers=True,pin_memory=True,drop_last=True,**ld)
        state('training',cat);history=[];checked=False;start_epoch=1
        if (raw/'last.pt').exists() and (out/'training.csv').exists():
            history=read(out/'training.csv')
            saved=torch.load(raw/'last.pt',map_location='cuda',weights_only=True)
            assert saved['epoch']==15 and len(history)==15 and int(history[-1]['epoch'])==15
            load_state(flow,saved['model']);optimizer.load_state_dict(saved['optimizer']);del saved
            prior_proof=json.loads((out/'first_gpu_step.json').read_text());assert prior_proof['status']=='passed'
            tf32(prior_proof['tf32_enabled']);checked=True;start_epoch=16
            (out/'resume.json').write_text(json.dumps(dict(reused_epoch=15,checkpoint_sha256=sha(raw/'last.pt'),
                previous_environment_sha256=sha(OUT/'environment_before_mask_label_fix.json')),indent=2))
            print('REUSED COMPLETED TRAINING CHECKPOINT',cat,15,flush=True)
        for epoch in range(start_epoch,16):
            flow.train();start=time.perf_counter();losses=[]
            for x,_ in loader:
                x=x.cuda(non_blocking=True)
                if not checked:verify_gpu(flow,criterion,x,cat);checked=True
                optimizer.zero_grad(set_to_none=True);value=loss(flow,criterion,x)
                assert torch.isfinite(value),(cat,epoch)
                value.backward();torch.nn.utils.clip_grad_norm_(flow.parameters(),max_norm=1.)
                optimizer.step();losses.append(float(value.detach()))
            checkpoint(flow,optimizer,epoch,raw/'last.pt')
            history.append(dict(epoch=epoch,loss=float(np.mean(losses)),n_batches=len(losses),seconds=time.perf_counter()-start))
            write(out/'training.csv',history);state('training',cat,epoch);print('EPOCH COMPLETE',cat,epoch,flush=True)
        del loader;state('evaluating',cat,15)
        result=evaluate(flow,train,test,raw,out)
        row=dict(dataset='btad',category=cat,n_train=ntr,n_images=nte,checkpoint_epoch=15,**result);rows.append(row)
        (out/'metrics.json').write_text(json.dumps(row,indent=2));write(OUT/'category_metrics.csv',rows)
        with np.load(raw/'predictions.npz') as z:write(out/'image_scores.csv',[dict(image_path=r['image_path'],label=r['label'],score=float(s)) for r,s in zip(test,z['image_scores'])])
        (out/'raw_manifest.json').write_text(json.dumps({name:dict(path=str(raw/name),sha256=sha(raw/name)) for name in ['predictions.npz','last.pt','template.pt','first_test_latent.pt']},indent=2))
        print('CATEGORY COMPLETE',cat,result,flush=True);del flow,optimizer;gc.collect();torch.cuda.empty_cache()
    state('verifying','03',15)
    subprocess.run([sys.executable,str(ROOT/'finish_pyramidflow.py')],check=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        p=OUT/'environment.json'
        if p.exists():
            env=json.loads(p.read_text());env.update(status='failed',error=traceback.format_exc());p.write_text(json.dumps(env,indent=2))
        raise
