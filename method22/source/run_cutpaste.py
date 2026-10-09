"""Full paper update count; global and patch models for all MVTec classes."""
import copy,gc,json,math,os,platform,random,subprocess,sys,time,traceback
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset,RandomSampler
from torchvision import transforms as T
from cutpaste_common import *
from cutpaste_acceleration import configure_acceleration
sys.path.insert(0,str(REPO))
from model import ProjectionNet
from cutpaste import CutPaste3Way,cut_paste_collate_fn

def seed(value=42):
    random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)
def worker_seed(i):
    s=torch.initial_seed()%2**32;random.seed(s);np.random.seed(s)
class Training(Dataset):
    def __init__(self,rows,kind):
        self.images=[image(r['image_path']) for r in rows];self.kind=kind
        self.jitter=T.ColorJitter(.1,.1,.1,.1);self.translate=T.RandomAffine(0,translate=(.1,.1))
        self.cp=CutPaste3Way(transform=NORM)
    def __len__(self):return len(self.images)
    def __getitem__(self,i):
        x=self.images[i].copy()
        if self.kind=='patch':x=T.RandomCrop(64)(x)
        x=self.jitter(self.translate(x));return self.cp(x)

def amp_context(enabled):return torch.autocast('cuda',dtype=torch.bfloat16,enabled=enabled)
def objective(model,x,y,bf16):
    with amp_context(bf16):return torch.nn.functional.cross_entropy(model(x)[1],y)
def check_acceleration(model,x,y,out):
    state=copy.deepcopy(model.state_dict());model.train()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ref=objective(model,x,y,False);ref.backward();ref_grad=[p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
    model.zero_grad(set_to_none=True);model.load_state_dict(state)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    fast=objective(model,x,y,True);fast.backward();grads=[p.grad for p in model.parameters() if p.grad is not None]
    rel=float(torch.sqrt(sum((a-b).square().sum() for a,b in zip(grads,ref_grad)))/torch.sqrt(sum(g.square().sum() for g in ref_grad)).clamp_min(1e-12))
    loss_rel=float((fast.detach()-ref.detach()).abs()/ref.detach().abs().clamp_min(1e-12));enabled=rel<=.05 and loss_rel<=.01
    if not enabled:
        model.zero_grad(set_to_none=True);model.load_state_dict(state)
        alt=objective(model,x,y,False);alt.backward();grads=[p.grad for p in model.parameters() if p.grad is not None]
        alt_rel=float(torch.sqrt(sum((a-b).square().sum() for a,b in zip(grads,ref_grad)))/torch.sqrt(sum(g.square().sum() for g in ref_grad)).clamp_min(1e-12))
        alt_loss=float((alt.detach()-ref.detach()).abs()/ref.detach().abs().clamp_min(1e-12))
        if alt_rel>.05 or alt_loss>.01:
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert all(torch.isfinite(g).all() for g in grads) and torch.isfinite(fast)
    proof=dict(status='passed',real_MVTec_batch=True,gpu=torch.cuda.get_device_name(),effective_batch=len(x),input_size=x.shape[-1],
        fp32_loss=float(ref.detach()),bf16_loss=float(fast.detach()),bf16_gradient_relative_L2=rel,bf16_loss_relative_difference=loss_rel,
        bf16_enabled=enabled,tf32_enabled=torch.backends.cuda.matmul.allow_tf32,thresholds=dict(gradient=.05,loss=.01),finite_gradients=True)
    (out/'first_gpu_step.json').write_text(json.dumps(proof,indent=2));model.zero_grad(set_to_none=True);model.load_state_dict(state)
    del state,ref,fast,ref_grad,grads;gc.collect();torch.cuda.empty_cache()
    print('REAL GPU TRAINING CHECK PASSED',str(out),'BF16',enabled,flush=True)
    return proof

def save(model,opt,step,raw,proof,scaler):
    temp=raw/'last.tmp.pt'
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,proof=proof,
        scaler=scaler.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),
        numpy_rng=np.random.get_state(),config=dict(steps=STEPS,seed=42,pretrained=False,head_layers=[512,128],base_batch=32)),temp)
    temp.replace(raw/'last.pt')

def train(cat,kind,rows,env):
    out=OUT/cat/kind;raw=RAW/cat/kind;out.mkdir(parents=True,exist_ok=True);raw.mkdir(parents=True,exist_ok=True)
    seed();model=ProjectionNet(pretrained=False,head_layers=[512,128],num_classes=3).cuda().to(memory_format=torch.channels_last)
    optimizer=torch.optim.SGD(model.parameters(),lr=.03,momentum=.9,weight_decay=3e-5,foreach=True)
    start_step=0;proof=None;history=[];scaler_state=None;runner=model;scaler=None
    if (raw/'last.pt').exists():
        saved=torch.load(raw/'last.pt',map_location='cpu',weights_only=False)
        assert saved['config']==dict(steps=STEPS,seed=42,pretrained=False,head_layers=[512,128],base_batch=32)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);start_step=saved['step'];proof=saved['proof'];scaler_state=saved.get('scaler')
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
        history=[r for r in read(out/'training.csv') if int(r['step'])<=start_step]
        torch.backends.cuda.matmul.allow_tf32=proof['tf32_enabled'];torch.backends.cudnn.allow_tf32=proof['tf32_enabled']
        del saved
        print('RESUMED CHECKPOINT',cat,kind,start_step,flush=True)
    if start_step<STEPS:
        ds=Training(rows,kind);sampler=RandomSampler(ds,replacement=True,num_samples=(STEPS-start_step)*32)
        loader=DataLoader(ds,batch_size=32,sampler=sampler,num_workers=4,worker_init_fn=worker_seed,
            collate_fn=cut_paste_collate_fn,persistent_workers=True,pin_memory=True,prefetch_factor=4)
        model.train();losses=[];timer=time.perf_counter()
        for step,data in enumerate(loader,start=start_step):
            # Keep pinned buffers through H2D; concatenate on GPU rather than creating an unpinned CPU batch.
            x=torch.cat([part.cuda(non_blocking=True) for part in data]).to(memory_format=torch.channels_last);y=torch.arange(3,device='cuda').repeat_interleave(32)
            if scaler is None:
                if proof is None:proof=check_acceleration(model,x,y,out)
                runner,scaler,proof=configure_acceleration(model,optimizer,x,y,out,proof,start_step)
                if scaler_state and proof['precision']=='fp16':scaler.load_state_dict(scaler_state)
            lr=.03*.5*(1+math.cos(math.pi*step/STEPS))
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16,enabled=proof['precision']=='fp16'):
                value=torch.nn.functional.cross_entropy(runner(x)[1],y)
            torch._assert_async(torch.isfinite(value))
            scaler.scale(value).backward();scaler.step(optimizer);scaler.update();losses.append(value.detach())
            if (step+1)%256==0:
                history.append(dict(step=step+1,paper_epoch=(step+1)//256,loss=float(np.mean(torch.stack(losses).cpu().double().numpy())),lr=lr,seconds=time.perf_counter()-timer))
                write(out/'training.csv',history);save(model,optimizer,step+1,raw,proof,scaler)
                env.update(status='training',category=cat,model_kind=kind,step=step+1,active_acceleration=proof);(OUT/'environment.json').write_text(json.dumps(env,indent=2))
                print('PAPER EPOCH COMPLETE',cat,kind,(step+1)//256,flush=True);losses=[];timer=time.perf_counter()
        del loader,ds
    assert (raw/'last.pt').exists()
    (out/'checkpoint.json').write_text(json.dumps(dict(path=str(raw/'last.pt'),sha256=sha(raw/'last.pt'),steps=STEPS),indent=2))
    del runner,model,optimizer,scaler;gc.collect();torch.cuda.empty_cache()

def main():
    assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==COMMIT
    assert not subprocess.check_output(['git','-C',str(REPO),'status','--porcelain','--untracked-files=no'],text=True).strip()
    OUT.mkdir(parents=True,exist_ok=True);RAW.mkdir(parents=True,exist_ok=True)
    if (OUT/'verification.json').exists() and json.loads((OUT/'verification.json').read_text())['status']=='passed':return
    os.environ['TORCHINDUCTOR_COMPILE_THREADS']='2'
    torch.set_num_threads(8);torch.backends.cudnn.benchmark=True
    env=dict(status='preparing',official_author_implementation=False,public_repo='https://github.com/Runinho/pytorch-cutpaste',commit=COMMIT,
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,python=platform.python_version(),seed=42,categories=CATS,dataset=str(DATA),
        model='scratch ResNet18 + public MLP512,128 + 3-way classifier',models_per_category=['image256','patch64'],
        steps_per_model=STEPS,paper_epochs=256,updates_per_paper_epoch=256,base_batch=32,effective_batch=96,
        optimizer='SGD lr.03 momentum.9 wd3e-5 cosine one cycle, foreach',no_pretraining=True,no_test_selection=True,
        augment='translation max10% (paper magnitude unspecified), global/patch ColorJitter.1, public CutPasteNormal/Scar 3way',
        density='LedoitWolf Gaussian raw pooled512; aligned location-wise otherwise pooled all train patches',
        localization='train random64patch; evaluate32patch stride4 57x57 (paper appendix A.5); Gaussian32 RF transposed conv sigma8',
        unspecified_choices='MLP512,128 from public CLI head1; translation10%; Gaussian sigma8. Original TensorFlow preprocessing/init differs.',
        repeat='single seed, not paper5-run mean/SE',raw_path=str(RAW),
        source_hashes={p.name:sha(p) for p in [REPO/'model.py',REPO/'cutpaste.py',REPO/'run_training.py',REPO/'density.py']},
        wrapper_sha256=sha(__file__),common_sha256=sha(ROOT/'cutpaste_common.py'),
        acceleration_helper_sha256=sha(ROOT/'cutpaste_acceleration.py'),
        data_pipeline=dict(workers=4,prefetch_factor=4,main_threads=8,assembly='separate pinned async H2D then GPU concat; tensor values/order unchanged',loss_reporting='GPU finite assert each step; detached loss scalars transferred together every256 updates and averaged in FP64',benchmark=str(OUT/'data_transfer_benchmark.json')),
        resume_limit='model/optimizer/scaler/main RNG restored, workers/sampler stream not exact across restart',
        acceleration_policy='compiled/eager FP16 + GradScaler + TF32, finite/loss gate and FP32 gradient diagnostic, channels_last, cuDNN benchmark, foreach SGD, cached images, pin/prefetch workers; per-model fallback if nonfinite or loss gate fails; compiled/eager FP16 comparison; evaluation FP32/FP64',error=None)
    (OUT/'environment.json').write_text(json.dumps(env,indent=2))
    for cat in CATS:
        folder=OUT/cat;folder.mkdir(exist_ok=True)
        if (folder/'metrics.json').exists() and (folder/'raw_manifest.json').exists():
            assert all(sha(a['path'])==a['sha256'] for a in json.loads((folder/'raw_manifest.json').read_text()))
            print('PRESERVED COMPLETED CATEGORY',cat,flush=True);continue
        tr=manifest(cat,'train');te=manifest(cat,'test')
        write(folder/'train_images.csv',tr);write(folder/'test_images.csv',te)
        for kind in ['image','patch']:
            env.update(status='training',category=cat,model_kind=kind,step=0);(OUT/'environment.json').write_text(json.dumps(env,indent=2))
            train(cat,kind,tr,env)
        env['status']='evaluating';(OUT/'environment.json').write_text(json.dumps(env,indent=2))
        subprocess.run([sys.executable,str(ROOT/'evaluate_cutpaste.py'),'--category',cat],check=True)
    env['status']='verifying';(OUT/'environment.json').write_text(json.dumps(env,indent=2))
    subprocess.run([sys.executable,str(ROOT/'finish_cutpaste.py')],check=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        p=OUT/'environment.json'
        if p.exists():
            env=json.loads(p.read_text());env.update(status='failed',error=traceback.format_exc());p.write_text(json.dumps(env,indent=2))
        raise
