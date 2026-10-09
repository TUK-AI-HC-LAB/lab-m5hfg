"""Official VT-AE/MDN training and inference for MuSc Table14 BTAD."""
import os,sys,json,random,time,gc,subprocess,importlib.metadata
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image
from vtadl_common import *

class BTAD(torch.utils.data.Dataset):
    def __init__(self,category,phase):
        self.category,self.phase=category,phase
        self.paths=sorted(p for p in (DATA/category/phase).glob('*/*') if p.suffix.lower() in ['.png','.bmp','.jpg','.jpeg','.tif','.tiff'])
        assert self.paths
        self.transform=T.Compose([T.Resize((550,550)),T.CenterCrop(512),T.ToTensor()])
        self.labels=[];self.masks=[]
        for p in self.paths:
            y=int(phase=='test' and p.parent.name!='ok');self.labels.append(y)
            candidates=list((DATA/category/'ground_truth'/p.parent.name).glob(p.stem+'.*')) if y else []
            assert not y or len(candidates)==1,(p,candidates)
            self.masks.append(candidates[0] if y else None)
    def __len__(self):return len(self.paths)
    def __getitem__(self,i):
        with Image.open(self.paths[i]) as im:x=self.transform(im.convert('RGB'))
        mask=torch.zeros(1,512,512)
        if self.masks[i]:
            with Image.open(self.masks[i]) as im:mask=(self.transform(im.convert('L'))>0).float()
        return x,mask
    def manifest(self):
        return [dict(image_path=str(p),sha256=sha(p),label=y,mask_path=str(m) if m else '',mask_sha256=sha(m) if m else '') for p,y,m in zip(self.paths,self.labels,self.masks)]

def runtime():
    sys.path.insert(0,str(REPO))
    import VT_AE,mdn1,pytorch_ssim
    from utility_fun import Filter
    return VT_AE,mdn1,pytorch_ssim,Filter

def seed():
    random.seed(123);np.random.seed(123);torch.manual_seed(123);torch.cuda.manual_seed_all(123)

def evaluate(category,checkpoint,env):
    VT_AE,mdn1,pytorch_ssim,Filter=runtime()
    seed();model=VT_AE.VT_AE(train=False).cuda();density=mdn1.MDN(coefs=150).cuda()
    saved=torch.load(checkpoint,map_location='cuda',weights_only=True)
    model.load_state_dict(saved['model'],strict=True);density.load_state_dict(saved['density'],strict=True)
    model.eval();density.eval();ssim=pytorch_ssim.SSIM()
    ds=BTAD(category,'test');loader=torch.utils.data.DataLoader(ds,batch_size=1,shuffle=False,num_workers=4,persistent_workers=True,pin_memory=True)
    maps=[];scores=[];masks=[];parts=[];patches=[]
    # Keep native test score: loss1 - loss2 + max(loss3), loss2=-SSIM.
    with torch.no_grad():
        for x,m in loader:
            x=x.cuda(non_blocking=True);vector,recon=model(x);pi,mu,sigma=density(vector)
            mse=F.mse_loss(recon,x);struct=ssim(x,recon)
            loss3=mdn1.mdn_loss_function(vector,mu,sigma,pi,test=True)
            patch=loss3.detach().cpu()
            up=torch.nn.UpsamplingBilinear2d((512,512))(patch.reshape(1,1,8,8))
            amap=Filter(up,type=0)[0,0]
            score=(mse+struct+loss3.max()).item()
            assert np.isfinite(amap).all() and np.isfinite(score)
            maps.append(amap);scores.append(score);masks.append(m.numpy()[0,0].astype(np.uint8));patches.append(patch.numpy()[0])
            parts.append(dict(mse=float(mse),ssim=float(struct),max_patch_density=float(loss3.max())))
    maps=np.asarray(maps,dtype=np.float32);masks=np.asarray(masks,dtype=np.uint8)
    y=np.asarray(ds.labels,dtype=np.int64);s=np.asarray(scores,dtype=np.float64)
    folder=OUT/category;rawfolder=RAW/category
    raw=rawfolder/'predictions.npz'
    np.savez_compressed(raw,image_labels=y,image_scores=s,anomaly_maps=maps,masks=masks,patch_density=np.asarray(patches),
        mse=np.array([p['mse'] for p in parts]),ssim=np.array([p['ssim'] for p in parts]),max_patch_density=np.array([p['max_patch_density'] for p in parts]))
    write(folder/'test_images.csv',ds.manifest())
    write(folder/'image_scores.csv',[dict(image_path=str(p),label=int(label),score=float(value),**part) for p,label,value,part in zip(ds.paths,y,s,parts)])
    row=dict(dataset='btad',category=category,n_images=len(ds),**metrics(y,s,masks,maps),checkpoint_epoch=saved['epoch']+1)
    (folder/'metrics.json').write_text(json.dumps(row,indent=2))
    (folder/'raw_manifest.json').write_text(json.dumps(dict(path=str(raw),sha256=sha(raw),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint)),indent=2))
    del model,density,saved,loader,ds,maps,masks;gc.collect();torch.cuda.empty_cache()
    return row

def main():
    OUT.mkdir(parents=True,exist_ok=True);RAW.mkdir(parents=True,exist_ok=True)
    assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==COMMIT
    assert not subprocess.check_output(['git','-C',str(REPO),'diff','--name-only'],text=True).strip()
    if (OUT/'verification.json').exists() and json.loads((OUT/'verification.json').read_text())['status']=='passed':return
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=True
    VT_AE,mdn1,pytorch_ssim,_=runtime()
    env=dict(status='starting',official_repo='https://github.com/pankajmishra000/VT-ADL',commit=COMMIT,
        gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,python=sys.version,
        packages={p:importlib.metadata.version(p) for p in ['torch','torchvision','numpy','einops','scipy','scikit-learn']},
        dataset=str(DATA),categories=['01','02','03'],seed=123,epochs=400,batch_size=8,lr=.0001,weight_decay=.0001,
        patch_size=64,image_size=512,gaussian_components=150,model_init='scratch; no pretrained backbone',
        train_loss='5*MSE -0.5*SSIM + native MDN loss',image_score='native test.py MSE+SSIM+max(patch MDN loss)',
        pixel_score='native MDN patch loss; UpsamplingBilinear2d512; native Gaussian sigma4',
        image_transform='Resize550x550 bilinear/CenterCrop512/ToTensor RGB, no normalization',
        mask_transform='Resize550x550 bilinear/CenterCrop512/ToTensor L then>0, matching native Process_mask',
        checkpoint_selection='minimum mean training loss; no test-set selection',
        precision='FP32 with TF32 matmul/cuDNN; no AMP; fused Adam',
        optimizer_zero_grad='all optimized parameters reset each step; original train.py resets only AE and accumulates MDN gradients, corrected explicitly',
        original_checkout_edits=False,native_sources={p:sha(REPO/p) for p in ['VT_AE.py','mdn1.py','student_transformer.py','spatial.py','train.py','test.py','mvtech.py']},
        paper_code_differences=['paper Gaussian150 vs native default10; run uses150','paper five decoder layers vs public six; public architecture retained'],
        raw_path=str(RAW),wrapper_sha256=sha(__file__),error=None)
    def state():(OUT/'environment.json').write_text(json.dumps(env,indent=2))
    state();rows=[]
    try:
        for category in ['01','02','03']:
            folder=OUT/category;folder.mkdir(exist_ok=True);rawfolder=RAW/category;rawfolder.mkdir(exist_ok=True)
            marker=folder/'metrics.json'
            if marker.exists():rows.append(json.loads(marker.read_text()));continue
            ds=BTAD(category,'train');assert not any(ds.labels)
            write(folder/'train_images.csv',ds.manifest())
            test_ds=BTAD(category,'test');assert {str(p) for p in ds.paths}.isdisjoint(str(p) for p in test_ds.paths)
            seed();model=VT_AE.VT_AE(train=True).cuda();density=mdn1.MDN(coefs=150).cuda()
            model.train();density.train();ssim=pytorch_ssim.SSIM()
            optimizer=torch.optim.Adam(list(model.parameters())+list(density.parameters()),lr=.0001,weight_decay=.0001,fused=True)
            loader=torch.utils.data.DataLoader(ds,batch_size=8,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True,prefetch_factor=2)
            start_epoch=0;best=float('inf');history=[]
            last=rawfolder/'last.pt';bestfile=rawfolder/'best.pt'
            if last.exists():
                saved=torch.load(last,map_location='cuda',weights_only=False)
                model.load_state_dict(saved['model']);density.load_state_dict(saved['density']);optimizer.load_state_dict(saved['optimizer'])
                start_epoch=saved['epoch']+1;best=saved['best'];history=read(folder/'training.csv')
                torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state_all([r.cpu() for r in saved['cuda_rng']]);np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
                del saved
            env.update(status='training',category=category,n_train=len(ds),n_test=len(test_ds));state()
            for epoch in range(start_epoch,400):
                losses=[];tic=time.perf_counter()
                for step,(x,_) in enumerate(loader):
                    x=x.cuda(non_blocking=True);optimizer.zero_grad(set_to_none=True)
                    vector,recon=model(x);pi,mu,sigma=density(vector)
                    l1=F.mse_loss(recon,x);l2=-ssim(x,recon);l3=mdn1.mdn_loss_function(vector,mu,sigma,pi)
                    loss=5*l1+.5*l2+l3
                    assert torch.isfinite(loss),('nonfinite loss',category,epoch,step)
                    loss.backward()
                    if epoch==0 and step==0:
                        assert all(torch.isfinite(p.grad).all() for p in list(model.parameters())+list(density.parameters()) if p.grad is not None)
                        (folder/'first_gpu_step.json').write_text(json.dumps(dict(status='passed',loss=float(loss),mse=float(l1),negative_ssim=float(l2),mdn_loss=float(l3),batch_size=len(x),coefs=150,
                            gpu=torch.cuda.get_device_name(0),allocated_mib=torch.cuda.max_memory_allocated()/1024**2,finite_gradients=True),indent=2))
                        print('REAL GPU TRAINING STEP PASSED',category,flush=True)
                    optimizer.step();losses.append(float(loss.detach()))
                mean=float(np.mean(losses));history.append(dict(epoch=epoch+1,mean_loss=mean,seconds=time.perf_counter()-tic,n_batches=len(losses)))
                write(folder/'training.csv',history)
                if mean<=best:
                    best=mean;torch.save(dict(model=model.state_dict(),density=density.state_dict(),epoch=epoch,mean_loss=mean),bestfile)
                # Save full optimizer/RNG every epoch for recovery. Atomic rename.
                temp=rawfolder/'last.tmp.pt'
                torch.save(dict(model=model.state_dict(),density=density.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,best=best,
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate()),temp)
                temp.replace(last)
                env.update(epoch=epoch+1,best_training_loss=best);state()
                print('EPOCH COMPLETE',category,epoch+1,'mean_loss',mean,flush=True)
            del model,density,optimizer,loader,ds,test_ds;gc.collect();torch.cuda.empty_cache()
            env['status']='evaluating';state();rows.append(evaluate(category,bestfile,env));write(OUT/'category_metrics.csv',rows)
        env['status']='evaluated';state()
        subprocess.run([sys.executable,str(ROOT/'finish_vtadl.py')],check=True)
    except Exception as e:
        env.update(status='failed',error=repr(e));state();raise
if __name__=='__main__':main()
