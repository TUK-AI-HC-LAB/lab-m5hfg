"""Train original Patch SVDD for all three BTAD products, evaluate final model."""
import sys,os,json,random,time,gc,subprocess,importlib.metadata
import numpy as np
import torch
from psvdd_common import *

def seed():
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)

def evaluate(category,checkpoint,env):
    from codes import inspection,mvtecad
    from codes.networks import EncoderHier
    train_paths=image_files(category,'train');test_paths,labels,masks,manifest=inputs(category)
    train=images(train_paths);test=images(test_paths);mean=train.astype(np.float32).mean(0)
    xtrain=(train.astype(np.float32)-mean)/255
    xtest=(test.astype(np.float32)-mean)/255
    del train,test
    # Redirect only dataset/label access; retain native patch inference, nearest
    # neighbors, patch-to-pixel distribution, sum/product fusion and metrics.
    mvtecad.get_x_standardized=lambda obj,mode='train':xtrain if mode=='train' else xtest
    mvtecad.get_label=lambda obj:labels
    mvtecad.get_mask=lambda obj:masks*255
    enc=EncoderHier(64,64).cuda();saved=torch.load(checkpoint,map_location='cuda',weights_only=False)
    enc.load_state_dict(saved['encoder'],strict=True);enc.eval()
    result=inspection.eval_encoder_NN_multiK(enc,category)
    folder=OUT/category;rawfolder=RAW/category
    raw=rawfolder/'predictions.npz'
    np.savez_compressed(raw,image_labels=labels,masks=masks,**{k:result[k] for k in ['maps_64','maps_32','maps_sum','maps_mult']})
    write(folder/'test_images.csv',manifest)
    modes=[]
    for mode in ['64','32','sum','mult']:
        maps=result['maps_'+mode];scores=maps.reshape(len(labels),-1).max(1)
        row=dict(dataset='btad',category=category,fusion=mode,n_images=len(labels),**metrics(labels,scores,masks,maps))
        assert abs(row['image_auroc']-result['det_'+mode])<1e-12 and abs(row['pixel_auroc']-result['seg_'+mode])<1e-12
        modes.append(row)
    write(folder/'all_fusion_metrics.csv',modes)
    write(folder/'image_scores.csv',[dict(image_path=str(p),label=int(y),score=float(s)) for p,y,s in zip(test_paths,labels,result['maps_mult'].reshape(len(labels),-1).max(1))])
    row=next(r for r in modes if r['fusion']=='mult');row['checkpoint_epoch']=saved['epoch']+1
    (folder/'metrics.json').write_text(json.dumps(row,indent=2))
    (folder/'raw_manifest.json').write_text(json.dumps(dict(path=str(raw),sha256=sha(raw),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint)),indent=2))
    del enc,saved,result,xtrain,xtest;gc.collect();torch.cuda.empty_cache()
    return row

def main():
    OUT.mkdir(parents=True,exist_ok=True);RAW.mkdir(parents=True,exist_ok=True)
    assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==COMMIT
    assert not subprocess.check_output(['git','-C',str(REPO),'diff','--name-only'],text=True).strip()
    if (OUT/'verification.json').exists() and json.loads((OUT/'verification.json').read_text())['status']=='passed':return
    sys.path.insert(0,str(REPO))
    from codes.datasets import PositionDataset,SVDD_Dataset
    from codes.utils import DictionaryConcatDataset,NHWC2NCHW,to_device
    from codes.networks import EncoderHier,PositionClassifier
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=True
    env=dict(status='starting',official_repo='https://github.com/nuclearboy95/Anomaly-Detection-PatchSVDD-PyTorch',commit=COMMIT,
        gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,python=sys.version,
        packages={p:importlib.metadata.version(p) for p in ['torch','torchvision','numpy','scikit-learn','ngt','Pillow']},
        dataset=str(DATA),categories=['01','02','03'],seed=42,epoch_slots=300,gradient_training_epochs=299,
        epoch_definition='native loop0 evaluates untrained model; train loop1..299. Wrapper preserves299 update epochs and slot0, removes repeated test eval.',
        batch_size=64,repeat=100,D=64,lr=.0001,lambda_value=1,
        lambda_choice='official CLI default1; BTAD setting not provided in MuSc, no test-based tuning; README bottle example1e-3 differs',
        image_transform='native current Pillow Resize256x256 defaultBICUBIC, subtract train per-position RGB mean, divide255',
        mask_transform='native current Pillow default resize256x256 then>128; filename-matched BTAD masks',
        model_init='scratch native EncoderHier64 and two position classifiers',
        train_loss='pos64+pos32+lambda*(SVDD64+SVDD32)',
        inference='native K64/S16 KDTree; K32/S4 NGT approximate NN1; native distribution and product fusion',
        checkpoint_selection='last slot300, no test selection',precision='FP32 with TF32 matmul/cuDNN; fusedAdam; noAMP',
        original_checkout_edits=False,wrapper_sha256=sha(__file__),
        native_sources={p:sha(REPO/p) for p in ['main_train.py','codes/datasets.py','codes/networks.py','codes/inspection.py','codes/nearest_neighbor.py','codes/utils.py','codes/mvtecad.py']},
        raw_path=str(RAW),error=None)
    def state():(OUT/'environment.json').write_text(json.dumps(env,indent=2))
    state();rows=[]
    try:
        for category in ['01','02','03']:
            folder=OUT/category;folder.mkdir(exist_ok=True);rawfolder=RAW/category;rawfolder.mkdir(exist_ok=True)
            if (folder/'metrics.json').exists():rows.append(json.loads((folder/'metrics.json').read_text()));continue
            paths=image_files(category,'train');x=images(paths);mean=x.astype(np.float32).mean(0)
            x=NHWC2NCHW((x.astype(np.float32)-mean)/255)
            np.save(rawfolder/'training_image_mean.npy',mean)
            write(folder/'train_images.csv',[dict(image_path=str(p),sha256=sha(p),label=0) for p in paths])
            seed();enc=EncoderHier(64,64).cuda();c64=PositionClassifier(64,64).cuda();c32=PositionClassifier(32,64).cuda()
            modules=[enc,c64,c32];params=[p for m in modules for p in m.parameters()]
            opt=torch.optim.Adam(params,lr=.0001,fused=True)
            datasets=dict(pos_64=PositionDataset(x,K=64,repeat=100),pos_32=PositionDataset(x,K=32,repeat=100),svdd_64=SVDD_Dataset(x,K=64,repeat=100),svdd_32=SVDD_Dataset(x,K=32,repeat=100))
            loader=torch.utils.data.DataLoader(DictionaryConcatDataset(datasets),batch_size=64,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True,prefetch_factor=2)
            last=rawfolder/'last.pt';start=0;history=[]
            if last.exists():
                saved=torch.load(last,map_location='cuda',weights_only=False)
                for name,m in zip(['encoder','classifier64','classifier32'],modules):m.load_state_dict(saved[name])
                opt.load_state_dict(saved['optimizer']);start=saved['epoch']+1;history=read(folder/'training.csv')
                torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state_all([r.cpu() for r in saved['cuda_rng']]);np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng']);del saved
            env.update(status='training',category=category,n_train=len(paths));state()
            for epoch in range(start,300):
                losses=[];tic=time.perf_counter()
                if epoch!=0:
                    for m in modules:m.train()
                    for step,d in enumerate(loader):
                        d=to_device(d,'cuda',non_blocking=True);opt.zero_grad(set_to_none=True)
                        p64=PositionClassifier.infer(c64,enc,d['pos_64']);p32=PositionClassifier.infer(c32,enc.enc,d['pos_32'])
                        s64=SVDD_Dataset.infer(enc,d['svdd_64']);s32=SVDD_Dataset.infer(enc.enc,d['svdd_32'])
                        loss=p64+p32+s64+s32
                        assert torch.isfinite(loss),(category,epoch,step)
                        loss.backward()
                        if epoch==1 and step==0:
                            assert all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
                            (folder/'first_gpu_step.json').write_text(json.dumps(dict(status='passed',gpu=torch.cuda.get_device_name(0),batch_size=64,loss=float(loss.detach()),
                                losses=[float(v.detach()) for v in [p64,p32,s64,s32]],finite_gradients=True,allocated_mib=torch.cuda.max_memory_allocated()/1024**2),indent=2))
                            print('REAL GPU TRAINING STEP PASSED',category,flush=True)
                        opt.step();losses.append(float(loss.detach()))
                history.append(dict(epoch=epoch+1,mean_loss=float(np.mean(losses)) if losses else 0,n_batches=len(losses),seconds=time.perf_counter()-tic))
                write(folder/'training.csv',history)
                temp=rawfolder/'last.tmp.pt'
                torch.save(dict(encoder=enc.state_dict(),classifier64=c64.state_dict(),classifier32=c32.state_dict(),optimizer=opt.state_dict(),epoch=epoch,
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate()),temp)
                temp.replace(last);env['epoch_slot']=epoch+1;state()
                print('EPOCH SLOT COMPLETE',category,epoch+1,flush=True)
            del enc,c64,c32,params,opt,modules,loader,datasets,x;gc.collect();torch.cuda.empty_cache()
            env['status']='evaluating';state();rows.append(evaluate(category,last,env));write(OUT/'category_metrics.csv',rows)
        env['status']='evaluated';state();subprocess.run([sys.executable,str(ROOT/'finish_psvdd.py')],check=True)
    except Exception as e:
        env.update(status='failed',error=repr(e));state();raise
if __name__=='__main__':main()
