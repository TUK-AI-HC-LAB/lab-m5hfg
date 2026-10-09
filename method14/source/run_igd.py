"""Train official local/global IGD and automatically evaluate all categories."""
import importlib
import importlib.metadata
import json
import os
import random
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import torch
from common import ROOT,REPO,RAW,OUT,COMMIT,CATEGORIES,prepare,setup,sha,SilentRange,ACCELERATED
from accelerate_igd import fast_model,fast_module
from resume_igd import enable_resume

prepare();setup()
BATCHES={32:int(os.environ.get('IGD_LOCAL_BATCH','2')),256:int(os.environ.get('IGD_GLOBAL_BATCH','16'))}
assert all(v>0 for v in BATCHES.values())
assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==COMMIT
assert not subprocess.check_output(['git','-C',str(REPO),'diff','--name-only'],text=True).strip()
env=dict(status='starting',official_repo='https://github.com/tianyu0207/IGD',commit=COMMIT,seed=42,
    seed_definition='local reproducibility choice; official job has no seed flag',
    gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,python=sys.version,
    packages={p:importlib.metadata.version(p) for p in ['torch','torchvision','numpy','Pillow','pytorch-msssim','tensorboardX']},
    training_executed=True,scales=[32,256],sample_rate=1.0,nominal_epochs=256,
    iterations='int(train_size/BATCH_SIZE*MAX_EPOCH), official train loops',batch_size={str(k):v for k,v in BATCHES.items()},
    batch_protocol='user-authorized larger batch; reduced optimizer updates' if BATCHES!={32:2,256:16} else 'official batches',
    train_patch_stride=16,precision='BF16 model autocast; FP32 losses and Gaussian statistics' if ACCELERATED else 'FP32; no AMP',tf32=ACCELERATED,
    cudnn_benchmark=ACCELERATED,channels_last=ACCELERATED,persistent_workers=ACCELERATED,
    prefetch_factor=4 if ACCELERATED else 2,
    fused_adam=ACCELERATED,non_blocking_transfer=ACCELERATED,
    loss_compile=bool(ACCELERATED and (OUT/'compile_preflight.json').exists() and json.loads((OUT/'compile_preflight.json').read_text()).get('enabled')),
    checkpoint_selection='final iteration; no selection by test AUROC',
    pretrained_encoder_sha256=sha(REPO/'Encoder_KD_ckpt'),
    runtime_source_manifest=str(OUT/'runtime_sources.json'),compatibility_patch=str(OUT/'compatibility.patch'),
    compatibility=['dataset paths redirected','iterator next method replaced by next()',
    'MS-SSIM size guard uses actual scale count; computations unchanged','Adam beta 0 converted to float 0.0',
    'recording/tqdm display disabled; full validation preserved'],
    original_checkout_edits=False,raw_output=str(RAW),eta_calculation=False,
    wrapper_sha256=sha(__file__))
def state():(OUT/'environment.json').write_text(json.dumps(env,indent=2))
state()
try:
    for scale in [32,256]:
        module=importlib.import_module(f'p{scale}.ssim_main')
        module.BATCH_SIZE=BATCHES[scale]
        module.tqdm=SilentRange
        module.recorder=None
        if ACCELERATED:fast_module(module)
        for category in CATEGORIES:
            destination=RAW/'checkpoints'/f'p{scale}'/category
            destination.mkdir(parents=True,exist_ok=True);(destination/'optimizer').mkdir(exist_ok=True)
            marker=OUT/f'p{scale}'/category/'training_complete.json'
            marker.parent.mkdir(parents=True,exist_ok=True)
            if marker.exists():continue
            env.update(status='training',current_scale=scale,current_category=category);state()
            random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
            generator=(module.twoin1Generator(64,latent_dimension=128) if scale==32 else module.twoin1Generator256(64,latent_dimension=128)).cuda()
            discriminator=(module.VisualDiscriminator(64) if scale==32 else module.VisualDiscriminator256(64)).cuda()
            if ACCELERATED:
                fast_model(generator);fast_model(discriminator)
            for p in generator.pretrain.parameters():p.requires_grad=False
            module.generator=generator;module.ckpt_path=str(destination)
            opt_g=torch.optim.Adam(generator.parameters(),lr=module.LR,betas=(0.0,.9),weight_decay=1e-6,fused=ACCELERATED)
            opt_d=torch.optim.Adam(discriminator.parameters(),lr=module.LR,betas=(0.0,.9),fused=ACCELERATED)
            original_train=module.train
            resume_path=destination/'recovery.pt'
            if resume_path.exists():
                saved=torch.load(resume_path,map_location='cuda',weights_only=True)
                if 'discriminator' not in saved:
                    archived=destination/f"recovery_interrupted_{saved['iteration']}.pt"
                    if not archived.exists():resume_path.rename(archived)
                    env.setdefault('interrupted_runs',[]).append(dict(category=category,scale=scale,
                        archived_checkpoint=str(archived),action='restart category; old checkpoint lacks discriminator state'))
                    state()
                    saved=None
            else:
                saved=None
            if saved is not None:
                assert saved.get('batch_size',2 if scale==32 else 16)==BATCHES[scale], 'checkpoint batch mismatch; use a separate run tag'
                generator.load_state_dict(saved['model']);generator.c=saved['c'];generator.sigma=saved['sigma']
                discriminator.load_state_dict(saved['discriminator'])
                if 'optimizer_g' in saved:
                    opt_g.load_state_dict(saved['optimizer_g']);opt_d.load_state_dict(saved['optimizer_d'])
                else:
                    opt_g.load_state_dict(torch.load(destination/'optimizer/g_opt.pth',map_location='cuda',weights_only=True))
                    opt_d.load_state_dict(torch.load(destination/'optimizer/d_opt.pth',map_location='cuda',weights_only=True))
                start=int(saved['iteration'])+1
                enable_resume(module,start)
                env.setdefault('resumes',[]).append(dict(category=category,scale=scale,start_iteration=start,
                    checkpoint_sha256=sha(resume_path),rng_restored=False,
                    limitation='shuffle/RNG restart; not bitwise equivalent to uninterrupted training'))
                state()
            native=module.validation
            def validate(*a,**kw):
                proof=OUT/f'p{scale}'/category/'first_training_validation.json'
                if ACCELERATED and not proof.exists():
                    assert all(torch.isfinite(p.grad).all() for p in generator.parameters() if p.grad is not None)
                    assert all(torch.isfinite(p.grad).all() for p in discriminator.parameters() if p.grad is not None)
                result=native(*a,**kw)
                iteration=a[1]
                checkpoint=dict(model=generator.state_dict(),c=generator.c,sigma=generator.sigma,
                    iteration=iteration,scale=scale,category=category,batch_size=BATCHES[scale],
                    optimizer_g=opt_g.state_dict(),optimizer_d=opt_d.state_dict(),
                    discriminator=discriminator.state_dict())
                path=destination/('final.pt' if a[6] else 'recovery.pt')
                torch.save(checkpoint,path)
                if not proof.exists():proof.write_text(json.dumps(dict(iteration=iteration,training_step_completed=True,
                    validation_auroc=float(result[0]),device=str(next(generator.parameters()).device),
                    accelerated=ACCELERATED,finite_gradients_checked=ACCELERATED),indent=2))
                return result
            module.validation=validate
            module.train(SimpleNamespace(sample_rate=1.0),category,generator,discriminator,opt_g,opt_d)
            module.validation=native
            module.train=original_train
            path=destination/'final.pt'
            assert path.exists()
            marker.write_text(json.dumps(dict(status='completed',checkpoint=str(path),sha256=sha(path),
                scale=scale,category=category,nominal_epochs=256,seed=42,batch_size=BATCHES[scale]),indent=2))
            del generator,discriminator,opt_g,opt_d
            torch.cuda.empty_cache()
    env.update(status='evaluating');state()
    subprocess.run([sys.executable,str(ROOT/'evaluate_igd.py')],check=True)
    env.update(status='completed');state()
    subprocess.run([sys.executable,str(ROOT/'finish_igd.py')],check=True)
except Exception as error:
    env.update(status='failed',error=repr(error));state();raise
