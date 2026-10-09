"""Short current-checkpoint speed trials; preserve and resume real training."""
import copy,gc,json,os,signal,subprocess,sys,time,traceback
import numpy as np
import torch
from cutpaste_common import *
from run_cutpaste import Training,ProjectionNet,cut_paste_collate_fn,seed

os.environ['TORCHINDUCTOR_COMPILE_THREADS']='2'
torch.set_num_threads(8);torch.backends.cudnn.benchmark=True
env=json.loads((OUT/'environment.json').read_text());cat=env['category'];kind=env['model_kind']
owner=int(Path('/home/test/cutpaste_results/run_20261008.pid').read_text())
assert 'run_cutpaste.py' in Path(f'/proc/{owner}/cmdline').read_text()
result=dict(category=cat,kind=kind,gpu=torch.cuda.get_device_name(),trials=[],training_pid=owner,
    limit='GPU training on a fixed actual batch; excludes data loading, checkpoints and full-run accuracy. No restart or precision change of real training.')
os.kill(owner,signal.SIGSTOP)
# Failsafe if this benchmark is interrupted before its finally block.
watchdog=subprocess.Popen([sys.executable,'-c',f'import os,signal,time; time.sleep(600); os.kill({owner},signal.SIGCONT)'],start_new_session=True)
try:
    seed();rows=read(OUT/cat/'train_images.csv');ds=Training(rows,kind)
    data=cut_paste_collate_fn([ds[i%len(ds)] for i in range(32)])
    x=torch.cat(data).cuda().to(memory_format=torch.channels_last);y=torch.arange(3,device='cuda').repeat_interleave(32)
    path=RAW/cat/kind/'last.pt';saved=torch.load(path,map_location='cpu',weights_only=False)
    state=saved['model'];result.update(checkpoint_sha256=sha(path),checkpoint_step=saved['step'],effective_batch=96,input_size=int(x.shape[-1]))
    del saved,ds
    model=ProjectionNet(pretrained=False,head_layers=[512,128],num_classes=3).cuda().to(memory_format=torch.channels_last)
    model.load_state_dict(state);model.train()
    reference_grad=None;reference_loss=None
    for name,dtype,tf32,compile_it in [('fp32',None,False,False),('tf32',None,True,False),('bf16',torch.bfloat16,True,False),('fp16',torch.float16,True,False),('compiled_bf16',torch.bfloat16,True,True)]:
        try:
            model.load_state_dict(state);model.zero_grad(set_to_none=True)
            torch.backends.cuda.matmul.allow_tf32=tf32;torch.backends.cudnn.allow_tf32=tf32
            opt=torch.optim.SGD(model.parameters(),lr=.003,momentum=.9,weight_decay=3e-5,foreach=True)
            scaler=torch.amp.GradScaler('cuda',enabled=dtype==torch.float16)
            runner=torch.compile(model) if compile_it else model
            def step(update=True):
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=dtype or torch.bfloat16,enabled=dtype is not None):loss=torch.nn.functional.cross_entropy(runner(x)[1],y)
                scaler.scale(loss).backward();scaler.unscale_(opt)
                if update:scaler.step(opt);scaler.update()
                return loss
            torch.cuda.synchronize();t=time.perf_counter();value=step(False);torch.cuda.synchronize();startup=time.perf_counter()-t
            gradients=[p.grad.detach() for p in model.parameters() if p.grad is not None]
            finite=all(torch.isfinite(g).all() for g in gradients) and bool(torch.isfinite(value))
            if name=='fp32':reference_grad=[g.clone() for g in gradients];reference_loss=float(value.detach())
            rel=float(torch.sqrt(sum((a-b).square().sum() for a,b in zip(gradients,reference_grad)))/torch.sqrt(sum(g.square().sum() for g in reference_grad)).clamp_min(1e-12))
            loss_rel=abs(float(value.detach())-reference_loss)/max(abs(reference_loss),1e-12)
            # GradScaler cannot be unscaled twice without resetting its stage.
            if dtype==torch.float16:scaler.update()
            for _ in range(3):step()
            timings=[]
            for _ in range(20):
                torch.cuda.synchronize();t=time.perf_counter();v=step();torch.cuda.synchronize();timings.append(time.perf_counter()-t)
            trial=dict(name=name,status='passed' if finite else 'nonfinite',median_step_seconds=float(np.median(timings)),
                startup_seconds=startup,loss_relative_difference=loss_rel,gradient_relative_L2_difference=rel,finite_gradients=finite)
            result['trials'].append(trial);print('TRIAL COMPLETE',json.dumps(trial),flush=True)
            del opt,scaler,runner,value,gradients
        except Exception:
            result['trials'].append(dict(name=name,status='failed',error=traceback.format_exc()));print('TRIAL FAILED',name,flush=True)
        gc.collect();torch.cuda.empty_cache()
finally:
    os.kill(owner,signal.SIGCONT);watchdog.terminate()
    result['real_training_resumed']=True
    (OUT/'acceleration_benchmark.json').write_text(json.dumps(result,indent=2))
    print('REAL TRAINING RESUMED',owner,flush=True)
