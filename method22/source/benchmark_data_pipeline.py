"""Compare unchanged CutPaste augmentation with 4/8/12 CPU workers."""
import gc,json,os,signal,subprocess,time
import numpy as np
import torch
from torch.utils.data import DataLoader,RandomSampler
from run_cutpaste import Training,ProjectionNet,cut_paste_collate_fn,worker_seed,seed
from cutpaste_common import *
from cutpaste_pipeline import CUDAPrefetch

def main():
    owner=int(Path('/home/test/cutpaste_results/run_20261008.pid').read_text())
    assert b'run_cutpaste.py' in Path(f'/proc/{owner}/cmdline').read_bytes()
    group=os.getpgid(owner);os.killpg(group,signal.SIGSTOP)
    watchdog=subprocess.Popen([os.sys.executable,'-c',f'import os,signal,time;time.sleep(240);os.killpg({group},signal.SIGCONT)'],start_new_session=True)
    workers_only='--workers-only' in os.sys.argv
    overlap='--overlap' in os.sys.argv
    actual='--actual' in os.sys.argv
    prefetch_only='--prefetch' in os.sys.argv
    result=dict(training_pid=owner,base_batch=32,effective_batch=96,augment_unchanged=True,trials=[])
    try:
        env=json.loads((OUT/'environment.json').read_text());cat=env['category'];kind=env['model_kind']
        result.update(category=cat,kind=kind)
        if not (RAW/cat/kind/'last.pt').exists():kind='image';result['kind']=kind
        saved=torch.load(RAW/cat/kind/'last.pt',map_location='cpu',weights_only=False)
        result['checkpoint_step']=saved['step']
        torch.set_num_threads(8);torch.backends.cudnn.benchmark=True
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        model=ProjectionNet(pretrained=False,head_layers=[512,128],num_classes=3).cuda().to(memory_format=torch.channels_last)
        model.load_state_dict(saved['model']);model.train();runner=torch.compile(model) if (not actual or saved['proof'].get('compiled',False)) else model
        result['compiled_model']=not actual or saved['proof'].get('compiled',False)
        y=torch.arange(3,device='cuda').repeat_interleave(32)
        if prefetch_only:settings=[(4,'gpu_concat',8),(4,'prefetch',8),(4,'prefetch_fused',8)]
        elif actual:settings=[(4,'gpu_concat_sync',8),(4,'gpu_concat',8)]
        elif overlap:settings=[(4,'gpu_concat_sync',8),(4,'gpu_concat',8),(4,'gpu_concat_fused',8)]
        elif workers_only:settings=[(n,'cpu_concat',8) for n in [4,8,12]]
        else:settings=[(4,'cpu_concat',8),(4,'gpu_concat',8),(4,'gpu_concat',1),(8,'gpu_concat',1)]
        for workers,transfer,threads in settings:
            torch.set_num_threads(threads)
            seed();model.load_state_dict(saved['model'])
            opt=torch.optim.SGD(model.parameters(),lr=.0001,momentum=.9,weight_decay=3e-5,**({'fused':True} if transfer.endswith('_fused') else {'foreach':True}))
            scaler=torch.amp.GradScaler('cuda',init_scale=128.)
            ds=Training(read(OUT/cat/'train_images.csv'),kind)
            loader=DataLoader(ds,batch_size=32,sampler=RandomSampler(ds,replacement=True,num_samples=32*48),num_workers=workers,
                worker_init_fn=worker_seed,collate_fn=cut_paste_collate_fn,persistent_workers=True,pin_memory=True,prefetch_factor=4)
            iterator=CUDAPrefetch(loader) if transfer.startswith('prefetch') else iter(loader);times=[];waits=[];copies=[];gpu=[];losses=[];block_start=None
            for i in range(40):
                if i==8:
                    torch.cuda.synchronize();block_start=time.perf_counter()
                if not overlap:torch.cuda.synchronize()
                t=time.perf_counter();data=next(iterator);ready=time.perf_counter()
                if transfer.startswith('prefetch'):x=data
                elif transfer.startswith('gpu_concat'):
                    x=torch.cat([part.cuda(non_blocking=True) for part in data]).to(memory_format=torch.channels_last)
                    if i==0:
                        reference=torch.cat(data).cuda().to(memory_format=torch.channels_last)
                        assert torch.equal(x,reference);del reference
                else:x=torch.cat(data).cuda(non_blocking=True).to(memory_format=torch.channels_last)
                if not overlap:torch.cuda.synchronize()
                copied=time.perf_counter()
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):loss=torch.nn.functional.cross_entropy(runner(x)[1],y)
                if transfer=='gpu_concat_sync':assert torch.isfinite(loss)
                else:torch._assert_async(torch.isfinite(loss))
                scaler.scale(loss).backward();scaler.step(opt);scaler.update()
                losses.append(loss.detach())
                if not overlap or transfer=='gpu_concat_sync':float(loss.detach())
                if not overlap:torch.cuda.synchronize()
                end=time.perf_counter()
                if i>=8:
                    times.append(end-t);waits.append(ready-t);copies.append(copied-ready);gpu.append(end-copied)
            torch.cuda.synchronize();block_seconds=time.perf_counter()-block_start
            loss_values=torch.stack(losses).cpu();assert torch.isfinite(loss_values).all()
            if transfer.startswith('prefetch'):iterator.close()
            row=dict(sustained_step_seconds=block_seconds/32,workers=workers,transfer=transfer,main_threads=threads,exact_input_verified=transfer=='gpu_concat',measured_steps=32,mean_step_seconds=float(np.mean(times)),median_step_seconds=float(np.median(times)),
                mean_data_wait_seconds=float(np.mean(waits)),mean_concat_transfer_seconds=float(np.mean(copies)),mean_gpu_seconds=float(np.mean(gpu)))
            result['trials'].append(row);print('DATA PIPELINE TRIAL',json.dumps(row),flush=True)
            del iterator,loader,ds,opt,scaler;gc.collect()
        key='sustained_step_seconds' if overlap else 'mean_step_seconds'
        best=min(result['trials'],key=lambda r:r[key]);baseline=result['trials'][0]
        result.update(selected=best if best[key]<baseline[key]*.95 else baseline,
            best_speedup=baseline[key]/best[key],limit='Short trained-checkpoint trials; worker RNG streams differ; excludes saving and evaluation.')
    finally:
        os.killpg(group,signal.SIGCONT);watchdog.terminate();result['real_training_resumed']=True
        filename='data_overlap_benchmark.json' if prefetch_only else ('data_async_actual_benchmark.json' if actual else ('data_sync_benchmark.json' if overlap else ('data_pipeline_benchmark.json' if workers_only else 'data_transfer_benchmark.json')))
        (OUT/filename).write_text(json.dumps(result,indent=2))
        print('REAL TRAINING RESUMED',owner,flush=True)

if __name__=='__main__':main()
