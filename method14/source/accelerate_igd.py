"""Fast model execution, FP32 loss/statistics, and persistent data workers."""
import functools
import json
import torch

def fast_model(model):
    model.to(memory_format=torch.channels_last)
    for name in ['forward','encoder','generate']:
        if not hasattr(model,name):continue
        original=getattr(model,name)
        def accelerated(*args,_original=original,**kwargs):
            args=tuple(x.contiguous(memory_format=torch.channels_last) if torch.is_tensor(x) and x.ndim==4 else x for x in args)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                result=_original(*args,**kwargs)
            # Keep downstream Gaussian/MS-SSIM/loss calculations FP32.
            if torch.is_tensor(result):return result.float()
            return result
        setattr(model,name,accelerated)
    return model

def fast_module(module):
    from common import OUT
    proof=OUT/'compile_preflight.json'
    if proof.exists() and json.loads(proof.read_text()).get('enabled'):
        compiled=torch.compile(module.ms_ssim,mode='default',dynamic=False)
        original_ssim=module.ms_ssim
        def safe_compiled(*a,**kw):
            try:return compiled(*a,**kw)
            except Exception as error:
                (OUT/'compile_runtime_fallback.json').write_text(json.dumps(dict(error=repr(error),
                    fallback='eager same loss; other accelerations remain enabled'),indent=2))
                module.ms_ssim=original_ssim
                return original_ssim(*a,**kw)
        module.ms_ssim=safe_compiled
    module.num_worker=8
    # Loader CPU work overlaps GPU work. Persist workers across iterator resets.
    original_loader=torch.utils.data.DataLoader
    class Loader(original_loader):
        def __init__(self,*a,**kw):
            if kw.get('num_workers',0)>0:
                kw.update(persistent_workers=True,prefetch_factor=4)
            kw['pin_memory']=True
            super().__init__(*a,**kw)
    # Only loaders constructed by this module's load_train/load_test functions
    # use the wrapper; keep the globally configured class otherwise intact.
    original_train=module.load_train
    original_test=module.load_test
    def invoke(fn,*a,**kw):
        previous=torch.utils.data.DataLoader
        torch.utils.data.DataLoader=Loader
        try:return fn(*a,**kw)
        finally:torch.utils.data.DataLoader=previous
    module.load_train=lambda *a,**kw:invoke(original_train,*a,**kw)
    module.load_test=lambda *a,**kw:invoke(original_test,*a,**kw)
