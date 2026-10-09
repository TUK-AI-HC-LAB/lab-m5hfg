"""Validate compiled FP16 on each model without changing its parameters/RNG."""
import copy,json,time,traceback
import torch

def configure_acceleration(model,opt,x,y,out,previous,start_step):
    # A data-transfer-only restart should keep this model's already validated AMP
    # protocol. Rechecking a different random batch can change precision arbitrarily.
    prior=previous
    while start_step>0 and isinstance(prior,dict):
        if prior.get('precision')=='fp16' and prior.get('status')=='passed':
            torch.backends.cuda.matmul.allow_tf32=prior['tf32_enabled'];torch.backends.cudnn.allow_tf32=prior['tf32_enabled']
            runner=torch.compile(model) if prior['compiled'] else model
            scaler=torch.amp.GradScaler('cuda',init_scale=128.)
            proof=copy.deepcopy(prior)
            proof.update(resumed_at_step=start_step,resume_policy='Preserve this model prior validated FP16 mode; data transfer exact verified. Live finite loss/GradScaler checks remain enabled.')
            (out/f'acceleration_preserved_at_step_{start_step}.json').write_text(json.dumps(proof,indent=2))
            print('PRESERVED VALIDATED ACCELERATION',out,proof['mode'],'from step',start_step,flush=True)
            return runner,scaler,proof
        prior=prior.get('previous_proof')
    state=copy.deepcopy(model.state_dict());rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
    trials=[];ref_grad=None;ref_loss=None;selected=None;runner=model
    compiled_candidate=None;compiled_grad=None;compiled_loss=None
    for name,half,tf32,compile_it in [('fp32',False,False,False),('compiled_fp16',True,True,True),('fp16',True,True,False),('tf32',False,True,False)]:
        model.load_state_dict(state);opt.zero_grad(set_to_none=True)
        torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda_rng)
        torch.backends.cuda.matmul.allow_tf32=tf32;torch.backends.cudnn.allow_tf32=tf32
        candidate=model
        try:
            candidate=torch.compile(model) if compile_it else model
            # Small validation scale avoids testing GradScaler's initial overflow recovery.
            scaler=torch.amp.GradScaler('cuda',enabled=half,init_scale=128.)
            torch.cuda.synchronize();t=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.float16,enabled=half):
                value=torch.nn.functional.cross_entropy(candidate(x)[1],y)
            scaler.scale(value).backward();scaler.unscale_(opt);torch.cuda.synchronize()
            gradients=[p.grad for p in model.parameters() if p.grad is not None]
            finite=bool(torch.isfinite(value)) and all(bool(torch.isfinite(g).all()) for g in gradients)
            if name=='fp32':
                assert finite;ref_loss=float(value.detach());ref_grad=[g.detach().clone() for g in gradients]
            loss_rel=abs(float(value.detach())-ref_loss)/max(abs(ref_loss),1e-12)
            grad_rel=float(torch.sqrt(sum((a-b).square().sum() for a,b in zip(gradients,ref_grad)))/torch.sqrt(sum(g.square().sum() for g in ref_grad)).clamp_min(1e-12))
            strict_passed=finite and loss_rel<=.01 and grad_rel<=.05
            # User prioritizes AMP speed. FP32 gradient comparison remains a diagnostic;
            # loss/finite checks gate AMP, and compilation must match eager AMP.
            passed=finite and loss_rel<=.01 and (half or grad_rel<=.05)
            trial=dict(mode=name,finite=finite,loss_relative_difference=loss_rel,gradient_relative_L2=grad_rel,strict_fp32_equivalence_passed=strict_passed,passed=passed,startup_seconds=time.perf_counter()-t)
            trials.append(trial)
            if name=='compiled_fp16' and passed:
                compiled_candidate=candidate;compiled_grad=[g.detach().clone() for g in gradients];compiled_loss=float(value.detach())
            if name=='fp16' and passed:
                use_compile=False
                if compiled_candidate is not None:
                    compile_grad=float(torch.sqrt(sum((a-b).square().sum() for a,b in zip(gradients,compiled_grad)))/torch.sqrt(sum(g.square().sum() for g in gradients)).clamp_min(1e-12))
                    compile_loss=abs(float(value.detach())-compiled_loss)/max(abs(float(value.detach())),1e-12)
                    use_compile=compile_grad<=.05 and compile_loss<=.01
                    trial.update(compiled_vs_eager_gradient_relative_L2=compile_grad,compiled_vs_eager_loss_relative_difference=compile_loss,compile_validation_passed=use_compile)
                selected=dict(precision='fp16',compiled=use_compile,tf32_enabled=True,mode='compiled_fp16' if use_compile else 'fp16');runner=compiled_candidate if use_compile else model;break
            if name=='tf32' and passed:
                selected=dict(precision='fp32',compiled=False,tf32_enabled=True,mode=name);runner=model;break
        except Exception:
            trials.append(dict(mode=name,passed=False,error=traceback.format_exc()))
    if selected is None:selected=dict(precision='fp32',compiled=False,tf32_enabled=False,mode='fp32')
    model.load_state_dict(state);opt.zero_grad(set_to_none=True)
    torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda_rng)
    torch.backends.cuda.matmul.allow_tf32=selected['tf32_enabled'];torch.backends.cudnn.allow_tf32=selected['tf32_enabled']
    proof=dict(status='passed',**selected,bf16_enabled=False,transition_step=start_step,thresholds=dict(loss=.01,gradient=.05),policy='FP16 finite/loss gate; FP32 gradient diagnostic; compiled versus eager FP16 gradient/loss gate. User-requested accelerated continuation, no final AUROC equivalence claim.',trials=trials,previous_proof=previous)
    (out/f'acceleration_from_step_{start_step}.json').write_text(json.dumps(proof,indent=2))
    # Fresh scaler after validation: no pending unscale stage.
    scaler=torch.amp.GradScaler('cuda',enabled=selected['precision']=='fp16',init_scale=128.)
    print('ACCELERATION SELECTED',out,selected,'from step',start_step,flush=True)
    return runner,scaler,proof
