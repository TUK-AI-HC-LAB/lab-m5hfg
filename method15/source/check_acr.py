"""One real-feature CUDA step and native detection; isolated from production models."""
import json
import os
import sys
import numpy as np
import torch
from prepare_acr import OUT,RAW,RUNTIME
sys.path.insert(0,str(RUNTIME));os.chdir(RUNTIME)
from config.parser import model_config_reader
from data_loader.mvtec import MVTecFeature
from trainers.zeroshot_trainer import ZeroShotMetaTrainer
torch.set_num_threads(8);torch.manual_seed(42);np.random.seed(42)
config=model_config_reader('config_mvtec.yml')
features=RAW/'features/wide_resnet50_2'
load=lambda p:torch.load(features/p,weights_only=False,mmap=True)
loader=MVTecFeature.__new__(MVTecFeature)
loader.batchsz=32;loader.k_query=30;loader.n_way=2;loader.qry_anomaly_ratio=.5
train=[load('train_cable.pt')];test=load('test_bottle.pt')
loader.test_image_num=len(test)
loader.test_gt_list=load('test_bottle_gt.pt');loader.test_gt_mask_list=load('test_bottle_gt_mask.pt')
loader.datasets_cache={'train':loader._load_data_cache_train(train),'test':loader._load_data_cache_test(test)}
loader.datasets={'train':train,'test':test};loader.indexes={'train':0,'test':0}
model=config['model_class'](config=config)
trainer=ZeroShotMetaTrainer(model,config['loss_class'](config=config),str(RAW/'preflight'),config)
optimizer=torch.optim.Adam(model.parameters(),lr=.0003)
loss=trainer._train(1,loader,optimizer)
assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
native=trainer.detect_outliers(loader,allowed_task_num=-1)
assert np.isfinite(trainer.last_scores).all() and trainer.last_scores.shape==(len(test),224,224)
(OUT/'cuda_preflight.json').write_text(json.dumps(dict(status='passed',device=str(trainer.device),finite_gradients=True,
    native_detection_completed=True,n_images=len(test),loss=float(loss.detach()),native_auroc=list(native),
    scope='isolated one-step smoke test on cable train and bottle test; does not replace full14-category,50-update fits'),indent=2))
print('ACR CUDA PREFLIGHT PASSED')
