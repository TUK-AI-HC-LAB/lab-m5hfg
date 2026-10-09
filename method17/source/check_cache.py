"""Compare real-image GPU native prediction with cached-feature prediction."""
import json
import os
import sys
import numpy as np
import torch
from torchvision import transforms as T
from graphcore_common import *
from run_graphcore import CachedModel,CachedDataset,dataset_module
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')

sys.path.insert(0,str(REPO))
import models.graphcore.gcn_lib.pos_embed as pos
pos.get_1d_sincos_pos_embed_from_grid.__globals__['np']=type('NPCompat',(),{'__getattr__':lambda self,k:np.float64 if k=='float' else getattr(np,k)})()
from models.graphcore.pyramid_vig import pvig_ti_224_gelu
torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
torch.manual_seed(42);torch.cuda.manual_seed_all(42)
model=pvig_ti_224_gelu(pretrained=False)
saved=torch.load(WEIGHT,map_location='cpu',weights_only=True)
state=saved.get('state_dict',saved.get('model',saved))
if all(k.startswith('module.') for k in state):state={k[7:]:v for k,v in state.items()}
model.load_state_dict(state,strict=True);model=model.cuda().eval()
module=dataset_module()
transform=T.Compose([T.Resize((224,224)),T.CenterCrop(224),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
mask_transform=T.Compose([T.Resize(224),T.CenterCrop(224),T.ToTensor()])
ds=module.MVTec2D('/home/test/data/mvtec',phase='test',data_transform=[transform,mask_transform])
ids=[i for i,t in enumerate(ds.task_ids_list) if t==0][:6]
features=torch.load(RAW/'bottle/features.pt',map_location='cpu',weights_only=True)['test']
captured=[]
probe_hooks=[model.backbone[i][-1].register_forward_hook(lambda m,x,y:captured.append(y.detach().cpu())) for i in [4,11]]
with torch.no_grad():model(ds[ids[0]]['img'][None].cuda())
print('FEATURE DIFFERENCES',[float((a-b[0:1]).abs().max()) for a,b in zip(captured,features)],flush=True)
for h in probe_hooks:h.remove()
bank=np.load(RAW/'bottle/4shot_round0_bank.npy')
GraphCore,_=native_class()
import faiss
runner=GraphCore.__new__(GraphCore);runner.config=dict(local_smoothing=False,n_neighbours=9,data_crop_size=224)
runner.device=torch.device('cuda');runner.features=[];runner.model=model
runner.index=faiss.IndexFlatL2(bank.shape[1]);runner.index.add(bank)
hooks=[model.backbone[i][-1].register_forward_hook(lambda m,x,y:runner.features.append(y)) for i in [4,11]]
runner.prediction(torch.utils.data.DataLoader(torch.utils.data.Subset(ds,ids),batch_size=1),0)
direct_maps=np.asarray(runner.pixel_pred_list);direct_scores=np.asarray(runner.img_pred_list)
for h in hooks:h.remove()
runner.device=torch.device('cpu');runner.model=CachedModel(runner,features)
runner.prediction(torch.utils.data.DataLoader(CachedDataset(ds,ids),batch_size=1),0)
cached_maps=np.asarray(runner.pixel_pred_list);cached_scores=np.asarray(runner.img_pred_list)
print('DIRECT SCORES',direct_scores,'CACHED SCORES',cached_scores,'MAX MAP DIFF',np.abs(direct_maps-cached_maps).max(),flush=True)
assert np.allclose(direct_scores,cached_scores,rtol=1e-5,atol=1e-5)
assert np.allclose(direct_maps,cached_maps,rtol=1e-5,atol=1e-5)
proof=dict(status='passed',n_images=6,category='bottle',shot=4,round=0,
    max_abs_image_score=float(np.abs(direct_scores-cached_scores).max()),max_abs_anomaly_map=float(np.abs(direct_maps-cached_maps).max()),
    all_test_images_bitwise_verified=False,rtol=1e-5,atol=1e-5,feature_cache_sha256=sha(RAW/'bottle/features.pt'),bank_sha256=sha(RAW/'bottle/4shot_round0_bank.npy'))
(OUT/'native_cache_comparison.json').write_text(json.dumps(proof,indent=2))
print(json.dumps(proof),flush=True)
