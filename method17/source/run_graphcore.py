"""Evaluate author GraphCore code, paper-aligned public Pyramid ViG, 4/8 shot."""
import gc
import importlib.metadata
import importlib.util
import json
import random
import subprocess
import sys
import time
import numpy as np
import torch
from torchvision import transforms as T
from graphcore_common import *

def dataset_module():
    spec=importlib.util.spec_from_file_location('native_mvtec',REPO/'dataset/mvtec2d.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

class CachedModel:
    def __init__(self,runner,features):self.runner,self.features=runner,features
    def eval(self):return self
    def __call__(self,x):
        i=int(x.item());self.runner.features.extend([f[i:i+1] for f in self.features])

class CachedDataset(torch.utils.data.Dataset):
    def __init__(self,ds,indices):self.ds,self.indices=ds,indices
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):
        row=self.ds[self.indices[i]];row['img']=torch.tensor([i]);return row

def extract(model,ds,indices,device):
    buffers=[[],[]];current=[]
    handles=[model.backbone[i][-1].register_forward_hook(lambda m,x,y:current.append(y.detach().cpu())) for i in [4,11]]
    loader=torch.utils.data.DataLoader(torch.utils.data.Subset(ds,indices),batch_size=1,shuffle=False,num_workers=4,pin_memory=True)
    with torch.no_grad():
        for row in loader:
            current.clear();model(row['img'].to(device,non_blocking=True))
            assert [f.shape[1:] for f in current]==[torch.Size([96,28,28]),torch.Size([240,14,14])]
            for b,f in zip(buffers,current):b.append(f)
    for h in handles:h.remove()
    return [torch.cat(b) for b in buffers]

def main():
    OUT.mkdir(parents=True,exist_ok=True);RAW.mkdir(parents=True,exist_ok=True)
    assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==COMMIT
    assert not subprocess.check_output(['git','-C',str(REPO),'diff','--name-only'],text=True).strip()
    if (OUT/'verification.json').exists() and json.loads((OUT/'verification.json').read_text())['status']=='passed':return
    sys.path.insert(0,str(REPO))
    # NumPy 2 removed np.float, used only for sinusoidal position initialization.
    import models.graphcore.gcn_lib.pos_embed as pos
    pos.get_1d_sincos_pos_embed_from_grid.__globals__['np']=type('NPCompat',(),{'__getattr__':lambda self,k:np.float64 if k=='float' else getattr(np,k)})()
    from models.graphcore.pyramid_vig import pvig_ti_224_gelu
    GraphCore,Projection=native_class()
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    import faiss;faiss.omp_set_num_threads(8)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    model=pvig_ti_224_gelu(pretrained=False)
    saved=torch.load(WEIGHT,map_location='cpu',weights_only=True)
    weights=saved.get('state_dict',saved.get('model',saved))
    if all(k.startswith('module.') for k in weights):weights={k[7:]:v for k,v in weights.items()}
    model.load_state_dict(weights,strict=True);model=model.cuda().eval()
    module=dataset_module()
    transform=T.Compose([T.Resize((224,224)),T.CenterCrop(224),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
    mask_transform=T.Compose([T.Resize(224),T.CenterCrop(224),T.ToTensor()])
    train=module.MVTec2D('/home/test/data/mvtec',phase='train',data_transform=[transform,mask_transform])
    test=module.MVTec2D('/home/test/data/mvtec',phase='test',data_transform=[transform,mask_transform])
    env=dict(status='starting',official_repo='https://github.com/M-3LAB/open-iad',commit=COMMIT,
        provenance_label='author benchmark implementation with explicit paper-aligned configuration; not exact MuSc GraphCore implementation',
        gpu=torch.cuda.get_device_name(0),python=sys.version,cuda=torch.version.cuda,
        packages={p:importlib.metadata.version(p) for p in ['torch','torchvision','numpy','timm','faiss-cpu','scikit-learn','scipy']},
        model='pvig_ti_224_gelu',weight=str(WEIGHT),weight_sha256=sha(WEIGHT),strict_weight_loading=True,
        feature_layers_0based=[4,11],feature_definition='last FFN outputs of pyramid stages 2 and 3; inference choice, paper does not specify these feature taps',
        feature_shapes=[[96,28,28],[240,14,14]],sampler_percentage=.01,n_neighbours=9,local_smoothing=False,
        seed=42,rounds=10,shots=[4,8],support_selection='10 local random.sample draws without replacement, same 8-shot draw prefix for4',
        precision='FP32; TF32 disabled for matmul and cuDNN; deterministic algorithms; cuDNN benchmark disabled',
        image_transform='native Resize224x224 bilinear/CenterCrop224/ToTensor/ImageNet normalization',
        mask_transform='native Resize(shorter edge224) bilinear/CenterCrop224/ToTensor; >=0.5',
        graphcore_training='memory-bank construction only; no gradient training',backbone_pretraining='public ImageNet weights',
        original_checkout_edits=False,native_source_sha256=sha(REPO/'arch/graphcore.py'),wrapper_sha256=sha(__file__),
        feature_cache='category CPU features extracted once at native batch1 eval; same features reused across support rounds',
        aupro_source_sha256=pro_function()[1],raw_path=str(RAW),error=None)
    def state():(OUT/'environment.json').write_text(json.dumps(env,indent=2))
    state();rows=[]
    try:
        for cat_id,category in enumerate(module.mvtec2d_classes()):
            indices_train=[i for i,t in enumerate(train.task_ids_list) if t==cat_id]
            indices_test=[i for i,t in enumerate(test.task_ids_list) if t==cat_id]
            folder=OUT/category;folder.mkdir(exist_ok=True)
            rawfolder=RAW/category;rawfolder.mkdir(exist_ok=True)
            if all((folder/f'{s}shot_round{r}_metrics.json').exists() for s in [4,8] for r in range(10)):
                rows.extend(json.loads((folder/f'{s}shot_round{r}_metrics.json').read_text()) for s in [4,8] for r in range(10));continue
            env.update(status='extracting_features',category=category);state()
            start=time.perf_counter()
            train_features=extract(model,train,indices_train,'cuda');test_features=extract(model,test,indices_test,'cuda')
            torch.cuda.synchronize()
            print('CUDA FEATURES READY',category,round(time.perf_counter()-start,2),'seconds',flush=True)
            cache=rawfolder/'features.pt';torch.save(dict(train=train_features,test=test_features),cache)
            (folder/'feature_cache.json').write_text(json.dumps(dict(path=str(cache),sha256=sha(cache)),indent=2))
            masks=torch.stack([test[i]['mask'] for i in indices_test]).numpy()[:,0]
            masks=(masks>=.5).astype(np.uint8)
            labels=np.array([test.labels_list[i] for i in indices_test],dtype=np.int64)
            paths=[test.imgs_list[i] for i in indices_test]
            write(folder/'images.csv',[dict(image_path=p,label=int(y),sha256=sha(p),mask_path=test.masks_list[i] or '',mask_sha256=sha(test.masks_list[i]) if test.masks_list[i] else '') for p,y,i in zip(paths,labels,indices_test)])
            # Independent local draw manifest; do not imply the authors' support IDs.
            rng=random.Random(42+cat_id)
            supports=[rng.sample(range(len(indices_train)),8) for _ in range(10)]
            write(folder/'supports.csv',[dict(round=r,position=j,train_index=i,image_path=train.imgs_list[indices_train[i]],sha256=sha(train.imgs_list[indices_train[i]])) for r,ids in enumerate(supports) for j,i in enumerate(ids)])
            loader=torch.utils.data.DataLoader(CachedDataset(test,indices_test),batch_size=1,shuffle=False,num_workers=0)
            for shot in [4,8]:
                for round_id in range(10):
                    marker=folder/f'{shot}shot_round{round_id}_metrics.json'
                    if marker.exists():rows.append(json.loads(marker.read_text()));continue
                    env.update(status='building_memory_bank',shot=shot,round=round_id);state()
                    runner=GraphCore.__new__(GraphCore)
                    runner.config=dict(num_epochs=1,local_smoothing=False,sampler_percentage=.01,n_neighbours=9,data_crop_size=224)
                    runner.device=torch.device('cpu');runner.features=[]
                    runner.embedding_coreset=np.array([]);runner.embedding_path=str(rawfolder)
                    runner.random_projector=Projection(n_components='auto',eps=.9,random_state=42+round_id)
                    support=[{'img':torch.tensor([[i]])} for i in supports[round_id][:shot]]
                    runner.model=CachedModel(runner,train_features)
                    np.random.seed(42+round_id)
                    with torch.no_grad():runner.train_model(support,cat_id)
                    assert runner.embedding_coreset.shape==(int(shot*784*.01),336)
                    bankfile=rawfolder/f'{shot}shot_round{round_id}_bank.npy';np.save(bankfile,runner.embedding_coreset)
                    runner.model=CachedModel(runner,test_features)
                    env['status']='predicting_from_cached_features';state()
                    runner.prediction(loader,cat_id)
                    maps=np.asarray(runner.pixel_pred_list,dtype=np.float32)
                    scores=np.asarray(runner.img_pred_list,dtype=np.float32)
                    assert np.array_equal(labels,np.asarray(runner.img_gt_list)) and np.array_equal(masks,np.asarray(runner.pixel_gt_list))
                    assert maps.shape==masks.shape==(len(paths),224,224) and np.isfinite(maps).all() and np.isfinite(scores).all()
                    raw=rawfolder/f'{shot}shot_round{round_id}.npz'
                    np.savez_compressed(raw,image_labels=labels,image_scores=scores,anomaly_maps=maps,masks=masks)
                    (folder/f'{shot}shot_round{round_id}_raw.json').write_text(json.dumps(dict(path=str(raw),sha256=sha(raw),bank_path=str(bankfile),bank_sha256=sha(bankfile)),indent=2))
                    write(folder/f'{shot}shot_round{round_id}_scores.csv',[dict(image_path=p,label=int(y),score=float(s)) for p,y,s in zip(paths,labels,scores)])
                    env['status']='computing_metrics';state()
                    row=dict(dataset='mvtec',shot=shot,category=category,round=round_id,n_images=len(paths),**metrics(labels,scores,masks,maps))
                    assert all(np.isfinite(row[k]) and -1e-12<=row[k]<=1+1e-12 for k in KEYS)
                    marker.write_text(json.dumps(row,indent=2));rows.append(row);write(OUT/'round_metrics.csv',rows)
                    print('ROUND COMPLETE',category,shot,round_id,flush=True)
                    del runner;gc.collect()
            del train_features,test_features;gc.collect();torch.cuda.empty_cache()
        write(OUT/'round_metrics.csv',rows);env['status']='evaluated';state()
        subprocess.run([sys.executable,str(ROOT/'finish_graphcore.py')],check=True)
    except Exception as e:
        env.update(status='failed',error=repr(e));state();raise

if __name__=='__main__':main()
