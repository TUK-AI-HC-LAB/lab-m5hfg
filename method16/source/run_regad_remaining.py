"""RegAD MVTec32 support extension and fixed-checkpoint BTAD4 transfer."""
import json,random,subprocess,sys,time,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset,DataLoader
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from prepare_regad import REPO,ASSETS,COMMIT,sha

ROOT=Path(__file__).resolve().parent;LAB=ROOT.parents[1]
OUT=ROOT/'results/remaining_mvtec32_btad4_20261010';RAW=Path('/home/test/regad_results/remaining_mvtec32_btad4_20261010')
RUNTIME=RAW/'runtime'
DATA={'mvtec':Path('/home/test/data/mvtec'),'btad':Path('/home/test/data/btad_original/BTech_Dataset_transformed')}
XFORM=T.Compose([T.Resize((224,224),interpolation=T.InterpolationMode.LANCZOS),T.ToTensor()])
MFORM=T.Resize((224,224),interpolation=T.InterpolationMode.NEAREST)
def write(p,rows):
    import csv
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def manifest(ds,cat,phase):
    meta=json.loads((DATA[ds]/'meta.json').read_text());rows=[]
    for r in meta[phase][cat]:
        p=DATA[ds]/r['img_path'];mp=DATA[ds]/r['mask_path'] if r.get('mask_path') else None
        rows.append(dict(image_path=str(p),label=int(r['anomaly']),sha256=sha(p),mask_path=str(mp) if mp else '',mask_sha256=sha(mp) if mp else ''))
    return rows
class Images(Dataset):
    def __init__(self,rows):self.rows=rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        with Image.open(self.rows[i]['image_path']) as im:return XFORM(im.convert('RGB'))
def embed(stn):
    layers=[stn.stn1_output,stn.stn2_output,stn.stn3_output]
    return torch.cat([layers[0],F.interpolate(layers[1],size=(56,56),mode='nearest'),F.interpolate(layers[2],size=(56,56),mode='nearest')],1).flatten(2)
@torch.no_grad()
def support_features(stn,support,path,funcs):
    augmented=[support]
    for angle in [-np.pi/4,-3*np.pi/16,-np.pi/8,-np.pi/16,np.pi/16,np.pi/8,3*np.pi/16,np.pi/4]:augmented.append(funcs.rot_img(support,angle))
    for a,b in [( .2,.2),(-.2,.2),(-.2,-.2),(.2,-.2),(.1,.1),(-.1,.1),(-.1,-.1),(.1,-.1)]:augmented.append(funcs.translation_img(support,a,b))
    augmented += [funcs.hflip_img(support),funcs.grey_img(support)]
    for k in [1,2,3]:augmented.append(funcs.rot90_img(support,k))
    images=torch.cat(augmented);images=images[torch.randperm(len(images))]
    features=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(len(images),448,3136))
    for start in range(0,len(images),16):
        stn(images[start:start+16].cuda());features[start:start+16]=embed(stn).cpu().numpy()
        if start==0:
            x=stn.stn1_output;y=stn.stn2_output;z=stn.stn3_output
            original=funcs.embedding_concat(funcs.embedding_concat(x[:1],y[:1],True),z[:1],True).flatten(2)
            assert torch.equal(original,embed(stn)[:1])
    features.flush();return features
@torch.no_grad()
def query_features(stn,rows,path):
    a=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(len(rows),448,3136));offset=0
    for x in DataLoader(Images(rows),batch_size=1,num_workers=4,pin_memory=True,persistent_workers=True):
        stn(x.cuda(non_blocking=True));a[offset:offset+len(x)]=embed(stn).cpu().numpy();offset+=len(x)
    assert offset==len(rows);a.flush();return a
@torch.no_grad()
def gaussian_score(train,query):
    result=np.empty((len(query),3136),dtype=np.float32);proof=[]
    for start in range(0,3136,16):
        stop=min(start+16,3136)
        a=torch.from_numpy(np.array(train[:,:,start:stop],copy=True)).permute(2,0,1).cuda()
        mean=a.mean(1);delta=a-mean[:,None,:]
        cov=(delta.transpose(1,2)@delta)/(len(train)-1)+.01*torch.eye(448,device='cuda')
        inv=torch.linalg.inv(cov)
        q=torch.from_numpy(np.array(query[:,:,start:stop],copy=True)).permute(2,0,1).cuda();d=q-mean[:,None,:]
        squared=((d@inv)*d).sum(-1);assert torch.isfinite(squared).all() and float(squared.min())>-1e-3
        distances=squared.clamp_min(0).sqrt();result[:,start:stop]=distances.T.cpu().numpy()
        if start in [0,1568,3120]:
            # First actual position in three chunks, against official scalar code.
            native_cov=torch.cov(a[0].T)+.01*torch.eye(448,device='cuda')
            assert torch.allclose(cov[0],native_cov,rtol=1e-4,atol=1e-4)
            ni=torch.linalg.inv(native_cov)
            for i in [0,len(query)//2,len(query)-1]:
                diff=q[0,i]-a[0].mean(0);ref=torch.dot(diff,ni@diff).sqrt()
                assert torch.allclose(distances[0,i],ref,rtol=1e-3,atol=1e-3)
                proof.append(dict(position=start,image=i,batch=float(distances[0,i]),native=float(ref)))
    return result.reshape(-1,56,56),proof
def postprocess(d):
    m=F.interpolate(torch.from_numpy(d).unsqueeze(1),size=(224,224),mode='bilinear',align_corners=False).squeeze(1).numpy()
    m=np.stack([gaussian_filter(x,sigma=4) for x in m]);assert m.max()>m.min()
    return (m-m.min())/(m.max()-m.min())
def masks(rows):
    result=[]
    for r in rows:
        mask=np.zeros((224,224),dtype=np.uint8)
        if r['mask_path']:
            with Image.open(r['mask_path']) as im:mask=(np.array(MFORM(im.convert('L')))>128).astype(np.uint8)
        result.append(mask)
    return np.stack(result)
def evaluate(stn,funcs,ds,cat,shot,round_id,checkpoint):
    out=OUT/ds/cat;raw=RAW/ds/cat;out.mkdir(parents=True,exist_ok=True);raw.mkdir(parents=True,exist_ok=True)
    marker=out/f'round{round_id}_verification.json'
    if marker.exists():
        p=json.loads(marker.read_text())
        if p['status']=='passed':
            assert sha(p['raw_path'])==p['raw_sha256'];return json.loads((out/f'round{round_id}_metrics.json').read_text())
    train=manifest(ds,cat,'train');test=manifest(ds,cat,'test');assert all(r['label']==0 for r in train)
    seed=668+round_id;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    selected=random.Random(seed).sample(train,shot)
    assert set(r['image_path'] for r in selected).isdisjoint(r['image_path'] for r in test)
    write(out/f'round{round_id}_support_images.csv',selected);write(out/'test_images.csv',test)
    support=torch.stack([Images(selected)[i] for i in range(shot)])
    temptrain=raw/f'temporary_support_round{round_id}.npy';tempquery=raw/f'temporary_query_round{round_id}.npy'
    t=time.perf_counter();tr=support_features(stn,support,temptrain,funcs);q=query_features(stn,test,tempquery)
    d,proof=gaussian_score(tr,q);maps=postprocess(d);gt=masks(test);labels=np.array([r['label'] for r in test],dtype=np.uint8)
    scores=maps.reshape(len(test),-1).max(1)
    pred=raw/f'round{round_id}_predictions.npz';np.savez_compressed(pred,labels=labels,masks=gt,image_scores=scores,score_maps=maps,mahalanobis56=d)
    row=dict(dataset=ds,shot=shot,category=cat,round=round_id,support_seed=seed,n_images=len(test),
             image_auroc=float(roc_auc_score(labels,scores)),pixel_auroc=float(roc_auc_score(gt.ravel(),maps.ravel())),seconds=time.perf_counter()-t)
    saved=np.load(pred);assert np.array_equal(saved['masks'],masks(test)) and np.array_equal(saved['labels'],labels)
    assert np.array_equal(saved['score_maps'],postprocess(saved['mahalanobis56']))
    assert np.array_equal(saved['image_scores'],saved['score_maps'].reshape(len(test),-1).max(1))
    assert abs(roc_auc_score(saved['labels'],saved['image_scores'])-row['image_auroc'])<1e-12
    assert abs(roc_auc_score(saved['masks'].ravel(),saved['score_maps'].ravel())-row['pixel_auroc'])<1e-12
    for r in selected+test:
        assert sha(r['image_path'])==r['sha256']
        if r['mask_path']:assert sha(r['mask_path'])==r['mask_sha256']
    (out/f'round{round_id}_metrics.json').write_text(json.dumps(row,indent=2))
    write(out/f'round{round_id}_image_scores.csv',[dict(image_path=r['image_path'],label=r['label'],score=float(s)) for r,s in zip(test,scores)])
    marker.write_text(json.dumps(dict(status='passed',raw_path=str(pred),raw_sha256=sha(pred),checkpoint_path=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        full_postprocessing_and_metrics_recomputed=True,input_and_support_SHA_checked=True,mahalanobis_samples=proof,
        embedding_concat_exact=True,support_augmentation_count=len(tr),support_forward_chunk=16,query_forward_batch=1,
        limitation='Sample covariance/distance checks; chunked eval support forward not bitwise equal to one huge forward'),indent=2))
    del tr,q,saved;temptrain.unlink();tempquery.unlink();torch.cuda.empty_cache()
    print('ROUND VERIFIED',row,flush=True);return row
def main():
    OUT.mkdir(parents=True,exist_ok=True);RAW.mkdir(parents=True,exist_ok=True)
    if (OUT/'verification.json').exists() and json.loads((OUT/'verification.json').read_text())['status']=='passed':return
    assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==COMMIT
    assert not subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip()
    # Separate compatibility runtime, preserving previous completed run artifacts.
    for p in REPO.rglob('*.py'):
        dest=RUNTIME/p.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(p.read_text().replace('Image.ANTIALIAS','Image.Resampling.LANCZOS'))
    sys.path.insert(0,str(RUNTIME));from models.stn import stn_net
    from utils import funcs
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=True
    env=dict(status='running',training_executed=False,checkpoint_source='Official MVTec4 trained public checkpoints',commit=COMMIT,
             conditions=['MVTec32 normal support extension','BTAD4 MVTec-to-BTAD transfer'],rounds=10,gpu=torch.cuda.get_device_name(),
             precision='FP32 convTF32 enabled, matmulTF32 disabled',support_chunk=16,query_batch=1,wrapper_sha256=sha(__file__),
             input='Resize224x224 LANCZOS ToTensor only',mask='Resize224x224 NEAREST >128; original metadata anomaly label',
             btad_checkpoint_selection='MVTec bottle4 checkpoint fixed a priori, no BTAD model selection/training',
             model_training_history='Official checkpoint training; no additional local gradient training',
             protocol_limits='MuSc did not disclose large-shot checkpoint or BTAD source selection; reference32 extends trained4 model; local random support sets10round',error=None)
    (OUT/'environment.json').write_text(json.dumps(env,indent=2));rows=[]
    cases=[('mvtec',cat,32) for cat in sorted(json.loads((DATA['mvtec']/'meta.json').read_text())['test'])]+[('btad',cat,4) for cat in ['01','02','03']]
    for ds,cat,shot in cases:
        checkpoint=ASSETS/'save_checkpoints/4'/(cat if ds=='mvtec' else 'bottle')/f"{cat if ds=='mvtec' else 'bottle'}_4_rotation_scale_model.pt"
        args=SimpleNamespace(input_channel=3,stn_mode='rotation_scale',shot=shot,img_size=224,obj=cat)
        stn=stn_net(args,pretrained=False).cuda();saved=torch.load(checkpoint,map_location='cpu',weights_only=True);stn.load_state_dict(saved['STN'],strict=True);stn.eval();del saved
        for rnd in range(10):
            env.update(dataset=ds,category=cat,shot=shot,round=rnd);(OUT/'environment.json').write_text(json.dumps(env,indent=2))
            with torch.no_grad():rows.append(evaluate(stn,funcs,ds,cat,shot,rnd,checkpoint))
            write(OUT/'round_metrics.csv',rows)
        del stn;torch.cuda.empty_cache()
    means=[]
    for ds,shot,ncats,nimages in [('mvtec',32,15,1725),('btad',4,3,741)]:
        subset=[r for r in rows if r['dataset']==ds];assert len(subset)==ncats*10
        macro=[{k:float(np.mean([r[k] for r in subset if r['round']==rnd])) for k in ['image_auroc','pixel_auroc']} for rnd in range(10)]
        means.append(dict(dataset=ds,shot=shot,n_rounds=10,n_categories=ncats,n_images=nimages,
                          **{k:float(np.mean([r[k] for r in macro])) for k in macro[0]},
                          **{k+'_std':float(np.std([r[k] for r in macro],ddof=0)) for k in macro[0]}))
    write(OUT/'mean_metrics.csv',means);assert sha(__file__)==env['wrapper_sha256']
    (OUT/'verification.json').write_text(json.dumps(dict(status='passed',n_conditions=2,n_category_rounds=180,
        training_executed=False,public_checkpoint_evaluation=True,all_round_inputs_raw_and_metrics_verified=True),indent=2))
    env['status']='completed';(OUT/'environment.json').write_text(json.dumps(env,indent=2))
    md=ROOT.parent/'markdown/RegAD_mvtec32_btad4_execution.md';text=md.read_text(encoding='utf-8').split('\n## 자체 측정 결과')[0]
    text+='\n## 자체 측정 결과\n\n단위 %,10 support round macro평균±표준편차(ddof0).\n\n| Dataset | Shot | Image AUROC | Pixel AUROC |\n|---|---|---:|---:|\n'
    for r in means:text+=f"| {r['dataset']} | {r['shot']} | {100*r['image_auroc']:.4f} ± {100*r['image_auroc_std']:.4f} | {100*r['pixel_auroc']:.4f} ± {100*r['pixel_auroc_std']:.4f} |\n"
    md.write_text(text,encoding='utf-8')
    subprocess.run([sys.executable,str(LAB/'method10/source/build_musc_reproduction_tables.py')],check=True)
    subprocess.run([sys.executable,str(LAB/'method10/source/audit_musc_remaining_work.py')],check=True)
    subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',r'C:\Users\test\Desktop\Codex\tools\update_obsidian_links.ps1'],check=True)
    print('REGAD REMAINING VERIFIED COMPLETE',means,flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        p=OUT/'environment.json'
        if p.exists():
            env=json.loads(p.read_text());env.update(status='failed',error=traceback.format_exc());p.write_text(json.dumps(env,indent=2))
        raise
