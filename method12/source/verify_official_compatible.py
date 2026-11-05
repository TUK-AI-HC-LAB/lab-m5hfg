"""Small integration verification of minimally fixed official source, no monkey patches."""
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = Path('/mnt/c/Users/test/Desktop/Codex/Project/Simple3D-official-compatible-20261104')
OUT = HERE / 'results/official_compatible_20261104'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'verification.json').exists():
        raise RuntimeError('Existing verification preserved')
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    import torch
    import open3d
    from patchcore_runner import PatchCore
    import main as entry
    from pointnet2_ops import pointnet2_utils
    from knn_cuda import KNN

    result = dict(status='running',tests=[],torch=torch.__version__,open3d=open3d.__version__,
        gpu=torch.cuda.get_device_name(),official_source=str(REPO),
        native_fps_module=pointnet2_utils.__file__,native_knn_module=sys.modules[KNN.__module__].__file__,
        no_runtime_replacement=True)

    def save():
        (OUT / 'verification.json').write_text(json.dumps(result,indent=2))

    save()
    try:
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        for dataset, category in [('shapenet','ashtray0'),('minishift','capsule')]:
            args=SimpleNamespace(dataset=dataset,device='cuda:0',num_group=4096,group_size=128,
                max_nn=40,use_LFSA=True,use_MSND=True,num_MSND=2,feature='FPFH',level='ALL',vis_save=False)
            model=PatchCore(args=args)
            train=model.get_dataloader(dataset,'train',category,'ALL')
            test=model.get_dataloader(dataset,'test',category,'ALL')
            ds=test.dataset
            assert len(train)>0 and len(test)>0
            if dataset=='minishift':
                for path, label in zip(ds.gt_paths, ds.labels):
                    if label: assert Path(path).is_file(),path
            else:
                for path in ds.pcd_paths: assert Path(path).is_file(),path
            start=time.perf_counter()
            pc,_,_,path=next(iter(train))
            model.method.collect_features(pc)
            model.method.run_coreset()
            torch.cuda.synchronize()
            fit_s=time.perf_counter()-start
            selected=[ds.labels.index(0),ds.labels.index(1)]
            samples=[]
            for index in selected:
                # Exercise original DataLoader shape/collation and predict path.
                loader=torch.utils.data.DataLoader(torch.utils.data.Subset(ds,[index]),batch_size=1,
                    shuffle=False,num_workers=1,pin_memory=True)
                pc,mask,label,path=next(iter(loader))
                model.method.init_para()
                torch.cuda.synchronize();start=time.perf_counter()
                model.method.predict(pc,mask,label,path)
                torch.cuda.synchronize()
                assert len(model.method.image_preds)==1
                assert torch.isfinite(torch.as_tensor(model.method.image_preds[0])).all()
                assert len(model.method.pixel_preds)==pc.shape[1]
                samples.append(dict(label=int(label.item()),path=path[0],points=int(pc.shape[1]),
                    predict_wall_s=time.perf_counter()-start,
                    object_score=float(model.method.image_preds[0]),output_points=len(model.method.pixel_preds)))
                print('PREDICT PASSED',dataset,category,samples[-1],flush=True)
            result['tests'].append(dict(dataset=dataset,category=category,
                full_train_count=len(train.dataset),full_test_count=len(ds),
                tested_train=1,tested_test=2,fit_wall_s=fit_s,samples=samples))
            save()
        entry.write_experiment_log('compatibility_verification','Official entry imports and log directory passed.\n')
        result.update(status='passed',scope='integration only, 1 normal training + 1 normal/1 anomalous test per dataset; not full-category metrics',
            official_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    except Exception:
        result.update(status='failed',error=traceback.format_exc())
        raise
    finally:
        result['source_sha256']={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in REPO.rglob('*.py')}
        save()


if __name__=='__main__':
    main()
