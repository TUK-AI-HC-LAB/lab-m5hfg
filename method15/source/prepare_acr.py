"""Create documented compatibility runtime without editing the official checkout."""
import difflib
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=Path('/home/test/ACR/mvtec-ad')
RAW=Path('/home/test/acr_results/mvtec_seed42_20261007')
OUT=ROOT/'results/mvtec_seed42_20261007'
RUNTIME=RAW/'runtime'
COMMIT='54f1a6026cda4a501d870e49b7d49004b38a16fe'
KEYS=['image_auroc','image_f1_max','image_ap','pixel_auroc','pixel_f1_max','pixel_ap','aupro']
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def prepare():
    OUT.mkdir(parents=True,exist_ok=True);RAW.mkdir(parents=True,exist_ok=True)
    diffs=[];sources=[]
    for path in sorted(REPO.rglob('*')):
        if path.suffix not in ['.py','.yml']:continue
        rel=path.relative_to(REPO);before=path.read_text();after=before
        if rel.as_posix()=='data_loader/mvtec.py':
            after=after.replace('Image.ANTIALIAS','Image.Resampling.LANCZOS')
            # Locally generated trusted feature/mask files; mmap reduces copies.
            after=after.replace("torch.load(os.path.join(root,", "torch.load(os.path.join(root,")
            for expr in ["'train_%s.pt' % class_name", "'test_%s.pt' % self.test_class_name", "'test_%s.pt' % class_name", "'test_%s_gt.pt' % self.test_class_name", "'test_%s_gt_mask.pt' % self.test_class_name"]:
                after=after.replace(f'torch.load(os.path.join(root, {expr}))',f'torch.load(os.path.join(root, {expr}),weights_only=False,mmap=True)')
            # These cross-category concatenations are unused by Gaussian tasks.
            after=after.replace('self.x_test_all_abnormal_data = torch.cat(self.x_test_all_abnormal_data, 0)','self.x_test_all_abnormal_data = []')
            after=after.replace('self.x_test_all_abnormal_data.append(torch.load(os.path.join(root, \'test_%s.pt\' % class_name),weights_only=False,mmap=True))','pass  # unused cross-category test cache')
            after=after.replace('self.train_all_abnormal_data = self.cache_abnormal_data(self.datasets["train"])','self.train_all_abnormal_data = None  # unused by Gaussian task sampler')
        if rel.as_posix()=='data_loader/extract_embedding.py':
            # Intermediate layer1/2 tensors do not contribute to layer3 embeddings.
            after=after.replace("OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])","OrderedDict([('layer3', [])])")
            after=after.replace('zip(train_outputs.keys(), outputs)','zip(train_outputs.keys(), outputs[-1:])').replace('zip(test_outputs.keys(), outputs)','zip(test_outputs.keys(), outputs[-1:])')
        if rel.as_posix()=='trainers/zeroshot_trainer.py':
            after=after.replace('class ZeroShotMetaTrainer:','logger = None  # fix undefined optional logger in official detection\n\nclass ZeroShotMetaTrainer:')
            after=after.replace('        gt_list = loader.test_gt_list','        self.last_scores = scores\n        self.last_masks = loader.test_gt_mask_list\n        self.last_labels = loader.test_gt_list\n        gt_list = loader.test_gt_list')
            after=after.replace('        return img_roc_auc, per_pixel_rocauc','        self.last_native_auroc = (float(img_roc_auc), float(per_pixel_rocauc))\n        return img_roc_auc, per_pixel_rocauc')
        target=RUNTIME/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(after)
        sources.append(dict(path=str(rel),official_sha256=sha(path),runtime_sha256=sha(target)))
        if before!=after:diffs.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=str(rel),tofile='runtime/'+str(rel)))
    (OUT/'runtime_sources.json').write_text(json.dumps(sources,indent=2))
    (OUT/'compatibility.patch').write_text(''.join(diffs))
    folder=RUNTIME/'data';folder.mkdir(exist_ok=True)
    feature=folder/'mvtec_feature_layer3'
    if not feature.exists():feature.symlink_to(RAW/'features',target_is_directory=True)
if __name__=='__main__':prepare()
