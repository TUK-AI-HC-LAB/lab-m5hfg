"""Read-only, low-priority inventories/streamed archives for research export."""
import hashlib,json,os,sys,tarfile,time
from pathlib import Path

BASE=Path('/mnt/c/Users/test/Desktop/Codex/lab-m5hfg')
RAW=['musc','aprilgan','draem','nsa','igd','acr','regad','graphcore','vtadl','psvdd','spade','pyramidflow','cutpaste']
REPOS=['MuSc','VAND-APRIL-GAN','DRAEM','NSA','IGD','ACR','RegAD','open-iad','VT-ADL','PatchSVDD','SPADE-pytorch','PyramidFlow','PyramidFlow-dependency','CutPaste']
SKIP={'.git','__pycache__','.venv','venv','node_modules'}

def files(root):
    if root.is_symlink() or root.is_file():yield root;return
    for parent,dirs,names in os.walk(root,followlinks=False):
        dirs[:]=sorted(d for d in dirs if d not in SKIP)
        for d in list(dirs):
            p=Path(parent)/d
            if p.is_symlink():yield p;dirs.remove(d)
        for name in sorted(names):
            if name=='.env' or name.startswith('.env.') or name in {'token','stored_tokens'} or name.endswith(('.pyc','.pyo')):continue
            yield Path(parent)/name

def inventory():
    groups=[];missing=[]
    for n,label in enumerate(RAW,10):
        roots=[BASE/f'method{n}',Path('/home/test')/(label+'_results')]
        paths=[]
        for root in roots:
            if root.exists():paths.extend(files(root))
            else:missing.append(str(root))
        groups.append(dict(name=f'method{n:02d}',files=[str(p) for p in paths],bytes=sum(p.lstat().st_size for p in paths)))
    roots=[Path('/home/test')/r for r in REPOS]+[Path('/home/test')/r for r in ['graphcore_assets','regad_assets','draem_weights','Project']]
    roots += [Path('/home/test/.cache/torch/hub/checkpoints'),Path('/home/test/.cache/huggingface/hub')]
    for root in roots:
        if root.exists():
            fs=list(files(root));groups.append(dict(name='dependency-'+root.name,files=[str(p) for p in fs],bytes=sum(p.lstat().st_size for p in fs)))
        else:missing.append(str(root))
    datasets=['mvtec','VisA_20220922','VisA_pytorch','VisA_20220922.tar','btad_original','btad.zip','btad_download.log','dtd']
    for name in datasets:
        root=Path('/home/test/data')/name
        if root.exists():
            fs=list(files(root));groups.append(dict(name='dataset-'+name.replace('.','-'),files=[str(p) for p in fs],bytes=sum(p.lstat().st_size for p in fs)))
        else:missing.append(str(root))
    return dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),groups=groups,missing_paths=missing,
        total_bytes=sum(g['bytes'] for g in groups),scope='Method10-22 present code/results including interrupted runs, all their raw results, present dependencies and weights, original/converted MVTec/VisA/BTAD datasets. Runtime environments/system files/credentials excluded. Missing or already-deleted files cannot be archived. CutPaste is a live unfinished snapshot.')

class HashedReader:
    def __init__(self,f):self.f=f;self.digest=hashlib.sha256()
    def read(self,n):
        data=self.f.read(n);self.digest.update(data);return data

def pack(inv,name):
    group=next(g for g in inv['groups'] if g['name']==name);manifest=[]
    with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz',compresslevel=1) as tar:
        for value in group['files']:
            p=Path(value)
            archive_path=('workspace/'+p.relative_to(BASE).as_posix()) if p.is_relative_to(BASE) else p.as_posix().lstrip('/')
            if not p.exists() and not p.is_symlink():raise FileNotFoundError(f'Inventory file disappeared: {p}')
            if p.is_symlink():
                info=tar.gettarinfo(str(p),arcname=archive_path);tar.addfile(info)
                manifest.append(dict(path=value,archive_path=archive_path,symlink=os.readlink(p)));continue
            with p.open('rb') as f:
                before=os.fstat(f.fileno());info=tar.gettarinfo(str(p),arcname=archive_path,fileobj=f)
                info.size=before.st_size;reader=HashedReader(f);tar.addfile(info,reader);after=os.fstat(f.fileno())
            manifest.append(dict(path=value,archive_path=archive_path,bytes=info.size,sha256=reader.digest.hexdigest(),
                mtime_ns=before.st_mtime_ns,changed_during_capture=before.st_mtime_ns!=after.st_mtime_ns or before.st_size!=after.st_size,
                capture='open-file bytes up to initial size; logs may be captured prefixes; independently captured live files are not a transaction'))
        import io
        content=json.dumps(dict(group=name,entries=manifest),indent=2).encode()
        info=tarfile.TarInfo('ARCHIVE_FILE_SHA256.json');info.size=len(content);tar.addfile(info,io.BytesIO(content))

if __name__=='__main__':
    os.nice(10)
    if sys.argv[1]=='inventory':print(json.dumps(inventory(),indent=2))
    else:pack(json.loads(Path(sys.argv[1]).read_text()),sys.argv[2])
