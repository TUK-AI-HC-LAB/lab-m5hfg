"""Bounded-disk, resumable release upload. Never modifies experiment files."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REPO='TUK-AI-HC-LAB/lab-m5hfg'
TAG='method10-22-data-20261009'
CHUNK=1024**3
PACK='/mnt/c/Users/test/Desktop/Codex/lab-m5hfg/github_exports/method10_22_20261009/pack_archive.py'
INVENTORY='/mnt/c/Users/test/Desktop/Codex/lab-m5hfg/github_exports/method10_22_20261009/inventory.json'
PYTHON='/home/test/miniforge3/envs/patchcore-gpu/bin/python'
TEMP=ROOT/'upload_chunks'
STATE=ROOT/'upload_status.json'

def gh(args,check=True):
    return subprocess.run(['gh',*args],capture_output=True,text=True,encoding='utf-8',check=check,creationflags=subprocess.CREATE_NO_WINDOW)
def checkpoint(state):
    temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2),encoding='utf-8');temp.replace(STATE)
def assets():
    return json.loads(gh(['api',f'repos/{REPO}/releases/tags/{TAG}']).stdout)['assets']
def upload(path,name,digest,state):
    size=path.stat().st_size
    for attempt in range(4):
        old=next((a for a in assets() if a['name']==name),None)
        if old:
            if old['size']==size and old.get('digest')=='sha256:'+digest:
                state['assets'][name]=dict(size=size,sha256=digest,id=old['id'],url=old['browser_download_url'],verified_server_digest=True)
                checkpoint(state);return
            raise RuntimeError(f'Existing asset does not match local SHA256: {name}')
        result=gh(['release','upload',TAG,str(path),'--repo',REPO],check=False)
        if result.returncode:
            if attempt==3:raise RuntimeError(result.stderr)
            time.sleep(5*(attempt+1));continue
        uploaded=next(a for a in assets() if a['name']==name)
        if uploaded['state']!='uploaded' or uploaded['size']!=size or uploaded.get('digest')!='sha256:'+digest:
            raise RuntimeError(f'Server digest/size verification failed: {name}')
        state['assets'][name]=dict(size=size,sha256=digest,id=uploaded['id'],url=uploaded['browser_download_url'],verified_server_digest=True)
        checkpoint(state);return

def main():
    inv=json.loads((ROOT/'inventory.json').read_text());TEMP.mkdir(exist_ok=True)
    state=json.loads(STATE.read_text()) if STATE.exists() else dict(status='uploading',release=f'https://github.com/{REPO}/releases/tag/{TAG}',assets={},completed_groups=[],total_source_bytes=inv['total_bytes'])
    state.update(status='uploading',started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));checkpoint(state)
    try:
        for group in inv['groups']:
            name=group['name']
            if name in state['completed_groups']:continue
            state['current_group']=name;checkpoint(state)
            error=(ROOT/'packing_errors.log').open('ab')
            proc=subprocess.Popen(['wsl','-d','Ubuntu','--',PYTHON,PACK,INVENTORY,name],stdout=subprocess.PIPE,stderr=error,creationflags=subprocess.CREATE_NO_WINDOW)
            parts=[];number=0
            try:
                while True:
                    first=proc.stdout.read(1024**2)
                    if not first:break
                    number+=1;asset=f'{name}.tar.gz.part{number:04d}';path=TEMP/asset
                    digest=hashlib.sha256();count=0
                    with path.open('wb') as f:
                        data=first
                        while data:
                            f.write(data);digest.update(data);count+=len(data)
                            if count>=CHUNK:break
                            data=proc.stdout.read(min(1024**2,CHUNK-count))
                    upload(path,asset,digest.hexdigest(),state);parts.append(asset)
                    assert path.resolve().is_relative_to(TEMP.resolve())
                    path.unlink() # Only this task's verified uploaded temporary chunk.
                if proc.wait()!=0:raise RuntimeError(f'Packing failed for {name}; see packing_errors.log')
            finally:
                if proc.poll() is None:proc.terminate();proc.wait()
                error.close()
            state['completed_groups'].append(name);state.setdefault('group_parts',{})[name]=parts;checkpoint(state)
            print('GROUP UPLOADED',name,len(parts),flush=True)
        state.update(status='complete',completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));checkpoint(state)
        manifest=ROOT/'release_manifest.json';manifest.write_text(json.dumps(state,indent=2),encoding='utf-8')
        upload(manifest,manifest.name,hashlib.sha256(manifest.read_bytes()).hexdigest(),state)
        for name in ['inventory.json','README.md','pack_archive.py','upload_archive.py']:
            p=ROOT/name;upload(p,name,hashlib.sha256(p.read_bytes()).hexdigest(),state)
        notes=ROOT/'release_notes_complete.md'
        notes.write_text('Method10–22 local reproduction archive. All inventoried present files uploaded and each asset verified against GitHub SHA-256 digest. Includes original MVTec/VisA/BTAD datasets, source/dependencies, interrupted runs, features/density/raw predictions/checkpoints/logs. CutPaste is a live unfinished snapshot; missing/deleted paths are listed in inventory.json. Join each group parts in numbered order and decompress/extract its tar.gz. See README.md and release_manifest.json.\n',encoding='utf-8')
        gh(['release','edit',TAG,'--repo',REPO,'--notes-file',str(notes)])
        checkpoint(state);print('ARCHIVE COMPLETE',state['release'],flush=True)
    except BaseException as error:
        state.update(status='failed',error=str(error));checkpoint(state);raise

if __name__=='__main__':main()
