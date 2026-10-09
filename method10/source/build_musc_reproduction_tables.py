"""Build all 18 MuSc table structures with measured local values only."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parents[1]
OUT = ROOT / 'results/musc_reproduction_tables'
MD = ROOT.parent / 'markdown/MuSc_all_reproduction_tables.md'
PAPER = next((ROOT.parent / 'paper').glob('ICLR24_MuSc*.pdf'))
KEYS = ['image_auroc', 'image_f1_max', 'image_ap', 'pixel_auroc', 'pixel_f1_max', 'pixel_ap', 'aupro']
HEAD = ['Image AUROC', 'Image F1-max', 'Image AP', 'Pixel AUROC', 'Pixel F1-max', 'Pixel AP', 'AUPRO']
MUSC = ROOT / 'results/mvtec_all_paper_20261002/category_metrics.csv'
APRIL = LAB / 'method11/source/results/mvtec_zero_shot_20261006/category_metrics.csv'

def read(p):
    with p.open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def load(p):
    rows = read(p)
    assert len(rows) == 15 and sum(int(r['n_images']) for r in rows) == 1725
    cats = {r['category']: r for r in rows}
    means = {k: sum(float(r[k]) for r in rows) / 15 for k in KEYS}
    return cats, means

mc, mm = load(MUSC)
ac, am = load(APRIL)
live_root=LAB/'method11/source/results/aprilgan_remaining_20261006'
live_rows=read(live_root/'aggregates.csv') if (live_root/'aggregates.csv').exists() else []
live={(r['dataset'],r['mode']):r for r in live_rows}
rscin_rows=read(live_root/'rscin_aggregates.csv') if (live_root/'rscin_aggregates.csv').exists() else []
rscin_live={(r['dataset'],r['mode'],r['rscin']):r for r in rscin_rows}
draem_root=LAB/'method12/source/results/mvtec_public_20261007'
draem_mean=draem_root/'rscin/mean_metrics.csv'
draem_rscin={r['rscin']:r for r in read(draem_mean)} if draem_mean.exists() else {}
nsa_root=LAB/'method13/source/results/logistic_seed923874273_20261007'
nsa_mean=nsa_root/'mean_metrics.csv'
nsa_verified=nsa_root/'verification.json'
nsa_result=read(nsa_mean)[0] if nsa_mean.exists() and nsa_verified.exists() and json.loads(nsa_verified.read_text())['status']=='passed' else None
igd_root=LAB/'method14/source/results/mvtec_full_seed42_20261007'
igd_accelerated=LAB/'method14/source/results/mvtec_accelerated_seed42_20261007'
if (igd_accelerated/'verification.json').exists() and json.loads((igd_accelerated/'verification.json').read_text())['status']=='passed':
    igd_root=igd_accelerated
igd_largebatch=LAB/'method14/source/results/mvtec_largebatch_seed42_20261007'
if (igd_largebatch/'verification.json').exists() and json.loads((igd_largebatch/'verification.json').read_text())['status']=='passed':
    igd_root=igd_largebatch
igd_batch8=LAB/'method14/source/results/mvtec_batch8_seed42_20261007'
if (igd_batch8/'verification.json').exists() and json.loads((igd_batch8/'verification.json').read_text())['status']=='passed':
    igd_root=igd_batch8
igd_mean=igd_root/'mean_metrics.csv'
igd_verified=igd_root/'verification.json'
igd_result=read(igd_mean)[0] if igd_mean.exists() and igd_verified.exists() and json.loads(igd_verified.read_text())['status']=='passed' else None
visa_path=LAB/'method11/source/results/visa_0shot_seed42_20261006'
vc={}
if (visa_path/'environment.json').exists() and json.loads((visa_path/'environment.json').read_text())['status']=='completed':
    vc={r['category']:r for r in read(visa_path/'category_metrics.csv')}
manifest = []
sections = []
index = []
pending = '미실행'
local = '측정값 있음·조건 대조 필요'
acr_root=LAB/'method15/source/results/mvtec_seed42_20261007'
acr_verified=acr_root/'verification.json'
acr_result=read(acr_root/'mean_metrics.csv')[0] if acr_verified.exists() and json.loads(acr_verified.read_text())['status']=='passed' else None
regad_root=LAB/'method16/source/results/mvtec_public_20261007'
regad_verified=regad_root/'verification.json'
regad_means={int(r['shot']):r for r in read(regad_root/'mean_metrics.csv')} if regad_verified.exists() and json.loads(regad_verified.read_text())['status']=='passed' else {}
regad_rscin={(int(r['shot']),r['rscin']):r for r in read(regad_root/'rscin_mean_metrics.csv')} if regad_means else {}
graphcore_root=LAB/'method17/source/results/mvtec_pvig_fp32_20261008'
graphcore_verified=graphcore_root/'verification.json'
graphcore_means={int(r['shot']):r for r in read(graphcore_root/'mean_metrics.csv')} if graphcore_verified.exists() and json.loads(graphcore_verified.read_text())['status']=='passed' else {}
graphcore_started=(graphcore_root/'environment.json').exists()
vtadl_root=LAB/'method18/source/results/btad_seed123_20261008'
vtadl_verified=vtadl_root/'verification.json'
vtadl_result=read(vtadl_root/'mean_metrics.csv')[0] if vtadl_verified.exists() and json.loads(vtadl_verified.read_text())['status']=='passed' else None
vtadl_started=(vtadl_root/'environment.json').exists()
psvdd_root=LAB/'method19/source/results/btad_seed42_20261008'
psvdd_verified=psvdd_root/'verification.json'
psvdd_result=read(psvdd_root/'mean_metrics.csv')[0] if psvdd_verified.exists() and json.loads(psvdd_verified.read_text())['status']=='passed' else None
psvdd_started=(psvdd_root/'environment.json').exists()
spade_root=LAB/'method20/source/results/btad_k50_seed42_20261008'
spade_verified=spade_root/'verification.json'
spade_result=read(spade_root/'mean_metrics.csv')[0] if spade_verified.exists() and json.loads(spade_verified.read_text())['status']=='passed' else None
spade_started=(spade_root/'environment.json').exists()
pyramidflow_root=LAB/'method21/source/results/btad_res18_seed0_20261008'
pyramidflow_verified=pyramidflow_root/'verification.json'
pyramidflow_result=read(pyramidflow_root/'mean_metrics.csv')[0] if pyramidflow_verified.exists() and json.loads(pyramidflow_verified.read_text())['status']=='passed' else None
pyramidflow_started=(pyramidflow_root/'environment.json').exists()
cutpaste_root=LAB/'method22/source/results/mvtec_scratch3way_seed42_20261008'
cutpaste_verified=cutpaste_root/'verification.json'
cutpaste_result=read(cutpaste_root/'mean_metrics.csv')[0] if cutpaste_verified.exists() and json.loads(cutpaste_verified.read_text())['status']=='passed' else None
cutpaste_started=(cutpaste_root/'environment.json').exists()
evidence = {
    'MuSc': '../source/results/mvtec_all_paper_20261002/category_metrics.csv',
    'APRIL-GAN': '../../method11/source/results/mvtec_zero_shot_20261006/category_metrics.csv',
}
if acr_result is not None:
    evidence['ACR']='../../method15/source/results/mvtec_seed42_20261007/category_metrics.csv'
if regad_means:
    evidence['RegAD']='../../method16/source/results/mvtec_public_20261007/category_metrics.csv'
if graphcore_means:
    evidence['GraphCore']='../../method17/source/results/mvtec_pvig_fp32_20261008/category_metrics.csv'

def values(row, keys=KEYS):
    return [f'{100*float(row[k]):.4f}' for k in keys]

def rendered_live(row,keys):
    return [f'{100*float(row[k]):.4f}'+(f' ± {100*float(row[k+"_std"]):.4f}' if row.get(k+'_std') else '') for k in keys]

def macro(method, dataset='MVTec AD', keys=KEYS, mode='zero_shot'):
    if method=='GraphCore' and dataset=='MVTec AD' and 4 in graphcore_means:
        return rendered_live(graphcore_means[4],keys)
    if method=='RegAD' and dataset=='MVTec AD' and 4 in regad_means:
        return rendered_live(regad_means[4],keys)
    if method=='ACR' and dataset=='MVTec AD' and acr_result is not None:
        return values(acr_result,keys)
    ds={'MVTec AD':'mvtec','VisA':'visa','BTAD':'btad'}[dataset]
    if method=='APRIL-GAN' and (ds,mode) in live:
        return rendered_live(live[ds,mode],keys)
    if dataset != 'MVTec AD' or method not in ('MuSc', 'APRIL-GAN'):
        return [pending] * len(keys)
    return values(mm if method == 'MuSc' else am, keys)

def ref(method):
    return f'[{method} CSV]({evidence[method]})' if method in evidence else '—'

def table(n, title, page, columns, rows, note=''):
    assert all(len(r) == len(columns) for r in rows), (n, columns)
    index.append([str(n), title, str(page), str(len(rows))])
    lines = [f'## Table {n}. {title}', '', f'원문 {page}쪽. 수치는 자체 측정값이며 단위는 별도 표기 외 %.', '']
    lines += ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---']*len(columns)) + ' |']
    lines += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    lines += ['', note, '']
    sections.append('\n'.join(lines))
    for i, row in enumerate(rows, 1):
        manifest.append(dict(table=n, row=i, title=title, paper_page=page,
                             cells=json.dumps(dict(zip(columns, row)), ensure_ascii=False)))

# Table 1: Preserve all four comparison blocks, including repeated MuSc rows.
rows = []
blocks = [('MVTec AD', '0-shot 비교', [('WinCLIP','0-shot'),('APRIL-GAN','0-shot'),('ACR','0-shot'),('MuSc','0-shot')]),
          ('VisA', '0-shot 비교', [('WinCLIP','0-shot'),('APRIL-GAN','0-shot'),('MuSc','0-shot')]),
          ('MVTec AD', '4-shot 비교', [('RegAD','4-shot'),('PatchCore','4-shot'),('WinCLIP','4-shot'),('APRIL-GAN','4-shot'),('GraphCore','4-shot'),('MuSc','0-shot')]),
          ('VisA', '4-shot 비교', [('PatchCore','4-shot'),('WinCLIP','4-shot'),('APRIL-GAN','4-shot'),('MuSc','0-shot')])]
for ds, group, methods in blocks:
    for m, shot in methods:
        available = ds == 'MVTec AD' and shot == '0-shot' and m in evidence
        if m=='RegAD' and ds=='MVTec AD' and shot=='4-shot' and 4 in regad_means:
            available=True
        if m=='GraphCore' and ds=='MVTec AD' and shot=='4-shot' and 4 in graphcore_means:
            available=True
        mode='few_shot' if shot=='4-shot' else 'zero_shot'
        case=live.get(('mvtec' if ds=='MVTec AD' else 'visa',mode)) if m=='APRIL-GAN' else None
        available=available or case is not None
        status=(f"{case['n_runs']}회·{case['status']}·조건 대조 필요" if case else local) if available else pending
        if m=='RegAD' and available:status='공개 가중치·support10round(capsule/grid 자체 생성)·재학습 없음'
        if m=='GraphCore' and available:status='저자 benchmark·공개 PyramidViG·support10round·feature층 선택/구조 차이 있음'
        link='[추가 실험 집계](../../method11/markdown/APRIL_GAN_remaining_experiments.md)' if case else ref(m) if available else '—'
        if m=='GraphCore' and graphcore_started and not available:
            status='평가 시작·최종 검증/집계 미완료'
            link='[GraphCore 조건·진행](../../method17/markdown/GraphCore_mvtec_execution.md)'
        rows.append([ds,group,m,shot] + (macro(m,ds,mode=mode) if available else [pending]*7) + [status,link])
table(1,'MVTec AD·VisA의 zero/few-shot 비교',7,
      ['Dataset','비교 블록','Method','Setting']+HEAD+['상태','근거'],rows,
      '원문의 ±는 반복 실험의 변동량이다. 단일 seed 결과에 ±를 붙이지 않는다. 원문에서 미보고한 metric도 자체 측정 가능하면 이후 채우되 그 사실을 기록한다.')

rows = []
for m, shot in [('CutPaste','full-shot'),('NSA','full-shot'),('IGD','full-shot'),('PatchCore','full-shot'),('RegAD','32-shot'),('GraphCore','8-shot'),('MuSc','0-shot')]:
    ok = m == 'MuSc'
    rows.append([m,shot]+(macro(m,keys=['image_auroc','pixel_auroc']) if ok else [pending]*2)+[local if ok else pending,ref(m) if ok else '—'])
    if m=='CutPaste' and cutpaste_result:
        rows[-1]=[m,shot]+values(cutpaste_result,['image_auroc','pixel_auroc'])+['자체 학습·15category·scratch3way·단일 seed·조건 차이 명시',
            '[CutPaste 실행](../../method22/markdown/CutPaste_mvtec_execution.md)']
    elif m=='CutPaste' and cutpaste_started:
        rows[-1]=[m,shot]+[pending]*2+['전체 학습 시작·최종 검증 미완료',
            '[CutPaste 실행](../../method22/markdown/CutPaste_mvtec_execution.md)']
    if m=='NSA' and nsa_result:
        rows[-1]=[m,shot]+values(nsa_result,['image_auroc','pixel_auroc'])+['자체 학습·15개 category·단일 seed',
            '[NSA 실행](../../method13/markdown/NSA_mvtec_execution.md)']
    if m=='GraphCore' and 8 in graphcore_means:
        rows[-1]=[m,shot]+rendered_live(graphcore_means[8],['image_auroc','pixel_auroc'])+['저자 benchmark·PyramidViG·10support round·구현 조건 차이 있음',
            '[GraphCore 실행·조건 대조](../../method17/markdown/GraphCore_mvtec_execution.md)']
    elif m=='GraphCore' and graphcore_started:
        rows[-1]=[m,shot]+[pending]*2+['평가 시작·최종 검증/집계 미완료','[GraphCore 조건·진행](../../method17/markdown/GraphCore_mvtec_execution.md)']
    if m=='IGD' and igd_result:
        rows[-1]=[m,shot]+values(igd_result,['image_auroc','pixel_auroc'])+['자체 학습·15개 category·평가 경로 복구·pooled pixel AUROC'+('·BF16/TF32' if igd_root in [igd_accelerated,igd_largebatch,igd_batch8] else '')+('·local batch4(공식2에서 변경)' if igd_root==igd_largebatch else '')+('·local batch8(공식2에서 변경)' if igd_root==igd_batch8 else ''),
            '[IGD 실행·집계 정의](../../method14/markdown/IGD_mvtec_execution.md)']
table(2,'MVTec AD의 many-shot 방법 비교',8,['Method','Setting','AC','AS','상태','근거'],rows)

def dual_rows(settings, baseline):
    return [[s]+(macro('MuSc',keys=['image_auroc','pixel_auroc']) if s == baseline else [pending]*2)+[pending]*2+
            ['MVTec 기본 측정값 있음·조건 대조 필요' if s == baseline else pending] for s in settings]

dual_head = ['조건','MVTec AC','MVTec AS','VisA AC','VisA AS','상태']
table(3,'LNAMD aggregation degree 제거 실험',8,dual_head,
      dual_rows(['{1}','{3}','{5}','{1,3}','{3,5}','{1,3,5}'],'{1,3,5}'),
      'AC=Image AUROC, AS=Pixel AUROC. 기본 행에만 기존 측정값을 연결했다. 다른 r 조합은 별도 실행한다.')
table(4,'MSM sample strategy 제거 실험',8,dual_head,
      dual_rows(['(a) min','(b) max','(c) mean','(d) 30% + min','(e) 30% + max','(f) 30% + mean'],'(f) 30% + mean'))
rows=[]
for ds in ['MVTec AD','VisA']:
    for flag in ['w/o','w']:
        ok=ds=='MVTec AD' and flag=='w'
        rows.append([ds,flag]+(macro('MuSc',keys=KEYS[:3]) if ok else [pending]*3)+[local if ok else pending])
table(5,'MuSc의 RsCIN 적용 전후',8,['Dataset','RsCIN','AUROC','F1-max','AP','상태'],rows,
      'w/o는 동일 raw image score에서 RsCIN만 제거한 평가가 필요하다. 현재 w 행은 기존 공식 MuSc 실행값이다.')
table(6,'MuSc의 이미지당 추론 시간·최대 GPU 메모리',9,
      ['s','Time (ms/image)','GPU cost (MB)','상태'],[[str(s),pending,pending,pending] for s in [1,2,3]],
      '논문은 RTX 3090에서 측정했다. 재현 표는 내 RTX 5080에서 측정한다. 모델 로딩·시각화·metric·raw 저장을 포함한 wall time을 이 칸에 넣지 않는다. 원문은 최대 200장인 pool의 메모리를 측정하므로 같은 pool 조건부터 맞춘다.')
rows=[]
for ds in ['MVTec AD','VisA']:
    for s in [1,2,3]:
        ok=ds=='MVTec AD' and s==1
        sizes=[int(r['n_images']) for r in mc.values()]
        rows.append([ds,str(s)]+(macro('MuSc',keys=['image_auroc','pixel_auroc']) if ok else [pending]*2)+
                    [f'{min(sizes)}–{max(sizes)}' if ok else pending,local if ok else pending])
table(7,'test pool 분할 수에 따른 성능·부분집합 크기',9,
      ['Dataset','s','AC','AS','Size of subsets','상태'],rows,
      '분할 후 모든 test 이미지의 예측을 합쳐 category metric을 계산한다. subset별 AUROC의 평균으로 대체하지 않는다.')

backbones=[('DINO','ViT-B-16','ImageNet-1k'),('DINO','ViT-B-8','ImageNet-1k'),
 ('DINOv2','ViT-B-14','LVD-142M'),('DINOv2','ViT-L-14','LVD-142M'),
 ('CLIP','ViT-B-32','WIT-400M'),('CLIP','ViT-B-16','WIT-400M'),
 ('CLIP','ViT-B-16-plus-240','LAION-400M'),('CLIP','ViT-L-14','WIT-400M'),
 ('CLIP','ViT-L-14-336','WIT-400M'),('—','Swin-B-4','ImageNet-22k'),('—','Swin-L-4','ImageNet-22k')]
rows=[]
for pre,arch,data in backbones:
    ok=arch=='ViT-L-14-336'
    rows.append([pre,arch,data]+(macro('MuSc',keys=['image_auroc','pixel_auroc']) if ok else [pending]*2)+[pending]*2+
                [local if ok else pending])
table(8,'사전학습 방식·backbone별 MuSc 성능',14,
      ['Pre-training','Arch.','Pre-training dataset','MVTec AC','MVTec AS','VisA AC','VisA AS','상태'],rows,
      '사전학습 데이터명은 실험 조건이며 성능 수치가 아니다. ViT-Base는 3층씩, ViT-Large는 6층씩 4 stage를 구성하고 Swin은 자체 stage를 따른다.')
table(9,'더 큰 aggregation degree 조합',15,dual_head,
      dual_rows(['{1}','{1,3}','{1,3,5}','{1,3,5,7}','{1,3,5,7,9}','{1,3,5,7,9,13}'],'{1,3,5}'))
table(10,'MSM percentage interval 선택',16,dual_head,
      dual_rows([f'{x}%–30%' for x in [0,2,4,6,8,10]],'0%–30%'),
      '오름차순 score의 하위 Y%를 제외하고 Y%–30% 구간 평균을 사용한다. Table 4의 sample strategy 실험과 구분한다.')

rows=[]
methods=[('SPADE','full-shot','MVTec AD'),('DRAEM','full-shot','MVTec AD'),('STPM','full-shot','MVTec AD'),
 ('APRIL-GAN','0-shot','MVTec AD'),('APRIL-GAN','4-shot','MVTec AD'),('APRIL-GAN','0-shot','VisA'),
 ('PatchCore','full-shot','MVTec AD'),('DSR','full-shot','MVTec AD'),('RegAD','2-shot','MVTec AD'),
 ('RegAD','4-shot','MVTec AD'),('RegAD','8-shot','MVTec AD'),('APRIL-GAN','4-shot','VisA')]
for m,shot,ds in methods:
    for flag in ['w/o','w']:
        ok=m=='APRIL-GAN' and shot=='0-shot' and ds=='MVTec AD' and flag=='w/o'
        mode='few_shot' if shot=='4-shot' else 'zero_shot'
        result=rscin_live.get(('mvtec' if ds=='MVTec AD' else 'visa',mode,flag)) if m=='APRIL-GAN' else None
        measured=rendered_live(result,KEYS[:3]) if result else macro(m,keys=KEYS[:3]) if ok else [pending]*3
        status=f"{result['n_runs']}회·자체 RsCIN 측정·조건 대조 필요" if result else local if ok else pending
        link='[RsCIN 집계](../../method11/markdown/APRIL_GAN_remaining_experiments.md)' if result else ref(m) if ok else '—'
        if m=='DRAEM' and ds=='MVTec AD' and flag in draem_rscin:
            measured=values(draem_rscin[flag],KEYS[:3])
            status='공개 가중치 자체 평가·15개 category·재학습 없음·조건 대조 필요'
            link='[DRAEM 실행](../../method12/markdown/DRAEM_mvtec_official_execution.md)'
        if m=='RegAD' and ds=='MVTec AD' and (int(shot.split('-')[0]),flag) in regad_rscin:
            measured=rendered_live(regad_rscin[(int(shot.split('-')[0]),flag)],KEYS[:3])
            status='공개 가중치·15개 category·support10round(capsule/grid 자체 생성)·자체 RsCIN'
            link='[RegAD 실행](../../method16/markdown/RegAD_mvtec_execution.md)'
        rows.append([ds,m,shot,flag]+measured+[status,link])
table(11,'비교군에 RsCIN을 적용한 classification 결과',17,
      ['Dataset','Method','Setting','RsCIN','AUROC','F1-max','AP','상태','근거'],rows,
      '원문의 *는 VisA를 뜻하며 여기서는 Dataset 열로 풀었다. 원문에 명시하지 않은 full-shot 표기는 해당 방법의 기본 조건으로 정리한 것이므로 세부 학습 설정을 추가 확인한다. RsCIN 입력은 ViT-L-14-336 class token으로 통일한다. 기존 PatchCore 실행을 이 실험의 w/o 행으로 자동 대입하지 않는다.')

rows=[]
for category,ds in [('screw','MVTec AD'),('hazelnut','MVTec AD'),('metal_nut','MVTec AD'),('capsules','VisA'),('macaroni2','VisA')]:
    vals=[pending]*2
    for cats in [ac if ds=='MVTec AD' else vc,mc if ds=='MVTec AD' else {}]:
        vals+=values(cats[category],['image_auroc','pixel_auroc']) if category in cats else [pending]*2
    rows.append([category,ds]+vals)
rows.append(['mean','선택된 5개 category']+[pending]*6)
rows.append(['mean-ALL','MVTec AD']+[pending]*2+macro('APRIL-GAN',keys=['image_auroc','pixel_auroc'])+macro('MuSc',keys=['image_auroc','pixel_auroc']))
table(12,'방향·크기가 일정하지 않은 category 비교',17,
      ['Category','Dataset','WinCLIP AC','WinCLIP AS','APRIL-GAN AC','APRIL-GAN AS','MuSc AC','MuSc AS'],rows,
      'metal nut은 파일 category metal_nut으로 썼다. capsules는 VisA이며 MVTec의 capsule과 다르다. mean은 5개 category가 모두 측정된 뒤 계산한다. mean-ALL은 원문의 값이 MVTec 전체 평균을 가리키므로 MVTec AD로 표시했다. 모든 채운 값은 조건 대조가 남아 있다.')

rows=[]
speed=([('RegAD','ResNet-18','4-shot','yes',None),('APRIL-GAN','ViT-L-14-336','0-shot','yes',None)]+
      [('MuSc','ViT-L-14-336','0-shot','no',s) for s in [1,2,3]]+
      [('APRIL-GAN','ViT-B-16-plus-240','0-shot','yes',None),('WinCLIP','ViT-B-16-plus-240','0-shot','no',None)]+
      [('MuSc','ViT-B-16-plus-240','0-shot','no',s) for s in [1,2,3]])
for m,arch,shot,train,s in speed:
    ok=arch=='ViT-L-14-336' and ((m=='MuSc' and s==1) or m=='APRIL-GAN')
    bench=LAB/'method11/source/results'/('benchmark_'+arch+'_20261006')/'benchmark.json'
    bm=json.loads(bench.read_text()) if m=='APRIL-GAN' and bench.exists() else None
    perf=macro(m,keys=['image_auroc','aupro']) if ok else [pending]*2
    if m=='RegAD' and 4 in regad_means:
        perf=rendered_live(regad_means[4],['image_auroc','aupro']);ok=True
    small_path=LAB/'method11/source/results/mvtec_0shot_vit_b16_plus_240_20261006'
    if m=='APRIL-GAN' and arch=='ViT-B-16-plus-240' and (small_path/'environment.json').exists():
        if json.loads((small_path/'environment.json').read_text())['status']=='completed':
            perf=values(read(small_path/'mean_metrics.csv')[0],['image_auroc','aupro'])
    rows.append([m,arch,shot,str(s) if s else '—',train,f"{bm['mean_ms']:.4f}" if bm else pending,
                 f"{bm['gpu_allocated_mib']:.4f}" if bm else pending]+
                perf+
                ['단독 속도 측정·조건 대조 필요'+('·자체 학습 checkpoint' if arch=='ViT-B-16-plus-240' else '') if bm else local+'·속도 미측정' if ok else pending])
table(13,'비교군의 추론 시간·GPU 메모리·성능',18,
      ['Method','Backbone','Setting','s','Training','Time (ms/image)','GPU allocated (MiB)','AC','AS=AUPRO','상태'],rows,
      '**이 표의 AS는 Pixel AUROC가 아니라 AUPRO다.** Training=yes는 방법에 학습된 모듈이 있다는 뜻이며 이번 실행에서 추가 학습했는지와 구분한다. 원문에서 미보고한 셀은 재현에서도 직접 측정하기 전에는 채우지 않는다.')

rows=[]
for m,shot in [('VT-ADL','full-shot'),('P-SVDD','full-shot'),('SPADE','full-shot'),('PaDiM','full-shot'),('PyramidFlow','full-shot'),('PatchCore','4-shot'),('RegAD','4-shot'),('APRIL-GAN','4-shot'),('APRIL-GAN','0-shot'),('MuSc','0-shot')]:
    mode='few_shot' if shot=='4-shot' else 'zero_shot'
    result=live.get(('btad',mode)) if m=='APRIL-GAN' else None
    rows.append([m,shot]+(rendered_live(result,['image_auroc','pixel_auroc']) if result else [pending]*2)+
                [f"{result['n_runs']}회·측정값 있음·BTAD 세부 조건 잠정" if result else pending])
    if m=='VT-ADL' and vtadl_result:
        rows[-1]=[m,shot]+values(vtadl_result,['image_auroc','pixel_auroc'])+['자체 학습·제품3개·400epoch·Gaussian150·공개코드 수정 명시']
    elif m=='VT-ADL' and vtadl_started:
        rows[-1]=[m,shot]+[pending]*2+['자체 학습 시작·최종 검증 미완료']
    if m=='P-SVDD' and psvdd_result:
        rows[-1]=[m,shot]+values(psvdd_result,['image_auroc','pixel_auroc'])+['자체 학습·제품3개·299학습epoch·공식 기본λ1·설정 차이 명시']
    elif m=='P-SVDD' and psvdd_started:
        rows[-1]=[m,shot]+[pending]*2+['자체 학습 시작·최종 검증 미완료']
    if m=='SPADE' and spade_result:
        rows[-1]=[m,shot]+values(spade_result,['image_auroc','pixel_auroc'])+['제품3개·K50·논문 수식 재현·비공식 구현·조건 차이 명시']
    elif m=='SPADE' and spade_started:
        rows[-1]=[m,shot]+[pending]*2+['특징 추출·검색 시작·최종 검증 미완료']
    if m=='PyramidFlow' and pyramidflow_result:
        rows[-1]=[m,shot]+values(pyramidflow_result,['image_auroc','pixel_auroc'])+['제품3개·Res18·15epoch·최종모델·설정 차이 명시']
    elif m=='PyramidFlow' and pyramidflow_started:
        rows[-1]=[m,shot]+[pending]*2+['자체 학습 시작·최종 검증 미완료']
table(14,'BTAD의 zero/few/full-shot 비교',18,['Method','Setting','AC','AS','상태'],rows,
      'MVTec AD 조건을 BTAD에 적용하는 원문 protocol을 따르고 BTAD test 데이터로 hyperparameter를 조정하지 않는다. VT-ADL: [실행·설정 차이](../../method18/markdown/VT_ADL_btad_execution.md). P-SVDD: [실행·설정 차이](../../method19/markdown/P_SVDD_btad_execution.md). SPADE: [실행·설정 차이](../../method20/markdown/SPADE_btad_execution.md). PyramidFlow: [실행·설정 차이](../../method21/markdown/PyramidFlow_btad_execution.md).')
for n,name in [(15,'MuSc'),(16,'MuSc+')]:
    rows=[]
    for shot in ['0-shot','1-shot','2-shot','4-shot','8-shot','16-shot','32-shot','full-shot']:
        ok=shot=='0-shot'
        rows.append([shot]+(macro('MuSc',keys=['image_auroc','pixel_auroc']) if ok else [pending]*2)+[local if ok else pending])
    table(n,f'MVTec AD의 {name} few/many-shot 확장',19,['Setting','AC','AS','상태'],rows,
          '0-shot은 MuSc 기본 결과다. MuSc는 정상 reference를 기존 test pool 상호 비교에 추가한다. MuSc+는 정상 reference만으로 patch score를 계산하므로 별도 구현·실험이 필요하다.')

rows=[[c]+values(r)+[local] for c,r in mc.items()]
rows.append(['Mean']+values(mm)+[local])
table(17,'MuSc의 MVTec AD 15개 category 상세 결과',20,['Class']+HEAD+['상태'],rows,
      '기존 공식 구현체 실행의 실제 값이다. 15개 category의 비가중 평균을 썼다. 공통 framework 결과와 섞지 않았다. 근거: '+ref('MuSc'))
visa=['candle','capsules','cashew','chewinggum','fryum','macaroni1','macaroni2','pcb1','pcb2','pcb3','pcb4','pipe_fryum','Mean']
table(18,'MuSc의 VisA 12개 category 상세 결과',20,['Class']+HEAD+['상태'],[[c]+[pending]*7+[pending] for c in visa])

assert len(index)==18 and [int(r[0]) for r in index]==list(range(1,19))
intro='''# MuSc 논문 전체 표 — 내 PC 재현 결과와 실행 목록

작성일: 2026-10-06. **본문·부록 Table 1–18의 모든 행과 metric을 Markdown으로 구성했다.** 논문의 성능·시간·메모리 수치를 결과 셀에 복사하지 않았다. 채운 숫자는 보존된 자체 실행 CSV에서만 가져왔다.

원문의 다단 헤더와 좌우 배치를 한 표로 풀고 Dataset·상태·근거 열을 추가했다. 원문의 숫자 순위에 따른 bold·underline와 개선폭을 가져오지 않았다. 자체 비교군이 모두 실행된 뒤 다시 계산한다.

- `미실행`: 해당 표 조건으로 채울 직접 측정값이 아직 이 집계에 없다. 다른 실험 결과가 없다는 뜻은 아니다.
- `측정값 있음·조건 대조 필요`: 자체 실행 근거가 있으나 논문 조건과 모든 항목이 같다고 확정하지 않았다. 완료 인증이 아니다.
- 기존 WinCLIP·PatchCore·PaDiM 결과는 보유해도 dataset·shot·backbone·reference·seed 조건의 일치가 검증되기 전에는 해당 행에 넣지 않는다.
- 단위: 성능은 %, 차이는 %p, 추론 시간은 ms/image, GPU 메모리는 MiB(bytes/1024²). 기본 AC는 Image AUROC, AS는 Pixel AUROC이며 Table 13만 AS=AUPRO다.
- 각 방법의 원래 정보 조건을 유지한다. 모든 비교군의 backbone·resize를 MuSc와 강제로 맞추지 않는다. 학습이 필요한 방법은 보조 데이터/target train 사용 조건을 기록한다.
- 공개 checkpoint 평가는 학습부터 재현한 결과와 구분한다. 4-shot 반복 횟수·reference 선택·표준편차 정의는 원문·공식 코드에서 확정한 뒤 실행한다. 단일 seed에 원문의 ±를 붙이지 않는다.

## 기존 측정값의 조건과 한계

MuSc는 MVTec AD 15개 category·1,725장·seed 42·CLIP ViT-L-14-336·518 입력·r={1,3,5}·s=1 실행이다. 기존 실행의 RsCIN 창 [1,2,3]과 원문 본문의 [2,3] 관계를 공식 구현과 함께 대조해야 한다. 각 표의 기본 행에 같은 기존 측정값을 재사용한 것은 신규 제거 실험을 수행했다는 뜻이 아니다.

APRIL-GAN은 같은 1,725장에 VisA 학습 공개 checkpoint를 사용한 MVTec AD 0-shot 평가다. MuSc와 원본 이미지·라벨·SHA-256은 일치하나 mask 처리와 방법별 scoring이 다르다. 단일 seed이며 보조 데이터 학습을 이번 PC에서 다시 수행하지 않았다.

표의 빈칸을 채우는 것과 논문 전체 재현 완료는 다르다. Figure 6–10 등의 qualitative 결과·곡선과 구현 검증은 별도 작업이다. 이번 요청에서는 Table 1–18을 만들었다.

## 근거와 재생성

- [MuSc 원문 PDF](../paper/ICLR24_MuSc_Zero_Shot_Industrial_Anomaly_Classification_and_Segmentation_with_Mutual_Scoring_of_the_Unlabeled_Images.pdf)
- [MuSc 공식 실행 CSV](../source/results/mvtec_all_paper_20261002/category_metrics.csv)
- [MuSc category 설정 예시](../source/results/mvtec_all_paper_20261002/bottle/config.json)
- [APRIL-GAN 공식 실행 보고서](../../method11/markdown/APRIL_GAN_mvtec_official_execution.md)
- [APRIL-GAN 나머지 실험 최신 상태·반복·RsCIN·benchmark](../../method11/markdown/APRIL_GAN_remaining_experiments.md)
- [DRAEM 공개 가중치 자체 평가·RsCIN 비교](../../method12/markdown/DRAEM_mvtec_official_execution.md)
- [NSA 자체 학습·평가 실행](../../method13/markdown/NSA_mvtec_execution.md)
- [IGD 자체 학습·평가 실행](../../method14/markdown/IGD_mvtec_execution.md)
- [집계 코드](../source/build_musc_reproduction_tables.py)
- [표의 모든 행 JSON/CSV](../source/results/musc_reproduction_tables/table_rows.csv)
- [집계 provenance](../source/results/musc_reproduction_tables/provenance.json)

## 전체 표 목록

'''
intro+='| Table | 내용 | 원문 쪽 | 재현 행 수 |\n|---|---|---|---|\n'
intro+='\n'.join('| '+' | '.join(r)+' |' for r in index)+'\n\n'
OUT.mkdir(parents=True,exist_ok=True)
with (OUT/'table_rows.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(manifest[0]));w.writeheader();w.writerows(manifest)
source_paths=[MUSC,APRIL]+[p for p in [live_root/'aggregates.csv',live_root/'rscin_aggregates.csv'] if p.exists()]
source_paths += [p for p in [small_path/'mean_metrics.csv',visa_path/'category_metrics.csv'] if p.exists()]
source_paths += sorted((LAB/'method11/source/results').glob('benchmark_*/benchmark.json'))
source_paths += [p for p in [draem_root/'category_metrics.csv',draem_root/'environment.json',
                            draem_mean,draem_root/'rscin/provenance.json',draem_root/'verification.json'] if p.exists()]
source_paths += [p for p in [nsa_mean,nsa_verified,nsa_root/'environment.json',nsa_root/'category_metrics.csv'] if p.exists()]
source_paths += [p for p in [igd_mean,igd_verified,igd_root/'environment.json',igd_root/'category_metrics.csv',igd_root/'evaluation_provenance.json'] if p.exists()]
source_paths += [p for p in [acr_verified,acr_root/'mean_metrics.csv',acr_root/'category_metrics.csv',acr_root/'environment.json'] if p.exists()]
source_paths += [p for p in [regad_verified,regad_root/'mean_metrics.csv',regad_root/'category_metrics.csv',regad_root/'rscin_mean_metrics.csv',regad_root/'environment.json'] if p.exists()]
source_paths += [p for p in [graphcore_verified,graphcore_root/'mean_metrics.csv',graphcore_root/'category_metrics.csv',graphcore_root/'environment.json'] if p.exists()]
source_paths += [p for p in [vtadl_verified,vtadl_root/'mean_metrics.csv',vtadl_root/'category_metrics.csv',vtadl_root/'environment.json'] if p.exists()]
source_paths += [p for p in [psvdd_verified,psvdd_root/'mean_metrics.csv',psvdd_root/'category_metrics.csv',psvdd_root/'environment.json'] if p.exists()]
source_paths += [p for p in [spade_verified,spade_root/'mean_metrics.csv',spade_root/'category_metrics.csv',spade_root/'environment.json'] if p.exists()]
source_paths += [p for p in [pyramidflow_verified,pyramidflow_root/'mean_metrics.csv',pyramidflow_root/'category_metrics.csv',pyramidflow_root/'environment.json'] if p.exists()]
source_paths += [p for p in [cutpaste_verified,cutpaste_root/'mean_metrics.csv',cutpaste_root/'category_metrics.csv',cutpaste_root/'environment.json'] if p.exists()]
provenance=dict(table_count=18,row_count=len(manifest),paper_sha256=hashlib.sha256(PAPER.read_bytes()).hexdigest(),
                paper_performance_numbers_copied=False,
                sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in source_paths],
                conditions_verified=False,aggregation='unweighted category mean: MVTec 15, VisA 12, BTAD 3; 4-shot mean and sample std across seeds 42,43,44; multiply performance by 100')
(OUT/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
MD.write_text(intro+'\n'.join(sections),encoding='utf-8')
print(MD)
print('tables:',len(index),'rows:',len(manifest))
