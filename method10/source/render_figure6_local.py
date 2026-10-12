"""Compose local metrics and saved local qualitative examples as Figure 6."""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import csv,json,hashlib,html,base64

root=Path(__file__).resolve().parents[2]
out=root/'method10/source/figures/figure6_local'
source=root/'method10/markdown/MuSc_all_reproduction_tables.md'
section=source.read_text(encoding='utf-8-sig').split('## Table 1.',1)[1].split('## Table 2.',1)[0]
lines=[x for x in section.splitlines() if x.startswith('|')]
rows=[[c.strip() for c in x.strip('|').split('|')] for x in lines[2:]]
means=list(csv.DictReader((root/'method10/source/results/tables3_7_seed42_20261012/mean_metrics.csv').open()))
visa=next(r for r in means if r['table']=='7' and r['dataset']=='visa' and r['condition']=='s1')
for row in rows:
    if row[0]=='VisA' and row[2]=='MuSc':
        for col,key in [(4,'image_auroc'),(5,'image_f1_max'),(6,'image_ap'),(7,'pixel_auroc')]: row[col]=str(float(visa[key])*100)
        row[11]='Table3–7 final verified; pixel F1/AP/AUPRO unavailable'

def fmt(v):
    if v in ('미실행','미측정','—','-'):return '-'
    return '±'.join(f'{float(x.strip()):.1f}' for x in v.split('±'))

S=3; W=1000; table_h=30+len(rows)*21; H=table_h+365
im=Image.new('RGB',(W*S,H*S),'white'); draw=ImageDraw.Draw(im)
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W*S}" height="{H*S}" viewBox="0 0 {W} {H}">',f'<rect width="{W}" height="{H}" fill="white"/>']
def line(x1,y1,x2,y2,width=.7):
    draw.line((x1*S,y1*S,x2*S,y2*S),fill='black',width=max(1,round(width*S)))
    svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="black" stroke-width="{width}"/>')
def text(x,y,s,size=12,bold=False,left=False):
    f=ImageFont.truetype('C:/Windows/Fonts/'+('timesbd.ttf' if bold else 'times.ttf'),round(size*S))
    draw.text((x*S,y*S),s,font=f,fill='black',anchor='lm' if left else 'mm')
    svg.append(f'<text x="{x}" y="{y}" text-anchor="{"start" if left else "middle"}" dominant-baseline="central" font-family="Times New Roman,serif" font-size="{size}" font-weight="{"bold" if bold else "normal"}">{html.escape(s)}</text>')
def tile(path,x,y,size=92):
    img=Image.open(path).convert('RGB').resize((size*S,size*S),Image.Resampling.LANCZOS)
    im.paste(img,(round(x*S),round(y*S)))
    data=base64.b64encode(Path(path).read_bytes()).decode()
    svg.append(f'<image x="{x}" y="{y}" width="{size}" height="{size}" href="data:image/png;base64,{data}"/>')

widths=[80,245,65]+[87]*7
centers=[];x=0
for width in widths: centers.append(x+width/2);x+=width
headers=['Dataset','Method','Setting','AUROC-cls','F1-max-cls','AP-cls','AUROC-segm','F1-max-segm','AP-segm','PRO-segm']
line(2,2,W-2,2,1.2);line(2,25,W-2,25)
for x,h in zip(centers,headers):text(x,13,h,size=11.4,bold=True)
authors={'WinCLIP':'WinCLIP (Jeong et al., 2023)','APRIL-GAN':'APRIL-GAN (Chen et al., 2023)','ACR':'ACR (Li et al., 2023)','RegAD':'RegAD (Huang et al., 2022)','PatchCore':'PatchCore (Roth et al., 2022)','GraphCore':'GraphCore (Xie et al., 2023)','MuSc':'MuSc (ours)'}
groups=[]
for i,row in enumerate(rows):
    group=(row[0],row[1])
    if not groups or groups[-1][0]!=group: groups.append([group,i,i])
    else: groups[-1][2]=i
    y=36+i*21;bold=row[2]=='MuSc'
    text(84,y,authors[row[2]],size=12,left=True,bold=bold)
    text(centers[2],y,row[3],size=12)
    for x,v in zip(centers[3:],row[4:11]): text(x,y,fmt(v),size=12,bold=bold)
for group,a,b in groups:
    text(centers[0],36+(a+b)*21/2,group[0],size=12)
    if b<len(rows)-1:line(2,46+b*21,W-2,46+b*21)
line(2,table_h-2,W-2,table_h-2,1.2)

samples=json.loads((out/'samples.json').read_text())
top=table_h+41
for panel,(ds,categories) in enumerate([('mvtec_ad',['bottle','metal_nut','pill']),('visa',['pcb1','cashew','pipe_fryum'])]):
    start=20+panel*510
    for j,label in enumerate(['Image','GT','APRIL-GAN','APRIL-GAN','MuSc']):
        x=start+j*92+46
        if j in (2,3): text(x,top-27,label,size=12);text(x,top-13,'(0-shot)' if j==2 else '(4-shot)',size=12)
        else:text(x,top-20,label,size=13,bold=True)
    for i,category in enumerate(categories):
        y=top+i*92;name=f'{ds}_{category}'
        # Category labels appear beside each row, as in the supplied example.
        label=category
        f=ImageFont.truetype('C:/Windows/Fonts/times.ttf',12*S)
        layer=Image.new('RGBA',(130*S,22*S),(255,255,255,0));ld=ImageDraw.Draw(layer)
        ld.text((65*S,11*S),label,font=f,anchor='mm',fill='black')
        layer=layer.rotate(90,expand=True)
        im.paste(layer,(round((start-20)*S),round((y+46)*S-layer.height/2)),layer)
        svg.append(f'<text x="{start-10}" y="{y+46}" text-anchor="middle" font-family="Times New Roman,serif" font-size="12" transform="rotate(-90 {start-10} {y+46})">{label}</text>')
        tile(out/f'{name}_image.png',start,y)
        tile(out/f'{name}_gt.png',start+92,y)
        for j in (2,3):
            # Missing predictions are shown explicitly, never copied from the paper.
            text(start+j*92+46,y+46,'-',size=24)
        tile(out/f'{name}_musc.png',start+4*92,y)
        for j in range(6): line(start+j*92,y,start+j*92,y+92,.35)
        line(start,y,start+460,y,.35);line(start,y+92,start+460,y+92,.35)
    text(start+230,top+294,'(a) MVTec AD' if panel==0 else '(b) VisA',size=16)
text(W/2,H-11,'Figure 6. Local comparison and qualitative examples.',size=16,bold=True)
im.save(out/'figure6.png',dpi=(300,300))
(out/'figure6.svg').write_text('\n'.join(svg+['</svg>']),encoding='utf-8')
(out/'table_data.json').write_text(json.dumps(dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),rows=rows,visa_musc_additional_source='method10/source/results/tables3_7_seed42_20261012/mean_metrics.csv'),ensure_ascii=False,indent=2),encoding='utf-8')
(out/'README.md').write_text('''# Figure 6 — 로컬 실행 결과

![Figure 6](figure6.png)

[PNG](figure6.png) · [SVG](figure6.svg) · [표 데이터](table_data.json) · [시각화 입력·선택 기록](samples.json)

위 표는 [로컬 비교표 Table 1](../../../markdown/MuSc_all_reproduction_tables.md)의 자체 측정값이다. MuSc VisA의 Image AUROC/F1-max/AP 및 Pixel AUROC는 검증 완료된 [Table 3–7 평균 CSV](../../results/tables3_7_seed42_20261012/mean_metrics.csv)에서 추가했다. 아직 집계하지 않은 MuSc VisA Pixel F1-max/AP/AUPRO는 `-`로 표시한다. 단위는 %, ±는 원본에 저장된 반복 표준편차이며 단일 실행에 임의로 붙이지 않았다. 굵은 행은 MuSc다.

아래 이미지는 각 category의 보존 loader 순서상 첫 이상 이미지를 사용한다. MuSc s=1 기본 patch 점수를 FP64 bilinear(518, align_corners=True)로 복원했으며 각 이미지에서 min-max 정규화 후 jet 색상으로 표시했다. GT는 해당 실행 loader의 이진 mask다. 영상별 정규화이므로 색상을 서로 다른 이미지의 절대 점수로 비교하지 않는다. 입력 SHA-256을 확인하고 원본 patch 점수·복원 map·mask를 각 *_figure_data.npz에 보존했다.

APRIL-GAN 0/4-shot의 숫자 결과는 있으나 원시 map과 시각화 파일은 현재 결과 폴더에서 찾지 못했다. 그 두 시각화 열은 `-`로 표시했다. 논문 히트맵을 자체 결과로 재사용하지 않았다. 정확한 논문 샘플 일치를 주장하지 않는다. pcb1은 원문의 pcb 이미지 계열에 해당하는 로컬 category다.

MuSc 근거: [최종 검증](../../results/tables3_7_seed42_20261012/verification.json) · [실행 조건](../../../markdown/MuSc_tables3_7_execution.md).
''',encoding='utf-8')
print(out/'figure6.png')
