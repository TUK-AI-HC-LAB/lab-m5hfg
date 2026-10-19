from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'markdown' / 'images' / 'high_speed_industrial_line'
OUT.mkdir(parents=True, exist_ok=True)
im = Image.new('RGB', (1600, 1050), '#f3f6fa')
d = ImageDraw.Draw(im)
font_path = 'C:/Windows/Fonts/malgun.ttf'
def font(n): return ImageFont.truetype(font_path, n)
def text(x, y, s, size=25, color='#21334a'):
    d.text((x, y), s, font=font(size), fill=color)
def vial(x,y,w=46,h=90,defect=False):
    d.rounded_rectangle((x+9,y,x+w-9,y+20),4,fill='#dce8f3',outline='#527292',width=2)
    d.rounded_rectangle((x,y+18,x+w,y+h),9,fill='#eef5fa',outline='#527292',width=2)
    d.rectangle((x+3,y+h//2,x+w-3,y+h-10),fill='#b8dcec')
    if defect: d.ellipse((x+15,y+58,x+25,y+68),fill='#e45050')
def arrow(a,b):
    d.line((a,b),fill='#7890a8',width=5)
    x,y=b; ax,ay=a
    if x>ax: p=[(x,y),(x-14,y-9),(x-14,y+9)]
    else: p=[(x,y),(x-9,y-14),(x+9,y-14)]
    d.polygon(p,fill='#7890a8')
def card(x,y,n,title):
    d.rounded_rectangle((x,y,x+470,y+325),20,fill='white',outline='#d5deea',width=2)
    d.rounded_rectangle((x+20,y+20,x+61,y+61),10,fill='#2463a6')
    text(x+32,y+24,str(n),25,'white'); text(x+77,y+23,title,27)

text(50,30,'생산라인에서 불량을 찾는 과정',43)
text(52,94,'논문 설정을 바탕으로 재구성한 설명용 그림 · 실제 장비 사진이나 실험 출력이 아님',23,'#586d83')
xs=[50,565,1080]; y=160
for i,t in enumerate(['제품 연속 촬영','3개 프레임 선택','작은 영역으로 분할']): card(xs[i],y,i+1,t)
for i in range(5): vial(106+i*67,255,defect=i==2)
text(75,374,'액체 용기 5개가 붙은 제품',26)
text(75,419,'움직이며 촬영 → 정상 기포도 발생',22)
for j,label in enumerate(['처음','8번째','마지막']):
    x=590+j*143
    d.rounded_rectangle((x,260,x+122,351),8,fill='#edf3fa',outline='#8ba1b9',width=2)
    vial(x+44,270,32,67)
    text(x+19,363,label,24)
text(590,419,'모든 프레임 대신 선택한 3장 검사',22)
for j in range(5):
    x=1120+j*77
    for k in range(4): d.rectangle((x,260+k*28,x+49,283+k*28),fill=['#e8ddef','#d8e8f8','#c3e3ed','#f2e4bd'][k],outline='#7890a8',width=2)
text(1105,389,'3장 × 용기 5개 × 영역 4개',25)
text(1105,427,'= 60개 패치를 묶어서 처리',23)
arrow((525,320),(558,320)); arrow((1040,320),(1073,320))
y=540
for i,t in enumerate(['이미지 복원','원본과 복원본 비교','제품 단위 불량 판정']): card(xs[i],y,i+4,t)
vial(105,646,65,110,True); vial(367,646,65,110)
arrow((190,697),(343,697)); text(214,648,'복원 모델',22)
text(85,768,'입력',23);text(345,768,'복원 예시',23)
text(75,819,'정상 구조 복원을 학습한 모델 사용',22)
d.rounded_rectangle((604,641,989,738),12,fill='#fff2ec')
text(626,659,'이상 점수 = 1 - SSIM',29)
text(602,764,'구조가 다를수록 이상 점수 증가',24)
text(590,812,'픽셀 차이 → 의심 위치를 색으로 표시',22)
text(1110,649,'한 영역이라도 기준을 넘으면',25)
d.rounded_rectangle((1120,704,1500,768),12,fill='#fff0ef')
text(1220,716,'불량 판정',30,'#c13e3e')
text(1105,810,'운영자에게 의심 위치 표시',24)
arrow((525,700),(558,700)); arrow((1040,700),(1073,700))
d.line([(1315,493),(1315,510),(285,510)],fill='#7890a8',width=3)
arrow((285,510),(285,533))
text(60,906,'처리 제약: 연속 촬영 사이 500ms 안에 검사',32,'#2463a6')
text(60,960,'주의: 프레임별 검사 결과를 통합하는 방식. 동작 순서 학습이나 온라인 모델 갱신을 뜻하지 않음.',23)
text(60,1003,'출처: Ferrari et al., arXiv:2603.07577v1, §5.2 및 §6 / 복원 그림은 원리 설명을 위한 예시',19,'#586d83')
im.save(OUT/'inspection_workflow.png')
print(OUT/'inspection_workflow.png')
