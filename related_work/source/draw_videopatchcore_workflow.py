from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math

OUT = Path(__file__).resolve().parents[1] / 'markdown/images/videopatchcore'
OUT.mkdir(parents=True, exist_ok=True)
im = Image.new('RGB', (1900, 1500), '#f3f6fa')
d = ImageDraw.Draw(im)
INK, MUTED, BLUE = '#21334a', '#586d83', '#2463a6'
def font(n, bold=False):
    return ImageFont.truetype('C:/Windows/Fonts/' + ('malgunbd.ttf' if bold else 'malgun.ttf'), n)
def text(x,y,s,size=25,color=INK,bold=False):
    d.text((x,y),s,font=font(size,bold),fill=color)
def box(x,y,w,h,title,accent=BLUE):
    d.rounded_rectangle((x,y,x+w,y+h),18,fill='white',outline='#d5deea',width=2)
    d.rounded_rectangle((x+20,y+20,x+27,y+58),3,fill=accent)
    text(x+42,y+21,title,29,accent,True)
def arrow(a,b,color='#7890a8',width=5):
    d.line((a,b),fill=color,width=width)
    angle=math.atan2(b[1]-a[1],b[0]-a[0]); x,y=b
    d.polygon([(x,y),(x-17*math.cos(angle-.5),y-17*math.sin(angle-.5)),
               (x-17*math.cos(angle+.5),y-17*math.sin(angle+.5))],fill=color)
def human(x,y,shift=0,color=BLUE):
    x+=shift
    d.ellipse((x-9,y,x+9,y+18),fill=color)
    d.line([(x,y+20),(x,y+49)],fill=color,width=6)
    d.line([(x-15,y+33),(x,y+24),(x+15,y+35)],fill=color,width=5)
    d.line([(x-15,y+69),(x,y+48),(x+16,y+67)],fill=color,width=5)
def frames(x,y):
    for i in range(3):
        a=x+i*148
        d.rounded_rectangle((a,y,a+130,y+110),8,fill='#edf3fa',outline='#91a6bd',width=2)
        d.line((a+8,y+91,a+122,y+91),fill='#becbd9',width=2)
        human(a+43,y+17,i*13)
        text(a+42,y+117,f't+{i}',20,MUTED)

text(55,28,'VideoPatchCore: 정상 기억과 다른 외형·움직임·문맥을 찾는다',43,bold=True)
text(58,94,'논문을 바탕으로 재구성한 설명용 도식 · 실제 입력 영상이나 실험 출력이 아님',25,MUTED)

text(55,155,'A. 정상 메모리 구축',32,BLUE,True)
box(55,215,500,270,'정상 동영상 입력')
frames(85,305)
box(700,215,500,270,'고정된 특징 추출기')
text(730,303,'객체: YOLOv5로 영역 검출',27)
text(730,349,'특징: 사전학습 CLIP 인코더',27)
text(730,405,'객체별 특징 + 전체 장면 특징',25,MUTED)
box(1345,215,500,270,'공간·시간·장면으로 분리')
text(1375,303,'국소 분기: 객체 외형·움직임',27)
text(1375,349,'전역 분기: 전체 장면 문맥',27)
text(1375,405,'연속 프레임의 특징을 이용',25,MUTED)
arrow((565,350),(687,350)); arrow((1210,350),(1332,350))

colors=['#2463a6','#187c77','#8353a1']
titles=['공간 메모리 · SM','시간 메모리 · TM','고수준 의미 메모리 · HSM']
lines=[['객체가 어떻게 생겼는가?','외형 특징을 저장'],
       ['객체가 어떻게 움직이는가?','인접 프레임 특징 차이'],
       ['전체 장면은 어떤 흐름인가?','시공간 문맥 특징을 저장']]
for i,x in enumerate([55,700,1345]):
    box(x,580,500,220,titles[i],colors[i])
    text(x+28,665,lines[i][0],26)
    text(x+28,710,lines[i][1],25,MUTED)
    for j in range(10):
        xx=x+32+j*44
        d.rounded_rectangle((xx,758,xx+25,777),4,fill=colors[i])
d.line([(1595,495),(1595,532),(305,532)],fill='#7890a8',width=4)
for x in [305,950,1595]: arrow((x,532),(x,568))
text(57,825,'정상 특징을 coreset으로 압축해 보관 → 검사 단계에서 비교 기준으로 사용',27,MUTED)

text(55,900,'B. 새 동영상 검사',32,BLUE,True)
box(55,964,500,235,'검사 프레임 묶음')
frames(85,1042)
box(700,964,500,235,'같은 세 종류의 특징 생성')
text(730,1055,'외형 / 프레임 간 변화 / 장면 문맥',25)
text(730,1103,'구축 단계와 같은 특징 표현',25,MUTED)
box(1345,964,500,235,'각 정상 메모리와 최근접 검색')
text(1375,1051,'가장 가까운 정상 특징과의 거리',25)
text(1375,1097,'거리가 클수록 이상 점수 증가',25,'#b94646')
arrow((565,1080),(687,1080)); arrow((1210,1080),(1332,1080))
arrow((1595,808),(1595,952),colors[2])
text(1613,851,'정상 기준 참조',22,colors[2])
text(1613,886,'SM · TM · HSM 모두',20,MUTED)

box(700,1270,1145,130,'세 점수 결합 → 프레임별 이상 점수',BLUE)
text(735,1343,'공간·시간 → 국소 점수   +   장면 문맥 → 전역 점수',26)
arrow((1595,1209),(1595,1258))
text(57,1285,'이상은 시간축의',27)
text(57,1329,'어느 구간에 있는가?',27,bold=True)
text(57,1430,'동영상 확장: 시간 정보를 사용함 / 검사 중 메모리를 계속 갱신하는 온라인 적응과는 구분',25,MUTED)
text(57,1470,'출처: Ahn et al., VideoPatchCore, ACCV 2024, §3–4 · arXiv:2409.16225v5',19,MUTED)
im.save(OUT / 'videopatchcore_workflow.png')
print(OUT / 'videopatchcore_workflow.png')
