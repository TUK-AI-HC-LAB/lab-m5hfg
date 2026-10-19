from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math

ROOT = Path(__file__).resolve().parents[1] / 'markdown/images'
INK, BLUE, MUTED = '#21334a', '#2463a6', '#586d83'

def build(kind):
    im = Image.new('RGB', (1800, 1330), '#f3f6fa')
    d = ImageDraw.Draw(im)
    def text(x,y,s,n=26,color=INK,bold=False):
        f = ImageFont.truetype('C:/Windows/Fonts/'+('malgunbd.ttf' if bold else 'malgun.ttf'),n)
        d.text((x,y),s,font=f,fill=color)
    def arrow(a,b,color='#7890a8'):
        d.line((a,b),fill=color,width=5)
        t=math.atan2(b[1]-a[1],b[0]-a[0]); x,y=b
        d.polygon([(x,y),(x-18*math.cos(t-.5),y-18*math.sin(t-.5)),
                   (x-18*math.cos(t+.5),y-18*math.sin(t+.5))],fill=color)
    def card(x,y,num,title,lines,color=BLUE):
        d.rounded_rectangle((x,y,x+520,y+345),18,fill='white',outline='#d5deea',width=2)
        d.rounded_rectangle((x+22,y+23,x+66,y+67),10,fill=color)
        text(x+34,y+25,str(num),27,'white',True)
        text(x+82,y+27,title,28,color,True)
        for j,s in enumerate(lines): text(x+26,y+150+j*43,s,25)
    if kind=='high_speed_industrial_line':
        title='고속 생산라인 모델: 정상 구조를 복원하도록 학습'
        subtitle='가중치 학습 → 별도 임계값 보정 → 생산라인 검사'
        cards=[
            ('정상 패치 준비',['256 × 256 흑백 영역','학습 정답은 원래 정상 패치 X','학습 / 검증: 90% / 10%']),
            ('입력에 잡음 더하기',['Perlin 잡음을 확률 0.75로 적용','작은 회전·수직 뒤집기 추가','원본 X는 복원 목표로 유지']),
            ('생성기로 복원',['인코더 → 64차원 병목 → 디코더','잡음 입력에서 정상 구조 복원','복원 특징의 일관성도 확인']),
            ('원본과 복원 결과 비교',['Huber + SSIM: 픽셀·구조 복원','판별기 특징: 원본과 복원 정렬','잠재 특징 일관성 + 잡음 손실']),
            ('GAN 교대 최적화',['생성기와 판별기를 번갈아 갱신','Adam · batch 32 · 10 epochs','초기 학습률 0.00015']),
            ('학습 종료 후 보정·검사',['별도 정상·불량 데이터로 임계값 보정','최종 테스트와 보정 데이터 분리','검사: 원본과 복원본 차이로 판정'])]
        note='정상 원본으로 복원을 학습하지만, 최종 판정 임계값 보정에는 실제 정상·불량 샘플을 사용한다.'
        source='출처: Ferrari et al., arXiv:2603.07577v1, §5.1–5.2 및 §6.3'
    else:
        title='VideoPatchCore: 가중치 학습 대신 정상 메모리 구축'
        subtitle='사전학습 모델 고정 → 정상 영상 특징 수집 → 대표 메모리 저장'
        cards=[
            ('정상 영상 준비',['정상 학습 영상의 연속 프레임','외형·움직임·장면의 정상 기준','별도 결함 영상으로 분류 학습 안 함']),
            ('사전학습 모델로 특징 추출',['YOLOv5: 객체 영역 검출','CLIP ResNet-101: 시각 특징','신경망 가중치는 고정']),
            ('세 종류의 특징 구성',['공간: 객체의 외형','시간: 인접 프레임 특징 차이','고수준: 장면의 시공간 문맥']),
            ('정상 특징 후보 수집',['공간 SM / 시간 TM / 의미 HSM','각 종류의 정상 특징을 모음','복원 정답·손실·역전파는 없음']),
            ('대표 특징으로 압축',['Greedy Coreset Subsampling','특징 공간을 대표하는 부분집합 선택','세 메모리 각각 중복 감소']),
            ('메모리 고정 후 검사',['새 영상에서 같은 종류 특징 추출','정상 메모리와 최근접 거리 계산','세 점수 결합 → 프레임 이상 점수'])]
        note='Training-free는 추가 가중치 학습이 없다는 뜻이다. 정상 영상으로 메모리를 만드는 준비는 필요하다.'
        source='출처: Ahn et al., VideoPatchCore, ACCV 2024, §3–4 · arXiv:2409.16225v5'
    text(50,30,title,42,bold=True)
    text(54,100,subtitle,27,MUTED)
    xs=[50,640,1230]
    for i,(h,lines) in enumerate(cards):
        x=xs[i%3]; y=190 if i<3 else 665
        card(x,y,i+1,h,lines,'#187c77' if i>=3 else BLUE)
    for y in [360,835]:
        arrow((580,y),(627,y)); arrow((1170,y),(1217,y))
    d.line([(1490,548),(1490,592),(310,592)],fill='#7890a8',width=4)
    arrow((310,592),(310,653))
    text(50,1080,note,27,BLUE,True)
    if kind=='high_speed_industrial_line':
        text(54,1140,'학습의 핵심: 훼손한 입력을 그대로 복사하지 않고 정상적인 구조를 되찾도록 만든다.',26)
        text(54,1190,'GAN 판별기는 학습을 돕는다. 기존 GRD-Net의 별도 U-Net 분할 분기는 이 적용에서 사용하지 않는다.',24,MUTED)
    else:
        text(54,1140,'구축의 핵심: 대표적인 정상 외형·움직임·문맥을 저장해 검사 시 검색할 기준을 만든다.',26)
        text(54,1190,'메모리 압축은 특징 선택 과정이며, 검사 중 계속 학습하거나 메모리를 갱신하는 과정과 다르다.',24,MUTED)
    text(54,1260,'논문 기반 설명용 도식 · 실제 학습 출력이나 실험 데이터가 아님',22,MUTED)
    text(54,1300,source,19,MUTED)
    target=ROOT/kind/'training_workflow.png'
    target.parent.mkdir(parents=True,exist_ok=True)
    im.save(target)
    print(target)

if __name__=='__main__':
    build('high_speed_industrial_line')
    build('videopatchcore')
