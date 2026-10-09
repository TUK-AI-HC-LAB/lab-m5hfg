"""Render measured DRAEM results into its research report."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/mvtec_public_20261007'
MD=ROOT.parent/'markdown/DRAEM_mvtec_official_execution.md'
def read(path):
    with path.open() as f:return list(csv.DictReader(f))
env=json.loads((OUT/'environment.json').read_text())
verification=json.loads((OUT/'verification.json').read_text())
assert env['status']=='completed' and verification['status']=='passed'
native=read(OUT/'category_metrics.csv');mean=read(OUT/'mean_metrics.csv')[0]
rscin=read(OUT/'rscin/category_metrics.csv');rmean=read(OUT/'rscin/mean_metrics.csv')
metrics=['image_auroc','image_f1_max','image_ap','pixel_auroc','pixel_f1_max','pixel_ap','aupro']
names=['Image AUROC','Image F1-max','Image AP','Pixel AUROC','Pixel F1-max','Pixel AP','AUPRO']
text=MD.read_text(encoding='utf-8').split('\n## 자체 측정 결과')[0]
text=text.replace('- result: 실행 진행 중; 완료 후 자체 CSV로 표를 생성한다.',
    '- result: [카테고리 CSV](../source/results/mvtec_public_20261007/category_metrics.csv), '+
    '[환경·가중치 SHA-256](../source/results/mvtec_public_20261007/environment.json), '+
    '[검증](../source/results/mvtec_public_20261007/verification.json)')
lines=['','## 자체 측정 결과','',
    'MVTec AD 15개 category, test 1,725장 전체 완료. 단위는 %. Mean은 이미지 수로 가중하지 않은 카테고리별 지표의 단순 평균이다. 단일 공개 체크포인트 평가이므로 seed 반복 표준편차는 보고하지 않는다. F1-max는 평가 라벨로 threshold를 탐색한 값이다.','',
    '| Category | Images | '+' | '.join(names)+' |',
    '|---|---|'+'---|'*len(metrics)]
for r in native+[mean]:
    lines.append('| '+r['category']+' | '+r['n_images']+' | '+' | '.join(f'{100*float(r[k]):.4f}' for k in metrics)+' |')
lines+=['','[반올림 전 전체 평균 CSV](../source/results/mvtec_public_20261007/mean_metrics.csv)','',
    'AUPRO는 DRAEM의 공식 test 출력에 없는 보조 지표다. APRIL-GAN 함수의 200 threshold, FPR<0.3 및 선택 FPR 범위를 min-max 정규화하는 구현을 적용했다. 원문 저자의 AUPRO와 동일하다고 단정하지 않는다.','',
    '## RsCIN 적용 전후','',
    '같은 DRAEM 이미지 점수에만 공식 Mobile_RsCIN을 적용했다. 분할 맵·픽셀 지표는 다시 생성하지 않는다. 공유 MuSc class token을 원래 DRAEM 이미지 순서로 정렬하고 이미지 경로·라벨·SHA-256을 모두 대조했다.','',
    '| Category | w/o AUROC | w AUROC | Δ AUROC (pp) | w/o F1-max | w F1-max | w/o AP | w AP |',
    '|---|---|---|---|---|---|---|---|']
for cat in [r['category'] for r in native]+['macro_mean']:
    selected=rmean if cat=='macro_mean' else [r for r in rscin if r['category']==cat]
    before=next(r for r in selected if r['rscin']=='w/o');after=next(r for r in selected if r['rscin']=='w')
    vals=[float(before['image_auroc']),float(after['image_auroc']),float(after['image_auroc'])-float(before['image_auroc']),
        float(before['image_f1_max']),float(after['image_f1_max']),float(before['image_ap']),float(after['image_ap'])]
    lines.append('| '+cat+' | '+' | '.join(f'{v*100:.4f}' for v in vals)+' |')
before=next(r for r in rmean if r['rscin']=='w/o');after=next(r for r in rmean if r['rscin']=='w')
delta=100*(float(after['image_auroc'])-float(before['image_auroc']))
lines+=['',
    '[RsCIN 카테고리 CSV](../source/results/mvtec_public_20261007/rscin/category_metrics.csv) · '+
    '[평균 CSV](../source/results/mvtec_public_20261007/rscin/mean_metrics.csv) · '+
    '[이미지·특징 정렬 검증](../source/results/mvtec_public_20261007/rscin/dataset_alignment.csv) · '+
    '[RsCIN provenance](../source/results/mvtec_public_20261007/rscin/provenance.json)','',
    '## 결과 해석과 검증','',
    f'전체 이미지 점수와 픽셀 맵이 유효하게 생성되어 실행 전 가설의 통합 동작 부분을 지지한다. RsCIN 적용 후 평균 Image AUROC는 {delta:+.4f} percentage point 변했다. 이는 이 공개 가중치와 로컬 공유 특징 조건에서의 관찰이며 저자의 Table 11 수치 재현이나 카테고리별 개선 보장은 아니다.','',
    '원시 NPZ에서 공식 4개 지표를 다시 계산하고 실행 중 수집된 값과 절대 오차 1e-12 이내 일치를 확인했다. 원시 픽셀 맵에 공식 21×21 pooling을 다시 적용한 이미지 점수는 저장된 FP32 점수와 정확히 일치했다. 30개 가중치의 해시, 15개 category·1,725장 수, 현재 원본 이미지의 해시, 비가중 평균, RsCIN 특징의 라벨·해시 정렬을 검증했다. 보조 AUPRO와 F1-max의 저자 환경 수치 동등성은 검증하지 않았다.','',
    f'공식 평가와 결과 수집·압축·보조 지표 계산을 포함한 wall time은 {env["wall_seconds"]/60:.2f}분이다. 다운로드·검증·RsCIN·보고서 시간은 제외한다. 순수 추론 시간으로 해석하지 않는다. PyTorch peak allocated는 {env["peak_allocated_bytes"]/1024**2:.2f} MiB다.','',
    '한계: 공개 가중치를 사용했으므로 현재 PC에서 DRAEM의 학습을 재현한 결과는 아니다. 학습 epoch·학습 seed·저자의 개별 train 이력은 가중치 파일명만으로 확인할 수 없다. 현대 PyTorch/OpenCV와 저자 의존성의 차이가 있고, MuSc Table 11에서 저자가 사용한 개별 DRAEM 가중치·class token 파일과의 동일성을 확인하지 않았다. 따라서 표의 상태를 조건 대조 필요로 유지한다.','',
    '[MuSc 전체 비교표 Table 11](../../method10/markdown/MuSc_all_reproduction_tables.md#table-11-비교군에-rscin을-적용한-classification-결과)','']
MD.write_text(text+'\n'.join(lines),encoding='utf-8')
print('REPORT COMPLETE',MD)
