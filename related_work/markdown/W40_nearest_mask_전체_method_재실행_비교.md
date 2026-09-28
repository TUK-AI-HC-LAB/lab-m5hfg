# W40 nearest mask 처리 후 전체 method 재실행 비교

> GitHub 코드·결과 근거: [공통 mask 전처리](../../method9/source/common_framework_validation_patch/framework_snapshot/datasets/base.py), [재실행 script](../../method9/source/common_framework_validation_patch/framework_snapshot/run_all_methods_bottle_nearest_mask_w40.sh), [raw CSV·status](../../method9/source/common_framework_validation_patch/results/all_methods_nearest_mask_20260928/).

## 목적

공통 framework의 정답 mask Resize 보간을 `bilinear`에서 `nearest`로 변경한 뒤, GLASS를 제외한 모든 등록 method를 MVTec AD `bottle`에서 다시 실행한다.

```text
기존: bilinear로 경계에 중간값 생성 → 0이 아니면 전부 결함
수정: nearest로 원래 0/1 경계 유지
```

## 변경 범위

| 코드 | 변경 |
|---|---|
| `datasets/base.py` | 공통 MVTec mask `Resize`에 `InterpolationMode.NEAREST` 지정 |
| `datasets/anomalyclip_mvtec.py` | AnomalyCLIP 전용 mask `Resize`에 `InterpolationMode.NEAREST` 지정 |

이미지 전처리, 모델 가중치, 학습, anomaly score/map 생성 코드는 바꾸지 않는다. 따라서 이 변경이 영향을 줄 수 있는 값은 정답 mask를 사용하는 Pixel AUROC와 saliency F1이며, Image AUROC와 prediction map은 원칙적으로 바뀌지 않아야 한다.

## 실행 설정

| 항목 | 값 |
|---|---|
| Dataset | MVTec AD `bottle` |
| 학습 | 정상 이미지 209장 |
| 평가 | 정상 20장 + 이상 63장 |
| seed | 0 |
| 대상 | PatchCore, PaDiM, WinCLIP, COAD, SimpleNet, RD, RD-Orig, PromptAD, Dinomaly, UniAD, AnomalyCLIP |
| 제외 | GLASS — DTD 텍스처 데이터 의존성이 아직 준비되지 않음 |
| 실행 script | [`run_all_methods_bottle_nearest_mask_w40.sh`](../../method9/source/common_framework_validation_patch/framework_snapshot/run_all_methods_bottle_nearest_mask_w40.sh) |
| 결과 | [`results/`](../../method9/source/common_framework_validation_patch/results/all_methods_nearest_mask_20260928/) |

## 결과 표

전체 실행이 끝난 뒤 아래 표에 기존 bilinear 실행과 nearest 실행의 결과를 기록한다.

| Method | 기존 Image AUROC | nearest Image AUROC | Image 차이 | 기존 Pixel AUROC | nearest Pixel AUROC | Pixel 차이 | 기존 Saliency F1 | nearest Saliency F1 | F1 차이 | 상태 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| PatchCore | 100.00% | 100.00% | +0.00%p | 98.78% | 98.69% | -0.09%p | 67.35% | 67.42% | +0.07%p | 완료 |
| PaDiM | 100.00% | 100.00% | +0.00%p | 98.72% | 98.61% | -0.11%p | 71.54% | 70.84% | -0.70%p | 완료 |
| WinCLIP | 99.92% | 99.92% | +0.00%p | 95.85% | 95.69% | -0.16%p | 70.52% | 69.26% | -1.26%p | 완료 |
| COAD | 100.00% | 100.00% | +0.00%p | 99.25% | 99.15% | -0.10%p | 69.03% | 69.58% | +0.55%p | 완료 |
| SimpleNet | 99.44% | 99.44% | +0.00%p | 87.76% | 88.41% | +0.65%p | 53.08% | 52.31% | -0.77%p | 완료 |
| RD | 100.00% | 100.00% | +0.00%p | 98.82% | 98.72% | -0.10%p | 75.33% | 74.66% | -0.67%p | 완료 |
| RD-Orig | 100.00% | 100.00% | +0.00%p | 98.85% | 98.75% | -0.10%p | 75.95% | 75.15% | -0.80%p | 완료 |
| PromptAD | 100.00% | 100.00% | +0.00%p | 99.02% | 98.98% | -0.04%p | 80.24% | 79.81% | -0.43%p | 완료 |
| Dinomaly | 100.00% | 100.00% | +0.00%p | 99.34% | 99.26% | -0.08%p | 60.53% | 61.37% | +0.84%p | 완료 |
| UniAD | 100.00% | 100.00% | +0.00%p | 98.48% | 98.35% | -0.13%p | 68.63% | 68.09% | -0.54%p | 완료 |
| AnomalyCLIP | 88.81% | 88.81% | +0.00%p | 90.38% | 90.31% | -0.07%p | - | - | - | 완료 |

## 해석 원칙

- Image AUROC 또는 raw anomaly map이 달라지면, mask 보간 외의 변수가 달라졌는지 실행 config·seed·checkpoint를 먼저 확인한다.
- Pixel AUROC/F1만 달라지면, 값의 변화는 모델이 결함을 더 잘 찾았다는 뜻이 아니라 **동일한 prediction map을 새 정답 mask 경계로 평가한 결과**일 수 있다.
- `bottle`은 정사각형 원본이므로 bilinear와 nearest의 차이가 작거나 없을 수 있다. 여러 직사각형 VisA 범주까지 바꾸었다는 결론은 내리지 않는다.

## 결과 해석: 무엇이 개선되었는가

### 1. 실행 범위

- GLASS를 제외한 11개 method가 모두 끝까지 실행되어 CSV를 만들었다.
- GLASS는 모델 문제로 제외한 것이 아니라, 정상 이미지에 합성 이상을 만들 때 필요한 DTD 텍스처 데이터가 없어서 이번에도 제외했다.

### 2. 모델 출력은 바뀌지 않았다

11개 method 모두 Image AUROC가 기존 값과 `0.00%p` 차이다. 이번 코드 변경은 정답 mask만 바꾸고 image tensor·모델 학습·추론·anomaly map 계산을 바꾸지 않았으므로, 예상한 대로 image-level 판별은 유지됐다.

### 3. Pixel 평가는 경계 정의에 맞게 바뀌었다

- Pixel AUROC 변화 범위는 `-0.16%p ~ +0.65%p`다.
- Saliency F1 변화 범위는 `-1.26%p ~ +0.84%p`다.
- bilinear는 경계의 0과 1 사이에 중간값을 만들고, 기존 코드의 `mask[mask != 0] = 1`이 그 중간값을 모두 결함으로 확장했다. nearest는 원래 mask의 0/1 레이블을 유지한다.

즉 개선된 점은 어떤 방법의 anomaly map이 더 좋아진 것이 아니라, **모든 method가 경계가 부풀지 않은 같은 의미의 정답 mask로 pixel-level 지표를 계산하게 된 것**이다. 값이 일부 낮아진 것은 결함 판정이 더 어려워졌다는 뜻일 수 있으며, 모델 성능 저하로 해석하지 않는다.

### 4. 이번 `bottle` 결과의 한계

`bottle`은 정사각형 원본이라 공간 왜곡 차이는 작다. nearest의 효과를 더 엄밀히 확인하려면 직사각형 원본이 있는 VisA 여러 범주에서 공식 image/mask 변환과 공통 변환을 비교해야 한다. 앞서 VisA `candle`에서는 공통 정사각형 mask 처리만 바꾸었을 때 Pixel AUROC가 -0.36%p 달라진 사례가 있다.

## Raw 결과

각 method의 CSV는 [`results/all_methods_nearest_mask_20260928/`](../../method9/source/common_framework_validation_patch/results/all_methods_nearest_mask_20260928/)에 있다.

## VisA 직사각형 원본 추가 검증

위 11개 method 재실행은 정사각형 원본인 MVTec AD `bottle` 조건이다. nearest mask의 공간 정합성 효과를 더 직접적으로 확인하기 위해, 원본이 모두 직사각형인 VisA 12개 test 범주에서도 AnomalyCLIP을 추가로 평가한다.

- image: 공식 `shortest-edge Resize(518) → CenterCrop(518)`로 고정
- mask 공식 조건: `shortest-edge Resize(518, nearest) → CenterCrop(518)`
- mask 공통 조건: `Resize((518,518), nearest)`
- 모델·checkpoint·anomaly map: 두 조건에서 고정

따라서 이 추가 검증에서의 Pixel AUROC 차이는 **mask 공간 좌표 변환 차이**이고, Image AUROC와 anomaly map의 성능 차이가 아니다. 범주별 결과와 raw JSON은 [W40 VisA 직사각형 mask 공간변환 비교](../../method6/markdown/W40_VisA_직사각형_mask_공간변환_비교.md)에 기록한다.
