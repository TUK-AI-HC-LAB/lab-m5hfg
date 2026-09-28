# W40 AnomalyCLIP mask 공통 처리 비교

## 질문

AnomalyCLIP의 정답 mask를 공식 공간 변환 대신 공통 Dataset 방식으로 처리하면 `bottle` 평가 결과가 달라지는가?

## 고정한 조건

- Dataset: MVTec AD `bottle` test 83장
- Prompt checkpoint: ViSA → MVTec 학습 checkpoint `epoch_15.pth`
- 이미지 전처리: 두 조건 모두 공식 AnomalyCLIP `shortest-edge Resize(518) → CenterCrop(518)`
- 모델, text prompt, anomaly map 계산: 두 조건 모두 동일

## 바꾼 한 가지

| 조건 | 정답 mask 공간 변환 |
|---|---|
| 공식 | `shortest-edge Resize(518) → CenterCrop(518) → ToTensor` |
| 공통 | `Resize((518, 518)) → ToTensor` |

공통 조건에서도 **이미지 변환은 바꾸지 않았다**. 따라서 이 비교는 모델 성능 비교가 아니라 mask 좌표 변환의 평가 영향만 확인한다.

## 결과

| 지표 | 공식 mask | 공통 mask | 차이 (공통 - 공식) |
|---|---:|---:|---:|
| Image AUROC | 88.81% | 88.81% | 0.00%p |
| Pixel AUROC | 90.38% | 90.38% | 0.00%p |
| 예측 image score | bitwise 동일 | bitwise 동일 | - |
| 예측 anomaly map | bitwise 동일 | bitwise 동일 | - |
| 정답 mask에서 다른 픽셀 | 0 / 22,270,892 | 0 / 22,270,892 | 0.00% |

## 직사각형 VisA `candle` 검증

MVTec AD에는 직사각형 test 원본이 없으므로, 원본 크기가 `1284×1168`인 VisA `candle` test 200장을 추가로 평가했다. 이 조건에서는 MVTec → VisA AnomalyCLIP prompt checkpoint를 사용했다.

| 지표 | 공식 mask | 공통 mask | 차이 (공통 - 공식) |
|---|---:|---:|---:|
| Image AUROC | 83.91% | 83.91% | 0.00%p |
| Pixel AUROC | 98.65% | 98.29% | -0.36%p |
| 정답 mask에서 다른 픽셀 | 82,348 / 53,664,800 |  | 0.153% |

모델은 한 번만 실행했으며, image score와 anomaly map은 두 평가 조건에서 같은 값을 사용했다. 따라서 Pixel AUROC의 -0.36%p 차이는 공통 mask의 강제 정사각형 Resize가 공식 이미지 좌표계와 맞지 않았기 때문에 생긴 평가 차이다.

## 해석

`bottle`의 원본 이미지와 정답 mask는 정사각형이다. 따라서 두 변환 모두 같은 518×518 좌표 결과를 만들었다. 이 범주에서는 mask 처리 변경이 평가 결과에 영향을 주지 않았다.

`bottle` 결과만으로 모든 범주에서 안전하다고 말할 수는 없다. 실제로 직사각형 VisA `candle`에서는 `shortest-edge Resize → CenterCrop`과 강제 정사각형 Resize가 서로 다른 좌표를 만들었고 Pixel AUROC가 달라졌다. 따라서 이미지와 mask에는 같은 기하 변환을 적용해야 한다.

## 재현 경로

- 비교 코드: [`evaluate_anomalyclip_common_mask.py`](../source/evaluate_anomalyclip_common_mask.py)
- raw 결과: [`w40_anomalyclip_mask_transform_comparison.json`](../source/results/w40_anomalyclip_mask_transform_comparison.json)
- VisA 직사각형 검증 코드: [`evaluate_anomalyclip_visa_mask_geometry.py`](../source/evaluate_anomalyclip_visa_mask_geometry.py)
- VisA raw 결과: [`w40_anomalyclip_visa_candle_mask_transform_comparison.json`](../source/results/w40_anomalyclip_visa_candle_mask_transform_comparison.json)
- 외부 공통 framework: `C:\Users\test\Downloads\dinomaly_share_codebase\dinomaly_share_codebase` (저장소에 복사하지 않음)
