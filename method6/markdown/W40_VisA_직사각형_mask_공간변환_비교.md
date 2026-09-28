# W40 VisA 직사각형 범주 mask 공간 변환 비교

## 질문

직사각형 원본을 가진 VisA 여러 범주에서 AnomalyCLIP의 공식 image/mask 공간 변환과 공통 정사각형 mask 변환은 Pixel AUROC를 얼마나 다르게 만드는가?

## 고정 조건

- 모델 입력 image: 공식 AnomalyCLIP `shortest-edge Resize(518) → CenterCrop(518)`
- 모델: MVTec AD → VisA 학습 AnomalyCLIP prompt checkpoint `epoch_15.pth`
- mask 보간: 두 조건 모두 `nearest`
- 변경 변수: mask의 공간 변환 순서만 변경

| 조건 | 정답 mask 변환 |
|---|---|
| 공식 | `shortest-edge Resize(518) → CenterCrop(518) → ToTensor` |
| 공통 | `Resize((518, 518)) → ToTensor` |

따라서 image score와 anomaly map은 조건 사이에서 같고, Pixel AUROC 차이는 mask 좌표계 차이만 반영한다.

## 결과

실행 중. 각 범주의 Image AUROC·Pixel AUROC·mask 차이 비율을 아래 표에 기록한다.

| VisA 범주 | 원본 크기 | 공식 Pixel AUROC | 공통 Pixel AUROC | 차이 | 다른 mask 픽셀 비율 |
|---|---|---:|---:|---:|---:|

## 실행 근거

- 코드: [`evaluate_anomalyclip_visa_all_categories_mask_geometry.py`](../source/evaluate_anomalyclip_visa_all_categories_mask_geometry.py)
- raw JSON: `method6/source/results/w40_anomalyclip_visa_all_categories_mask_transform_comparison.json`
