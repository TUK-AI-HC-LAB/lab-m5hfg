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

12개 범주 평가를 완료했다. 모든 조건에서 Image AUROC와 anomaly map은 동일했으며, 아래 값은 mask 공간 변환만 바꿔 Pixel AUROC를 다시 계산한 결과다.

| VisA 범주 | 원본 크기 | 공식 Pixel AUROC | 공통 Pixel AUROC | 차이 | 다른 mask 픽셀 비율 |
|---|---|---:|---:|---:|---:|
| candle | 1284×1168 | 98.67% | 98.33% | −0.34%p | 0.14% |
| capsules | 1500×1000 | 93.41% | 65.95% | −27.46%p | 0.60% |
| cashew | 1274×1176 | 92.51% | 92.49% | −0.03%p | 0.17% |
| chewinggum | 1342×1118 | 99.15% | 99.02% | −0.13%p | 0.34% |
| fryum | 1500×1000 | 92.68% | 91.39% | −1.30%p | 2.12% |
| macaroni1 | 1500×1000 | 97.99% | 87.52% | −10.48%p | 0.09% |
| macaroni2 | 1500×1000 | 97.37% | 73.46% | −23.91%p | 0.06% |
| pcb1 | 1404×1070 | 89.54% | 87.49% | −2.05%p | 0.38% |
| pcb2 | 1404×1070 | 90.38% | 87.16% | −3.23%p | 0.31% |
| pcb3 | 1562×960 | 88.01% | 84.48% | −3.53%p | 0.50% |
| pcb4 | 1358×1104 | 94.50% | 93.72% | −0.78%p | 0.70% |
| pipe_fryum | 1300×1154 | 97.82% | 97.69% | −0.13%p | 0.26% |
| **12개 단순 평균** | — | **94.34%** | **88.22%** | **−6.11%p** | **0.39%** |

### 해석

`capsules`, `macaroni1`, `macaroni2`처럼 같은 종횡비의 원본이라도 결함 위치·모양에 따라 mask 좌표 오차가 Pixel AUROC에 크게 반영될 수 있었다. 서로 다른 mask 픽셀의 비율이 작아도, 그 픽셀이 결함 경계 또는 고점 anomaly map 주변에 놓이면 순위 기반 Pixel AUROC가 크게 달라질 수 있다.

따라서 이 비교는 “공통 정사각형 mask가 모델 성능을 낮춘다”는 뜻이 아니다. **같은 model output을 잘못된 좌표의 정답과 비교하면 평가값이 달라진다**는 검증이다. AnomalyCLIP의 image transform이 shortest-edge resize와 center crop이면, 정답 mask도 동일한 순서로 변환해야 한다.

## 실행 근거

- 코드: [`evaluate_anomalyclip_visa_all_categories_mask_geometry.py`](../source/evaluate_anomalyclip_visa_all_categories_mask_geometry.py)
- raw JSON: `method6/source/results/w40_anomalyclip_visa_all_categories_mask_transform_comparison.json`
- 공통 코드베이스 설명: [`related_work/markdown/W40_공통_전처리_후처리_분석.md`](../../related_work/markdown/W40_공통_전처리_후처리_분석.md)
