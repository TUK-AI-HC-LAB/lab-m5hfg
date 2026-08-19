# RD4AD MVTec AD 재현 결과

## 결과 요약

공식 RD4AD 구현체를 MVTec AD 15개 범주에서 두 번 실행함. 두 번째 실행은 논문 본문에 명시된 WideResNet-50, 256 x 256, batch 16, Adam learning rate 0.005, beta `(0.5, 0.999)`, 200 epoch, Gaussian smoothing sigma 4 조건을 명시한 별도 실행임. 아래 값은 두 번째 실행의 200 epoch 마지막 평가값임.

| 구분 | Image AUROC | Pixel AUROC |
|---|---:|---:|
| 논문 전체 평균 [1] | 98.50% | 97.80% |
| 논문 조건 재실행 전체 평균 | 98.69% | 97.79% |
| 재현 - 논문 | +0.19%p | -0.01%p |

- raw table: [`RD4AD_MVTecAD_WR50_paper_protocol_results.csv`](../source/results/RD4AD_MVTecAD_WR50_paper_protocol_results.csv)
- setup: [`RD4AD_reproduction_setup.md`](RD4AD_reproduction_setup.md)

## 범주별 결과

| 범주 | Image AUROC | Pixel AUROC | AU-PRO |
|---|---:|---:|---:|
| carpet | 98.60% | 99.00% | 97.00% |
| bottle | 100.00% | 98.70% | 96.60% |
| hazelnut | 100.00% | 98.90% | 95.50% |
| leather | 100.00% | 99.40% | 99.10% |
| cable | 96.00% | 97.20% | 90.90% |
| capsule | 97.30% | 98.70% | 95.80% |
| grid | 100.00% | 99.30% | 97.60% |
| pill | 96.40% | 98.10% | 96.60% |
| transistor | 96.20% | 92.30% | 78.30% |
| metal_nut | 100.00% | 97.30% | 92.40% |
| screw | 98.90% | 99.60% | 98.50% |
| toothbrush | 99.40% | 99.10% | 94.30% |
| zipper | 98.60% | 98.20% | 95.40% |
| tile | 99.70% | 95.50% | 90.40% |
| wood | 99.30% | 95.50% | 91.20% |
| 평균 | **98.69%** | **97.79%** | **93.97%** |

## 해석과 제한

- 논문 전체 평균과 비교하면 Image AUROC는 +0.19%p, Pixel AUROC는 -0.01%p 차이로 가까움.
- 첫 실행과 두 번째 논문 조건 재실행의 15개 값이 모두 같음. 같은 seed 111, 같은 데이터·설정·현재 환경에서 실행했으므로 이 환경에서는 결정론적으로 같은 결과가 재현됐음.
- `transistor`의 Pixel AUROC 92.30%와 AU-PRO 78.30%가 다른 범주보다 낮아, 위치 찾기 실패 사례를 우선 확인할 후보임.
- AU-PRO는 공식 코드가 계산한 보조 지표임. 논문 표의 핵심 평균 비교는 Image AUROC와 Pixel AUROC로만 함.
- 공식 구현체의 최종 200 epoch 평가값을 사용했으며, multi-seed 평균·저자와 같은 GPU·CUDA·라이브러리 버전까지의 bitwise 동일 재현은 아님.

## 참고문헌

[1] Deng, Hanqiu, and Xingyu Li. "Anomaly Detection via Reverse Distillation From One-Class Embedding." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2022, pp. 9737-9746.
