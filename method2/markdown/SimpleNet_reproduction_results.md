# SimpleNet MVTec AD 재현 결과

## 결과 요약

논문 본문에 적힌 `resize 256 / crop 224`, batch size 4 조건으로 MVTec AD 15개 범주를 모두 실행함. 각 범주의 40개 meta epoch 중 image AUROC가 가장 높은 checkpoint를 공식 코드가 최종 결과로 저장함.

| 구분 | Image AUROC | Pixel AUROC |
|---|---:|---:|
| 논문 전체 평균 [1] | 99.60% | 98.10% |
| 이번 논문 조건 재현 | 99.35% | 97.33% |
| 재현 - 논문 | -0.25%p | -0.77%p |

- raw table: [`SimpleNet_MVTecAD_WR50_paper_protocol_results.csv`](../source/results/SimpleNet_MVTecAD_WR50_paper_protocol_results.csv)
- setup: [`SimpleNet_reproduction_setup.md`](SimpleNet_reproduction_setup.md)

## 범주별 결과

| 범주 | Image AUROC | Pixel AUROC | PRO-AUROC |
|---|---:|---:|---:|
| bottle | 100.00% | 97.76% | 92.21% |
| cable | 99.42% | 97.33% | 87.51% |
| capsule | 97.77% | 98.75% | 91.01% |
| carpet | 99.64% | 98.12% | 92.21% |
| grid | 99.67% | 96.69% | 89.83% |
| hazelnut | 100.00% | 98.01% | 89.30% |
| leather | 100.00% | 98.99% | 96.35% |
| metal_nut | 100.00% | 97.98% | 88.21% |
| pill | 98.64% | 98.40% | 93.18% |
| screw | 95.29% | 98.77% | 93.92% |
| tile | 99.82% | 94.56% | 86.67% |
| toothbrush | 100.00% | 98.54% | 90.38% |
| transistor | 100.00% | 96.80% | 89.21% |
| wood | 100.00% | 91.08% | 73.54% |
| zipper | 100.00% | 98.22% | 93.75% |
| 평균 | **99.35%** | **97.33%** | **89.82%** |

## 해석과 제한

- Image AUROC는 논문보다 0.25%p, Pixel AUROC는 0.77%p 낮음.
- 이 표는 이전 공식 코드 기본 입력값 `329 / 288`, batch 8 기준선이 아니라 논문 본문 `256 / 224`, batch 4 실행 결과임.
- 논문에 명시된 방법·전처리·학습 설정은 맞췄지만, 공개 코드와 현재 라이브러리 환경이 논문 작성 환경과 달라 bitwise 동일 재현은 주장하지 않음.
- PRO-AUROC는 공식 코드가 저장하는 보조 지표이며, 논문 표 1의 전체 평균 열과 직접 비교하지 않음.

## 참고문헌

[1] Liu, Zhikang, et al. "SimpleNet: A Simple Network for Image Anomaly Detection and Localization." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2023, pp. 20402-20411.
