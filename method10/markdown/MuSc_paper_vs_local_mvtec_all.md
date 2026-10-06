# MuSc MVTec AD 전체 category 실행 및 논문 비교

## 실제 결과와 판단

MVTec AD 15개 category, test 이미지 1,725장 평가를 완료했다. 정상 train은 사용하지 않고 category별 전체 unlabeled test pool로 scoring했다. 완료된 bottle은 기존 실행을 재사용했고 나머지 14개는 같은 설정으로 새로 실행했다. VisA와 ablation은 이번 결과에 포함하지 않는다.

## 전체 평균: 논문 Table 17과 비교

모든 값은 %. Δ = 내 실행 − 논문 공개값, 단위 %p. 각 category 지표의 동일 가중 평균(macro mean)이며 모든 픽셀을 한꺼번에 합친 micro 지표가 아니다. 논문은 소수점 한 자리만 공개하므로 Δ에는 논문 반올림 오차가 포함된다.

| 지표 | 논문 (%) | 내 실행 (%) | Δ (%p) |
|---|---:|---:|---:|
| Image AUROC | 97.8 | 97.7694 | -0.0306 |
| Image F1-max | 97.5 | 97.3219 | -0.1781 |
| Image AP | 99.1 | 99.0706 | -0.0294 |
| Pixel AUROC | 97.3 | 97.1149 | -0.1851 |
| Pixel F1-max | 62.6 | 62.1593 | -0.4407 |
| Pixel AP | 62.7 | 62.2587 | -0.4413 |
| AUPRO | 93.8 | 93.4755 | -0.3245 |

## Category별 실제 결과

지표 순서는 Image AUROC, Image F1-max, Image AP, Pixel AUROC, Pixel F1-max, Pixel AP, AUPRO. 모든 값은 %다.

| Category | 이미지 수 | I-AUROC | I-F1 | I-AP | P-AUROC | P-F1 | P-AP | AUPRO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bottle | 83 | 99.9206 | 99.2126 | 99.9752 | 98.4773 | 79.1673 | 83.0416 | 96.1027 |
| cable | 150 | 99.1004 | 97.2973 | 99.4938 | 95.7627 | 60.9783 | 57.6998 | 89.6186 |
| capsule | 132 | 96.6494 | 94.9309 | 99.3325 | 98.9564 | 49.7988 | 48.4481 | 95.5044 |
| carpet | 117 | 99.8395 | 98.8889 | 99.9499 | 99.4536 | 73.3367 | 76.0552 | 97.5819 |
| grid | 78 | 98.6633 | 96.4912 | 99.5372 | 98.1653 | 43.9301 | 38.2309 | 93.9317 |
| hazelnut | 110 | 99.6071 | 98.5507 | 99.7907 | 99.3755 | 73.4079 | 73.2828 | 92.2300 |
| leather | 124 | 100.0000 | 100.0000 | 100.0000 | 99.7170 | 62.8446 | 64.4696 | 98.7349 |
| metal_nut | 115 | 96.5787 | 97.3822 | 99.1445 | 86.1210 | 46.2250 | 47.5429 | 89.3400 |
| pill | 167 | 96.3448 | 95.8904 | 99.3325 | 97.4723 | 65.5317 | 67.2379 | 98.0108 |
| screw | 160 | 82.5579 | 88.4462 | 91.0461 | 98.7672 | 41.8867 | 36.1239 | 94.3990 |
| tile | 117 | 100.0000 | 100.0000 | 100.0000 | 97.8983 | 74.7124 | 78.9029 | 94.6450 |
| toothbrush | 42 | 100.0000 | 100.0000 | 100.0000 | 99.5346 | 70.1953 | 67.7893 | 95.4786 |
| transistor | 100 | 99.3750 | 95.2381 | 99.1022 | 91.3843 | 59.2512 | 58.4143 | 77.2170 |
| wood | 79 | 98.5088 | 98.3333 | 99.5248 | 97.2348 | 68.6367 | 74.7484 | 94.5024 |
| zipper | 151 | 99.3960 | 99.1667 | 99.8300 | 98.4039 | 62.4870 | 61.8920 | 94.8351 |

## Category별 논문 대비 차이

다음 표의 모든 값은 Δ(%p)다. 음수는 내 실행 값이 낮다는 의미다. 논문 Table 17의 같은 category 행과 비교했다.

| Category | I-AUROC | I-F1 | I-AP | P-AUROC | P-F1 | P-AP | AUPRO |
|---|---:|---:|---:|---:|---:|---:|---:|
| bottle | +0.0206 | +0.0126 | -0.0248 | -0.1227 | -0.4327 | -0.1584 | -0.0973 |
| cable | +0.1004 | -0.0027 | +0.0938 | -0.5373 | -1.3217 | -1.3002 | -1.1814 |
| capsule | -0.0506 | -0.3691 | +0.0325 | +0.0564 | +0.2988 | +0.3481 | +0.0044 |
| carpet | -0.0605 | -0.5111 | -0.0501 | -0.0464 | -0.0633 | +0.4552 | -0.0181 |
| grid | -0.0367 | -0.0088 | +0.0372 | -0.2347 | -0.6699 | -1.1691 | -0.7683 |
| hazelnut | +0.0071 | -0.0493 | -0.0093 | -0.0245 | -0.6921 | -0.5172 | -0.4700 |
| leather | +0.0000 | +0.0000 | +0.0000 | +0.0170 | -0.8554 | -0.9304 | +0.0349 |
| metal_nut | +0.2787 | -0.0178 | +0.0445 | +0.1210 | -0.0750 | +0.0429 | -0.1600 |
| pill | -0.0552 | -0.3096 | +0.0325 | -0.1277 | -0.9683 | -0.7621 | +0.0108 |
| screw | -0.9421 | -0.9538 | -0.2539 | -0.1328 | -1.0133 | -1.4761 | -0.4010 |
| tile | +0.0000 | +0.0000 | -0.0000 | -0.2017 | -0.5876 | +0.0029 | -0.2550 |
| toothbrush | +0.0000 | +0.0000 | -0.0000 | +0.0346 | -0.1047 | +0.2893 | -0.3214 |
| transistor | +0.2750 | +0.5381 | +0.3022 | -0.6157 | -0.3488 | -0.5857 | -1.0830 |
| wood | +0.0088 | +0.0333 | +0.0248 | -0.1652 | -0.3633 | -0.7516 | -0.2976 |
| zipper | -0.5040 | -0.4333 | -0.1700 | +0.0039 | +0.0870 | +0.3920 | +0.1351 |

공개 반올림 값 기준 가장 큰 절대 차이는 screw의 Pixel AP: -1.4761%p다. 차이의 원인은 코드 버전·환경·전처리·metric을 하나씩 고정해 추가 진단해야 하며 이 실행만으로 확정하지 않는다.

## 설정과 환경

- 공식 commit: `b76b93da8bd3096a99964a96ae29d46f197a0651`.
- RTX 5080 / WSL2 / Python 3.12.13 / PyTorch 2.11.0+cu128 / CUDA 12.8.
- OpenAI CLIP ViT-L/14-336, 입력 518×518, feature layers `[5,11,17,23]`, aggregation `[1,3,5]`.
- IA 낮은 30%, RsCIN 공식 MVTec `[1,2,3]`, batch 4, seed 42, TF32 off, 공식 autocast 유지.
- 모든 category에서 divide_num=1. 해상도 축소·pool 분할·scoring 알고리즘 변경 없음.
- 논문 당시 환경과 dependency 버전 및 출판 결과 생성 commit은 동일성 미확인. 공식 scoring/loader/metric 코드는 수정하지 않았다.
- AUPRO는 공식 200-threshold 및 FPR<0.3 선택 구간 min–max 정규화 구현을 사용한다. mask 변환 역시 공식 loader를 유지한다.

## 시간과 GPU 메모리

재개한 마지막 batch wall time: 543.14초 (9.05분). 이미 완료된 10개 category는 재사용하고 남은 5개를 실행한 시간이다. 중단 이전 대기 시간까지 포함한 최초 시작부터의 경과 시간은 아니다.
최종 채택된 15개 category의 성공한 실행 wall time 합계: 1798.99초 (29.98분). bottle을 포함하고 중단된 tile 시도는 제외한다. category별 시간에는 모델 로딩·raw 저장·metric·시각화가 포함되며, 각 실행 wrapper 바깥의 interpreter 시작 비용은 제외된다.
최초 batch는 tile의 지표 계산 도중 종료되어 2026-10-03 재개했고, tile부터 새로 실행했다. 중단된 대용량 출력과 최초 완료 상태는 repository에 포함하지 않았다.

| Category | 전체 실행 (초) | Peak allocated (GiB) | Peak reserved (GiB) |
|---|---:|---:|---:|
| bottle | 75.11 | 6.293 | 7.088 |
| cable | 164.15 | 6.640 | 10.752 |
| capsule | 130.45 | 6.552 | 9.064 |
| carpet | 111.74 | 6.466 | 9.195 |
| grid | 66.24 | 6.264 | 7.527 |
| hazelnut | 106.25 | 6.431 | 9.637 |
| leather | 122.18 | 6.510 | 9.525 |
| metal_nut | 121.53 | 6.461 | 9.271 |
| pill | 191.50 | 6.733 | 11.223 |
| screw | 193.27 | 6.699 | 10.213 |
| tile | 125.40 | 6.466 | 9.195 |
| toothbrush | 32.38 | 6.074 | 6.799 |
| transistor | 96.09 | 6.384 | 7.297 |
| wood | 70.10 | 6.271 | 7.539 |
| zipper | 192.62 | 6.649 | 9.992 |

Peak는 model load 후 reset한 PyTorch allocator 값이며 GPU 전체 사용량이 아니다. warm-up·반복·동일 측정 범위를 갖추지 않았으므로 논문 RTX 3090 대비 speedup을 주장하지 않는다.

## 검증 및 근거 경로

- 15개 category 완료 상태, 이미지별 score 행 수, 정상/이상 count, 지표 범위와 macro 평균을 자동 확인했다.
- commit: `b76b93da8bd3096a99964a96ae29d46f197a0651`
- sh: [run_mvtec_all.sh](../source/run_mvtec_all.sh)
- result: [category_metrics.csv](../source/results/mvtec_all_paper_20261002/category_metrics.csv), [mean_metrics.csv](../source/results/mvtec_all_paper_20261002/mean_metrics.csv)
- 논문 raw table: [paper_table17.csv](../source/results/mvtec_all_paper_20261002/paper_table17.csv)
- 모든 지표 비교: [paper_vs_local_all.csv](../source/results/mvtec_all_paper_20261002/paper_vs_local_all.csv)
- 데이터 점검: [dataset_audit.csv](../source/results/mvtec_all_paper_20261002/dataset_audit.csv)
- 실행 상태는 category metric과 batch summary에 반영했다. 개인 경로가 담긴 runtime status/log는 repository에 포함하지 않는다.
- 시간/메모리 raw: [execution_costs.csv](../source/results/mvtec_all_paper_20261002/execution_costs.csv)
- category별 CSV·image scores·config·environment: `../source/results/mvtec_all_paper_20261002/<category>/`.
- 대용량 NPZ·로그·heatmap 원본과 총괄 로그는 repository에 포함하지 않았다. 검토 가능한 요약 지표와 실행 환경은 위 CSV/JSON artifact로 남겼다.
- 논문 근거: [공식 PDF](../paper/ICLR24_MuSc_Zero_Shot_Industrial_Anomaly_Classification_and_Segmentation_with_Mutual_Scoring_of_the_Unlabeled_Images.pdf), p.20 Table 17 [1].

## 참고문헌

[1] Li, Xurui, et al. "MuSc: Zero-Shot Industrial Anomaly Classification and Segmentation with Mutual Scoring of the Unlabeled Images." International Conference on Learning Representations, 2024.
