# W46 Simple3D — 캡슐 GPU KD-tree FPFH 20·40·60 비교

- 기존 전체 GPU KD-tree판 G 기준. **FPFH 이웃 스케일만 40·80·120 → 20·40·60**으로 변경함.
- 원본 점군 50만 점, 중심점 4,096개, 평활화 중심점 1,024개·이웃 12개, 특징 집계 이웃 16개 및 나머지 설정 유지함. 특징 연결은 20·40·40·60으로 132차원 유지함.
- 동일한 정상 54개에서 각 설정에 맞춰 뱅크를 구성함. 테스트 130개 동일. FPFH120은 이전 실행 결과라 실행 시점 차이가 있음.
- 차이는 FPFH60−FPFH120. 지표 차이는 %p, 시간은 원래 단위. 추론은 특징 추출과 점수 계산 합계이며 입출력·지표 계산 제외함.

| 항목 | 기존 FPFH120 | 새 FPFH60 | 차이 |
|---|---:|---:|---:|
| 메모리뱅크 구성 (s) | 131.77 | 98.88 | -32.88 |
| 평균 추론 (ms) | 2865.16 | 2308.44 | -556.72 |
| 중앙값 (ms) | 2854.85 | 2300.88 | -553.97 |
| p95 (ms) | 3067.05 | 2415.56 | -651.49 |
| 최대 Torch 할당 (GiB) | 4.77 | 4.77 | +0.00 |

## 성능

| 난이도 | 지표 | FPFH120 (%) | FPFH60 (%) | 차이 (%p) |
|---|---|---:|---:|---:|
| easy | 물체 AUROC | 97.75 | 98.75 | +1.00 |
| easy | 포인트 AUROC | 95.32 | 92.73 | -2.59 |
| easy | 포인트 AP | 38.77 | 30.66 | -8.11 |
| medium | 물체 AUROC | 89.50 | 94.25 | +4.75 |
| medium | 포인트 AUROC | 81.43 | 65.89 | -15.54 |
| medium | 포인트 AP | 18.86 | 12.90 | -5.97 |
| hard | 물체 AUROC | 88.25 | 91.00 | +2.75 |
| hard | 포인트 AUROC | 60.05 | 60.63 | +0.58 |
| hard | 포인트 AP | 6.87 | 5.27 | -1.60 |
| ALL | 물체 AUROC | 91.83 | 94.67 | +2.83 |
| ALL | 포인트 AUROC | 78.20 | 72.19 | -6.01 |
| ALL | 포인트 AP | 20.93 | 15.69 | -5.24 |

## 단계별 추론 시간

단계별 테스트 총 시간을 샘플 수로 나눈 ms/샘플. GPU event와 동기화 wall time은 계측 범위가 중첩될 수 있어 단순 합산하지 않음.

| 단계 | FPFH120 (ms) | FPFH60 (ms) | 차이 (ms) |
|---|---:|---:|---:|
| FPFH large | 22.129 | 10.704 | -11.426 |
| FPFH medium | 13.608 | 6.635 | -6.973 |
| FPFH small | 5.985 | 3.364 | -2.621 |
| FPFH 전체 점 이웃 조회 (120→60) | 1145.646 | 444.368 | -701.277 |
| coordinate_knn_gpu_kdtree / feature | 7.041 | 7.846 | +0.805 |
| coordinate_knn_gpu_kdtree / smooth | 5.482 | 6.137 | +0.655 |
| feature_concatenation | 2.405 | 2.666 | +0.261 |
| feature_or_score_mean | 0.117 | 0.114 | -0.003 |
| feature_scale_concatenation | 2.487 | 2.864 | +0.377 |
| fps | 1087.337 | 1215.226 | +127.890 |
| fps_gather | 0.015 | 0.015 | +0.000 |
| gpu_kdtree_build | 16.473 | 18.379 | +1.906 |
| image_top80 | 0.582 | 0.642 | +0.060 |
| input_loading | 2.132 | 2.180 | +0.048 |
| memory_bank_cdist | 3.458 | 3.506 | +0.048 |
| nearest_neighbor_min | 0.361 | 0.383 | +0.022 |
| neighborhood_feature_gather | 0.175 | 0.205 | +0.030 |
| normal_estimation_including_neighbor_search | 105.457 | 106.501 | +1.044 |
| point_score_interpolation | 296.491 | 331.854 | +35.364 |
| positions_normals_upload | 1.138 | 1.144 | +0.005 |
| scoremap_reorganization_cpu_gpu | 0.820 | 1.020 | +0.200 |

## 자원 사용량 (테스트 구간)

GPU 전체 VRAM은 다른 작업·화면 출력 포함. Torch 할당은 CuPy 메모리 제외.

| 항목 | FPFH120 | FPFH60 | 차이 |
|---|---:|---:|---:|
| GPU 사용률 평균 (%; 차이는 %p) | 87.86 | 86.07 | -1.79 |
| 전체 VRAM 최대 (MiB) | 9674.00 | 9641.00 | -33.00 |
| GPU 전력 평균 (W) | 135.42 | 118.84 | -16.58 |
| CPU RSS 최대 (GiB) | 2.94 | 2.94 | +0.00 |

## 원본 기록

- [FPFH60 결과 폴더](../source/results/capsule_gpu_kdtree_fpfh60_20261108/gpu_kdtree/minishift/capsule/)
- [기존 FPFH120 결과](../source/results/capsule_gpu_kdtree_k16_20261108/gpu_kdtree/minishift/capsule/summary.json)
- [실행 코드](../source/measure_simple3d_capsule_fpfh60.py), [실행 스크립트](../source/run_simple3d_capsule_fpfh60.sh).
