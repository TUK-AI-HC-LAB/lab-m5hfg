# W46 Simple3D — 캡슐 GPU KD-tree FPFH 60·100·140 비교

- 기존 전체 GPU KD-tree판 G 기준. **FPFH 이웃 스케일만 20·40·60 → 60·100·140**으로 변경함.
- 원본 점군 50만 점, 중심점 4,096개, 평활화 중심점 1,024개·이웃 12개, 특징 집계 이웃 16개 및 나머지 설정 유지함. 특징 연결은 60·100·100·140으로 132차원 유지함.
- 동일한 정상 54개에서 각 설정에 맞춰 뱅크를 구성함. 테스트 130개 동일. FPFH60은 이전 실행 결과라 실행 시점 차이가 있음.
- 차이는 FPFH140−FPFH60. 지표 차이는 %p, 시간은 원래 단위. 추론은 특징 추출과 점수 계산 합계이며 입출력·지표 계산 제외함.

| 항목 | 기존 FPFH60 | 새 FPFH140 | 차이 |
|---|---:|---:|---:|
| 메모리뱅크 구성 (s) | 98.88 | 154.52 | +55.64 |
| 평균 추론 (ms) | 2308.44 | 3286.39 | +977.95 |
| 중앙값 (ms) | 2300.88 | 3280.14 | +979.27 |
| p95 (ms) | 2415.56 | 3496.26 | +1080.70 |
| 최대 Torch 할당 (GiB) | 4.77 | 4.77 | +0.00 |

## 성능

| 난이도 | 지표 | FPFH60 (%) | FPFH140 (%) | 차이 (%p) |
|---|---|---:|---:|---:|
| easy | 물체 AUROC | 98.75 | 97.00 | -1.75 |
| easy | 포인트 AUROC | 92.73 | 97.58 | +4.85 |
| easy | 포인트 AP | 30.66 | 41.02 | +10.37 |
| medium | 물체 AUROC | 94.25 | 88.50 | -5.75 |
| medium | 포인트 AUROC | 65.89 | 82.71 | +16.81 |
| medium | 포인트 AP | 12.90 | 20.94 | +8.04 |
| hard | 물체 AUROC | 91.00 | 85.25 | -5.75 |
| hard | 포인트 AUROC | 60.63 | 65.70 | +5.07 |
| hard | 포인트 AP | 5.27 | 7.69 | +2.43 |
| ALL | 물체 AUROC | 94.67 | 90.25 | -4.42 |
| ALL | 포인트 AUROC | 72.19 | 81.19 | +9.01 |
| ALL | 포인트 AP | 15.69 | 22.72 | +7.04 |

## 단계별 추론 시간

단계별 테스트 총 시간을 샘플 수로 나눈 ms/샘플. GPU event와 동기화 wall time은 계측 범위가 중첩될 수 있어 단순 합산하지 않음.

| 단계 | FPFH60 (ms) | FPFH140 (ms) | 차이 (ms) |
|---|---:|---:|---:|
| FPFH large | 10.704 | 26.137 | +15.433 |
| FPFH medium | 6.635 | 17.180 | +10.546 |
| FPFH small | 3.364 | 9.407 | +6.043 |
| FPFH 전체 점 이웃 조회 (60→140) | 444.368 | 1583.641 | +1139.273 |
| coordinate_knn_gpu_kdtree / feature | 7.846 | 6.954 | -0.892 |
| coordinate_knn_gpu_kdtree / smooth | 6.137 | 5.430 | -0.707 |
| feature_concatenation | 2.666 | 2.370 | -0.297 |
| feature_or_score_mean | 0.114 | 0.115 | +0.001 |
| feature_scale_concatenation | 2.864 | 2.437 | -0.427 |
| fps | 1215.226 | 1065.522 | -149.705 |
| fps_gather | 0.015 | 0.015 | -0.000 |
| gpu_kdtree_build | 18.379 | 16.076 | -2.303 |
| image_top80 | 0.642 | 0.643 | +0.001 |
| input_loading | 2.180 | 2.422 | +0.243 |
| memory_bank_cdist | 3.506 | 3.403 | -0.103 |
| nearest_neighbor_min | 0.383 | 0.382 | -0.002 |
| neighborhood_feature_gather | 0.205 | 0.166 | -0.038 |
| normal_estimation_including_neighbor_search | 106.501 | 106.056 | -0.444 |
| point_score_interpolation | 331.854 | 289.805 | -42.050 |
| positions_normals_upload | 1.144 | 1.121 | -0.023 |
| scoremap_reorganization_cpu_gpu | 1.020 | 0.797 | -0.224 |

## 자원 사용량 (테스트 구간)

GPU 전체 VRAM은 다른 작업·화면 출력 포함. Torch 할당은 CuPy 메모리 제외.

| 항목 | FPFH60 | FPFH140 | 차이 |
|---|---:|---:|---:|
| GPU 사용률 평균 (%; 차이는 %p) | 86.07 | 89.51 | +3.43 |
| 전체 VRAM 최대 (MiB) | 9641.00 | 9814.00 | +173.00 |
| GPU 전력 평균 (W) | 118.84 | 141.56 | +22.72 |
| CPU RSS 최대 (GiB) | 2.94 | 2.94 | -0.00 |

## 원본 기록

- [FPFH140 결과 폴더](../source/results/capsule_gpu_kdtree_fpfh60_100_140_20261108/gpu_kdtree/minishift/capsule/)
- [기존 FPFH60 결과](../source/results/capsule_gpu_kdtree_fpfh60_20261108/gpu_kdtree/minishift/capsule/summary.json)
- [실행 코드](../source/measure_simple3d_capsule_fpfh60_100_140.py), [실행 스크립트](../source/run_simple3d_capsule_fpfh60_100_140.sh).
