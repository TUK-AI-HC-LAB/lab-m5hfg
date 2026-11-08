# W46 Simple3D — 캡슐 GPU KD-tree 집계 이웃 16개 비교

- 기존 전체 GPU KD-tree판 G 기준. **특징 집계 이웃만 128→16개**로 변경함.
- 원본 점군 50만 점, 중심점 4,096개, 평활화 중심점 1,024개·이웃 12개, FPFH 40·80·120 및 나머지 설정 유지함.
- 동일한 정상 54개에서 각 설정에 맞춰 뱅크를 구성함. 테스트 130개 동일. k128은 이전 실행 결과라 실행 시점 차이가 있음.
- 차이는 k16−k128. 지표 차이는 %p, 시간은 원래 단위. 추론은 특징 추출과 점수 계산 합계이며 입출력·지표 계산 제외함.

## 핵심 결과

- **집계용 이웃 탐색**은 31.90 → 7.04ms로 24.86ms 짧아졌음. 이웃 특징을 모으는 시간도 0.81 → 0.18ms로 줄었음.
- **전체 추론**은 이번 측정에서 2,779.21 → 2,865.16ms로 85.95ms(3.1%) 길었음. 변경하지 않은 FPS와 전체 점의 120개 이웃 탐색도 각각 59.88ms, 38.84ms 길게 측정됨. 두 결과의 실행 시점이 달라, 전체 추론 증가를 k16 설정 자체의 영향으로 단정할 수 없음.
- **탐지 성능(ALL):** 물체 AUROC는 +0.58%p, 포인트 AUROC는 −2.55%p, 포인트 AP는 −1.84%p였음. 이번 측정에서는 집계용 이웃 탐색이 빨라졌지만 전체 속도 개선은 확인되지 않았고, 위치 탐지 지표는 낮아졌음.

## 실행 시간·최대 메모리

| 항목 | 기존 k128 | 새 k16 | 차이 |
|---|---:|---:|---:|
| 메모리뱅크 구성 (s) | 128.78 | 131.77 | +2.99 |
| 평균 추론 (ms) | 2779.21 | 2865.16 | +85.95 |
| 중앙값 (ms) | 2755.90 | 2854.85 | +98.95 |
| p95 (ms) | 2894.77 | 3067.05 | +172.28 |
| 최대 Torch 할당 (GiB) | 4.77 | 4.77 | +0.00 |

## 성능

| 난이도 | 지표 | k128 (%) | k16 (%) | 차이 (%p) |
|---|---|---:|---:|---:|
| easy | 물체 AUROC | 99.50 | 97.75 | -1.75 |
| easy | 포인트 AUROC | 98.15 | 95.32 | -2.82 |
| easy | 포인트 AP | 41.23 | 38.77 | -2.47 |
| medium | 물체 AUROC | 90.50 | 89.50 | -1.00 |
| medium | 포인트 AUROC | 82.32 | 81.43 | -0.89 |
| medium | 포인트 AP | 21.69 | 18.86 | -2.83 |
| hard | 물체 AUROC | 83.75 | 88.25 | +4.50 |
| hard | 포인트 AUROC | 64.10 | 60.05 | -4.05 |
| hard | 포인트 AP | 7.06 | 6.87 | -0.19 |
| ALL | 물체 AUROC | 91.25 | 91.83 | +0.58 |
| ALL | 포인트 AUROC | 80.75 | 78.20 | -2.55 |
| ALL | 포인트 AP | 22.77 | 20.93 | -1.84 |

## 단계별 추론 시간

단계별 테스트 총 시간을 샘플 수로 나눈 ms/샘플. GPU event와 동기화 wall time은 계측 범위가 중첩될 수 있어 단순 합산하지 않음.

| 단계 | k128 (ms) | k16 (ms) | 차이 (ms) |
|---|---:|---:|---:|
| coordinate_knn_gpu_kdtree / feature | 31.902 | 7.041 | -24.861 |
| coordinate_knn_gpu_kdtree / smooth | 5.269 | 5.482 | +0.213 |
| feature_concatenation | 2.288 | 2.405 | +0.118 |
| feature_or_score_mean | 0.448 | 0.117 | -0.330 |
| feature_scale_concatenation | 2.319 | 2.487 | +0.168 |
| fpfh_gpu_kdtree_scale_120 | 20.632 | 22.129 | +1.497 |
| fpfh_gpu_kdtree_scale_40 | 5.735 | 5.985 | +0.250 |
| fpfh_gpu_kdtree_scale_80 | 12.568 | 13.608 | +1.040 |
| fps | 1027.453 | 1087.337 | +59.884 |
| fps_gather | 0.015 | 0.015 | +0.000 |
| gpu_kdtree_all_points_k120 | 1106.807 | 1145.646 | +38.838 |
| gpu_kdtree_build | 15.605 | 16.473 | +0.869 |
| image_top80 | 0.612 | 0.582 | -0.031 |
| input_loading | 3.759 | 2.132 | -1.627 |
| memory_bank_cdist | 4.031 | 3.458 | -0.572 |
| nearest_neighbor_min | 0.428 | 0.361 | -0.067 |
| neighborhood_feature_gather | 0.805 | 0.175 | -0.631 |
| normal_estimation_including_neighbor_search | 109.956 | 105.457 | -4.500 |
| point_score_interpolation | 278.043 | 296.491 | +18.448 |
| positions_normals_upload | 1.187 | 1.138 | -0.048 |
| scoremap_reorganization_cpu_gpu | 0.786 | 0.820 | +0.034 |

## 자원 사용량 (테스트 구간)

GPU 전체 VRAM은 다른 작업·화면 출력 포함. Torch 할당은 CuPy 메모리 제외.

| 항목 | k128 | k16 | 차이 |
|---|---:|---:|---:|
| GPU 사용률 평균 (%; 차이는 %p) | 87.81 | 87.86 | +0.05 |
| 전체 VRAM 최대 (MiB) | 8448.00 | 9674.00 | +1226.00 |
| GPU 전력 평균 (W) | 138.16 | 135.42 | -2.74 |
| CPU RSS 최대 (GiB) | 2.95 | 2.94 | -0.00 |

## 원본 기록

- [실행 검증](../source/results/capsule_gpu_kdtree_k16_20261108/verification.json): 동일 입력·다른 설정 유지, 특징 집계 k16 및 평활화 k12 테스트 호출 각 130회 확인.
- [k16 결과 폴더](../source/results/capsule_gpu_kdtree_k16_20261108/gpu_kdtree/minishift/capsule/)
- [기존 k128 결과](../source/results/official_vs_gpu_kdtree_20261104/gpu_kdtree/minishift/capsule/summary.json)
- [실행 코드](../source/measure_simple3d_capsule_k16.py), [실행 스크립트](../source/run_simple3d_capsule_k16.sh).
