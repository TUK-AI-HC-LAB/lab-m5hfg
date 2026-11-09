# W46 Simple3D — 캡슐 GPU KD-tree에서 FPS-Prune 적용

캡슐에서 평균 추론 시간은 **30.8% 감소**했지만, 물체 AUROC는 **28.42%p**, 포인트 AUROC는 **13.93%p** 하락함. 이번 75% 가지치기 설정은 탐지 성능을 유지하는 대체안으로 보기 어려움.

- 기존 GPU KD-tree판의 PointNet2 CUDA FPS를 기준으로 비교함. **FPS만** [FlashFPS 공식 FPS-Prune](https://github.com/Yuzhe-Fu/FlashFPS/blob/f32a4b7886501873f2d7d380dc84f07ae77492df/FlashFPS-Openpoints/openpoints/models/layers/subsample.py)으로 교체함.
- 입력 앞 25%에 대해 일부 점을 FPS로 뽑고 나머지는 같은 후보 구간의 미선택 점으로 채움. 데이터 순서 유지. 내부 FPS는 기존 PointNet2 CUDA 사용. 정수 stride로 목표를 넘긴 경우 초과분만 잘라냄.
- Simple3D의 두 FPS는 같은 전체 점군에서 독립적으로 샘플링하므로 계층형 FPS-Cache는 적용하지 않음. 집계 이웃 128개 유지.
- 원본 점군 50만 점, 중심점 4,096개, 평활화 중심점 1,024개·이웃 12개, FPFH 40·80·120 및 나머지 설정 유지함.
- 동일한 정상 54개에서 각 설정에 맞춰 뱅크를 구성함. 테스트 130개 동일. 기존 FPS는 이전 실행 결과라 실행 시점 차이가 있음.
- 차이는 FPS-Prune−기존 FPS. 지표 차이는 %p, 시간은 원래 단위. 추론은 특징 추출과 점수 계산 합계이며 입출력·지표 계산 제외함.

| 항목 | 기존 GPU KD-tree FPS | FPS-Prune | 차이 |
|---|---:|---:|---:|
| 메모리뱅크 구성 (s) | 128.78 | 91.67 | -37.11 |
| 평균 추론 (ms) | 2779.21 | 1923.76 | -855.45 |
| 중앙값 (ms) | 2755.90 | 1895.44 | -860.46 |
| p95 (ms) | 2894.77 | 2052.02 | -842.75 |
| 최대 Torch 할당 (GiB) | 4.77 | 4.77 | +0.00 |

## 성능

| 난이도 | 지표 | 기존 GPU KD-tree FPS (%) | FPS-Prune (%) | 차이 (%p) |
|---|---|---:|---:|---:|
| easy | 물체 AUROC | 99.50 | 58.75 | -40.75 |
| easy | 포인트 AUROC | 98.15 | 66.86 | -31.29 |
| easy | 포인트 AP | 41.23 | 7.84 | -33.39 |
| medium | 물체 AUROC | 90.50 | 66.25 | -24.25 |
| medium | 포인트 AUROC | 82.32 | 70.94 | -11.38 |
| medium | 포인트 AP | 21.69 | 4.94 | -16.75 |
| hard | 물체 AUROC | 83.75 | 63.50 | -20.25 |
| hard | 포인트 AUROC | 64.10 | 62.98 | -1.12 |
| hard | 포인트 AP | 7.06 | 1.81 | -5.26 |
| ALL | 물체 AUROC | 91.25 | 62.83 | -28.42 |
| ALL | 포인트 AUROC | 80.75 | 66.81 | -13.93 |
| ALL | 포인트 AP | 22.77 | 4.38 | -18.40 |

## 단계별 추론 시간

단계별 테스트 총 시간을 샘플 수로 나눈 ms/샘플. GPU event와 동기화 wall time은 계측 범위가 중첩될 수 있어 단순 합산하지 않음.

| 단계 | 기존 GPU KD-tree FPS (ms) | FPS-Prune (ms) | 차이 (ms) |
|---|---:|---:|---:|
| coordinate_knn_gpu_kdtree / feature | 31.902 | 31.361 | -0.541 |
| coordinate_knn_gpu_kdtree / smooth | 5.269 | 5.079 | -0.190 |
| feature_concatenation | 2.288 | 2.506 | +0.218 |
| feature_or_score_mean | 0.448 | 0.417 | -0.031 |
| feature_scale_concatenation | 2.319 | 2.527 | +0.208 |
| fpfh_gpu_kdtree_scale_120 | 20.632 | 22.865 | +2.233 |
| fpfh_gpu_kdtree_scale_40 | 5.735 | 6.176 | +0.442 |
| fpfh_gpu_kdtree_scale_80 | 12.568 | 13.896 | +1.328 |
| fps | 1027.453 | 72.064 | -955.388 |
| ↳ 특징 집계 중심점 4,096개 | 821.961 | 57.489 | -764.472 |
| ↳ 점수 평활화 중심점 1,024개 | 205.492 | 14.576 | -190.916 |
| fps_gather | 0.015 | 0.148 | +0.133 |
| gpu_kdtree_all_points_k120 | 1106.807 | 1189.015 | +82.207 |
| gpu_kdtree_build | 15.605 | 16.907 | +1.302 |
| image_top80 | 0.612 | 0.568 | -0.044 |
| input_loading | 3.759 | 2.073 | -1.685 |
| memory_bank_cdist | 4.031 | 3.304 | -0.727 |
| nearest_neighbor_min | 0.428 | 0.209 | -0.219 |
| neighborhood_feature_gather | 0.805 | 0.515 | -0.291 |
| normal_estimation_including_neighbor_search | 109.956 | 103.016 | -6.941 |
| point_score_interpolation | 278.043 | 303.757 | +25.715 |
| positions_normals_upload | 1.187 | 1.112 | -0.074 |
| scoremap_reorganization_cpu_gpu | 0.786 | 0.786 | +0.000 |

두 FPS는 합계 약 14.3배 빨라졌지만, FPFH용 전체 점군 이웃 탐색이 1,189ms로 남아 전체 추론 시간은 같은 비율로 줄지 않음. 탐지 성능 하락의 가능한 원인은 FPS 후보를 입력 순서상 앞 25%로 제한하면서 중심점 분포가 달라진 것임. 이 실험은 특징 집계와 점수 평활화의 FPS를 함께 교체했으므로, 어느 쪽이 성능 하락에 더 크게 기여했는지는 분리할 수 없음.

## 자원 사용량 (테스트 구간)

GPU 전체 VRAM은 다른 작업·화면 출력 포함. Torch 할당은 CuPy 메모리 제외.

| 항목 | 기존 GPU KD-tree FPS | FPS-Prune | 차이 |
|---|---:|---:|---:|
| GPU 사용률 평균 (%; 차이는 %p) | 87.81 | 83.50 | -4.31 |
| 전체 VRAM 최대 (MiB) | 8448.00 | 9485.00 | +1037.00 |
| GPU 전력 평균 (W) | 138.16 | 165.41 | +27.25 |
| CPU RSS 최대 (GiB) | 2.95 | 3.18 | +0.23 |

## 원본 기록

- [FPS-Prune 결과 폴더](../../method12/source/results/capsule_gpu_kdtree_flashfps_20261109/gpu_kdtree/minishift/capsule/)
- [설정·데이터·호출 수 검증](../../method12/source/results/capsule_gpu_kdtree_flashfps_20261109/verification.json)
- [기존 FPS 결과](../../method12/source/results/official_vs_gpu_kdtree_20261104/gpu_kdtree/minishift/capsule/summary.json)
- [FlashFPS 어댑터](../../method12/source/simple3d_flashfps.py), [실행 코드](../../method12/source/measure_simple3d_capsule_flashfps.py), [실행 스크립트](../../method12/source/run_simple3d_capsule_flashfps.sh).
