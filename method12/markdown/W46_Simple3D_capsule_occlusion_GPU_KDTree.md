# W46 Simple3D — 캡슐 부분 가림·점 수 대조 실험

- 기존 G: FPFH와 128개·12개 이웃 탐색을 GPU KD-tree로 실행함. 기존 모델 코드 수정 없음.
- 원본 정상 54개로 메모리뱅크 한 번 구성. 동일 테스트 130개에 5조건 적용함.
- 50% 가림은 25만 점, 75% 가림은 12만 5천 점을 남김. 낮은 X 쪽 연속 영역을 보존하는 평면 절단이며 실제 카메라 가림 시뮬레이션은 아님.
- 무작위 대조군도 같은 점 수를 남김. 마스크는 변형 위치 선정에 사용하지 않음. 원본 전처리 뒤 잘라 동일 좌표계 유지.
- 입력별 조건 실행 순서를 순환함. 모든 조건에서 동일 메모리뱅크, 원본 정답에서 살아남은 점의 라벨만 사용함.
- 물체 AUROC는 원래 정상·불량 라벨 사용. 결함이 완전히 가려진 개수 별도 기록. 같은 점 비교는 원본 점수를 해당 조건과 동일한 점에 제한한 값임.
- 평균 추론=특징 추출+점수 계산. 입력 읽기·가림 생성·진단·저장·지표 계산 제외. 세부 단계의 CUDA event와 wall time은 중첩되어 단순 합산할 수 없음.
- 뱅크 구성 146.97초, 전체 실행 23.13분.

| 조건 | 남은 점 | 물체 AUROC (%) | 포인트 AUROC (%) | AP (%) | 같은 점의 원본 AP (%) | AP 변화 (%p) | 결함 미관측 | 평균 추론 (ms) | p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 500000 | 91.25 | 80.75 | 22.77 | 22.77 | 0.00 | 0 | 3045.05 | 3284.85 |
| occluded_50 | 250000 | 73.25 | 77.68 | 20.01 | 17.60 | 2.41 | 62 | 1581.08 | 1718.74 |
| random_removed_50 | 250000 | 96.75 | 88.10 | 30.00 | 22.87 | 7.13 | 0 | 1572.68 | 1681.75 |
| occluded_75 | 125000 | 62.50 | 72.71 | 19.52 | 14.52 | 5.00 | 89 | 857.02 | 951.08 |
| random_removed_75 | 125000 | 90.08 | 93.17 | 36.10 | 22.82 | 13.27 | 0 | 849.05 | 905.09 |

## 동일 점 수에서 가림과 무작위 축소 비교

가림−무작위 값은 같은 개수의 점을 남긴 두 조건의 차이임. AP는 두 조건이 서로 다른 점을 남겨 결함 비중이 다를 수 있으므로, 같은 점의 원본 대비 AP 변화도 함께 확인함.

| 제거 비율 | 물체 AUROC 차이 (%p) | 포인트 AUROC 차이 (%p) | AP 차이 (%p) | 원본 대비 AP 변화 차이 (%p) | 추론 차이 (ms) |
|---|---:|---:|---:|---:|---:|
| 50% | -23.50 | -10.42 | -9.99 | -4.71 | 8.40 |
| 75% | -27.58 | -20.46 | -16.58 | -8.27 | 7.97 |

## 단계별 추론 시간

테스트 단계별 총 시간을 130샘플로 나눈 값(ms/샘플). 가림 생성·저장·평가·별도 이웃 진단 시간은 CSV에서 구분함.

| 단계 | full | occluded_50 | random_removed_50 | occluded_75 | random_removed_75 |
|---|---:|---:|---:|---:|---:|
| coordinate_knn_gpu_kdtree / CuPy exact GPU KDTree k=12 | 5.70 | 5.41 | 5.46 | 5.17 | 5.27 |
| coordinate_knn_gpu_kdtree / CuPy exact GPU KDTree k=128 | 36.15 | 33.61 | 34.57 | 33.24 | 33.40 |
| feature_concatenation / cuda:0 | 2.50 | 1.12 | 1.14 | 0.40 | 0.40 |
| feature_or_score_mean / cpu | 0.07 | 0.06 | 0.06 | 0.06 | 0.05 |
| feature_or_score_mean / cuda:0 | 0.36 | 0.38 | 0.37 | 0.37 | 0.37 |
| feature_scale_concatenation / gpu_wall | 2.64 | 1.28 | 1.29 | 0.59 | 0.56 |
| fpfh_gpu_kdtree_scale_120 / Open3D-derived Float32 SPFH+FPFH CUDA; k=120 | 23.32 | 11.92 | 11.85 | 6.00 | 6.05 |
| fpfh_gpu_kdtree_scale_40 / Open3D-derived Float32 SPFH+FPFH CUDA; k=40 | 6.30 | 3.33 | 3.35 | 1.70 | 1.71 |
| fpfh_gpu_kdtree_scale_80 / Open3D-derived Float32 SPFH+FPFH CUDA; k=80 | 14.14 | 7.35 | 7.25 | 3.71 | 3.78 |
| fps / cuda:0 | 1142.82 | 573.83 | 573.77 | 291.11 | 291.43 |
| fps_gather / cuda:0 | 0.02 | 0.03 | 0.02 | 0.02 | 0.02 |
| gpu_kdtree_all_points_k120 / gpu_wall | 1229.81 | 653.10 | 645.21 | 365.28 | 357.28 |
| gpu_kdtree_build / gpu_wall | 17.38 | 10.54 | 10.59 | 7.74 | 7.55 |
| image_top80 / cpu | 0.68 | 0.32 | 0.33 | 0.17 | 0.17 |
| memory_bank_cdist / cuda:0 | 3.12 | 1.66 | 1.65 | 1.65 | 1.64 |
| nearest_neighbor_min / cuda:0 | 0.37 | 0.21 | 0.21 | 0.21 | 0.21 |
| neighborhood_feature_gather / gpu_wall | 0.83 | 0.81 | 0.82 | 0.78 | 0.78 |
| normal_estimation_including_neighbor_search / cpu | 102.88 | 48.02 | 47.13 | 23.20 | 22.67 |
| point_score_interpolation / cuda:0 | 311.39 | 155.00 | 155.04 | 77.62 | 77.98 |
| positions_normals_upload / gpu_wall | 1.77 | 0.98 | 0.95 | 0.68 | 0.63 |
| scoremap_reorganization_cpu_gpu / cpu_gpu_wall | 0.99 | 0.69 | 0.65 | 0.49 | 0.48 |

## 같은 위치에서 점수와 이웃 범위가 바뀌었는지

표의 점수 차이는 변형 점수−원본 점수임. 각 샘플의 평균 차이를 구하고 샘플 간 중앙값을 취함. 결함 점수는 결함이 남은 샘플만 사용함. 가림 경계는 남은 점 중 절단면에 가장 가까운 X 상위 10%이며, 무작위 조건도 같은 방식의 X 상위 10%를 대조군으로 사용함. 이웃 거리는 각 조건에서 뽑은 동일한 점 256개의 원본·변형 비교임.

| 조건 | 결함 점수 변화 | 정상 경계 점수 변화 | 경계 밖 정상 점수 변화 | 40번째 이웃 거리 배수 | 80번째 | 120번째 |
|---|---:|---:|---:|---:|---:|---:|
| occluded_50 | 11.89 | 0.01 | 0.00 | 1.00 | 1.00 | 1.00 |
| random_removed_50 | 13.15 | 0.07 | 0.08 | 1.41 | 1.41 | 1.41 |
| occluded_75 | 36.90 | 0.04 | -0.00 | 1.00 | 1.00 | 1.00 |
| random_removed_75 | 29.88 | 0.72 | 0.51 | 1.98 | 1.99 | 1.99 |

## 자원 사용량

GPU 사용률·전력은 테스트 구간 표본 평균, 전체 VRAM·CPU RSS는 표본 최대. Torch 할당은 샘플별 최대치 중 최댓값이며 CuPy 할당 제외. 전체 VRAM은 다른 작업과 화면 출력 포함.

| 조건 | GPU 사용률 (%) | 전체 VRAM 최대 (MiB) | 전력 (W) | CPU RSS 최대 (GiB) | Torch 할당 최대 (GiB) |
|---|---:|---:|---:|---:|---:|
| full | 86.27 | 10359.00 | 133.36 | 2.99 | 4.77 |
| occluded_50 | 83.48 | 10355.00 | 133.25 | 2.99 | 4.77 |
| random_removed_50 | 84.09 | 10363.00 | 131.24 | 2.99 | 4.77 |
| occluded_75 | 80.74 | 10363.00 | 122.61 | 2.99 | 4.76 |
| random_removed_75 | 84.78 | 10363.00 | 124.52 | 2.99 | 4.76 |

## 원본과 원인 확인 근거

- 아래 기록에서 남은 결함 점, 같은 위치 점수 변화, 정상 절단 경계 주변의 점수 상승, 같은 위치에서 40·80·120번째 이웃 거리 변화를 함께 확인함. 한 카테고리·한 절단 방향 결과로 일반화하지 않음.
- [최종 지표](../source/results/capsule_occlusion_gpu_kdtree_20261107/metrics.csv), [샘플별 추론과 결함 보존](../source/results/capsule_occlusion_gpu_kdtree_20261107/test_timings.csv), [단계별 요약](../source/results/capsule_occlusion_gpu_kdtree_20261107/stage_summary.csv), [단계 호출 원본](../source/results/capsule_occlusion_gpu_kdtree_20261107/stage_timings.csv).
- [같은 점의 점수·이웃 거리 진단](../source/results/capsule_occlusion_gpu_kdtree_20261107/paired_diagnostics.csv), [자원 시계열](../source/results/capsule_occlusion_gpu_kdtree_20261107/resources.csv), [실행 검증](../source/results/capsule_occlusion_gpu_kdtree_20261107/progress.json).
- [실행 코드](../source/measure_simple3d_occlusion.py), [실행 스크립트](../source/run_simple3d_occlusion.sh).
