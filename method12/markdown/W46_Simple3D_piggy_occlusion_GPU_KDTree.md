# W46 Simple3D — piggy 부분 가림·점 수 대조 실험

- 기존 G: FPFH와 128개·12개 이웃 탐색을 GPU KD-tree로 실행함. 기존 모델 코드 수정 없음.
- 원본 정상 100개로 메모리뱅크 한 번 구성. 동일 테스트 130개에 5조건 적용함.
- 50% 가림은 25만 점, 75% 가림은 12만 5천 점을 남김. 낮은 X 쪽 연속 영역을 보존하는 평면 절단이며 실제 카메라 가림 시뮬레이션은 아님.
- 무작위 대조군도 같은 점 수를 남김. 마스크는 변형 위치 선정에 사용하지 않음. 원본 전처리 뒤 잘라 동일 좌표계 유지.
- 입력별 조건 실행 순서를 순환함. 모든 조건에서 동일 메모리뱅크, 원본 정답에서 살아남은 점의 라벨만 사용함.
- 물체 AUROC는 원래 정상·불량 라벨 사용. 결함이 완전히 가려진 개수 별도 기록. 같은 점 비교는 원본 점수를 해당 조건과 동일한 점에 제한한 값임.
- 평균 추론=특징 추출+점수 계산. 입력 읽기·가림 생성·진단·저장·지표 계산 제외. 세부 단계의 CUDA event와 wall time은 중첩되어 단순 합산할 수 없음.
- 뱅크 구성 275.42초, 전체 실행 26.42분.

| 조건 | 남은 점 | 물체 AUROC (%) | 포인트 AUROC (%) | AP (%) | 같은 점의 원본 AP (%) | AP 변화 (%p) | 결함 미관측 | 평균 추론 (ms) | p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 500000 | 62.83 | 66.90 | 2.98 | 2.98 | 0.00 | 0 | 3208.23 | 3311.58 |
| occluded_50 | 250000 | 63.75 | 62.95 | 2.48 | 2.75 | -0.26 | 42 | 1679.26 | 1753.51 |
| random_removed_50 | 250000 | 59.42 | 70.33 | 1.42 | 2.94 | -1.52 | 0 | 1669.03 | 1728.32 |
| occluded_75 | 125000 | 46.33 | 61.12 | 3.33 | 2.29 | 1.05 | 76 | 930.52 | 983.68 |
| random_removed_75 | 125000 | 47.42 | 71.11 | 1.04 | 2.94 | -1.91 | 0 | 904.21 | 939.68 |

## 동일 점 수에서 가림과 무작위 축소 비교

가림−무작위 값은 같은 개수의 점을 남긴 두 조건의 차이임. AP는 두 조건이 서로 다른 점을 남겨 결함 비중이 다를 수 있으므로, 같은 점의 원본 대비 AP 변화도 함께 확인함.

| 제거 비율 | 물체 AUROC 차이 (%p) | 포인트 AUROC 차이 (%p) | AP 차이 (%p) | 원본 대비 AP 변화 차이 (%p) | 추론 차이 (ms) |
|---|---:|---:|---:|---:|---:|
| 50% | 4.33 | -7.38 | 1.06 | 1.25 | 10.23 |
| 75% | -1.08 | -9.99 | 2.30 | 2.95 | 26.32 |

## 단계별 추론 시간

테스트 단계별 총 시간을 130샘플로 나눈 값(ms/샘플). 가림 생성·저장·평가·별도 이웃 진단 시간은 CSV에서 구분함.

| 단계 | full | occluded_50 | random_removed_50 | occluded_75 | random_removed_75 |
|---|---:|---:|---:|---:|---:|
| coordinate_knn_gpu_kdtree / CuPy exact GPU KDTree k=12 | 5.95 | 5.98 | 6.02 | 6.15 | 5.66 |
| coordinate_knn_gpu_kdtree / CuPy exact GPU KDTree k=128 | 40.41 | 40.29 | 37.50 | 38.63 | 34.80 |
| feature_concatenation / cuda:0 | 2.56 | 1.11 | 1.12 | 0.41 | 0.42 |
| feature_or_score_mean / cpu | 0.07 | 0.06 | 0.09 | 0.05 | 0.05 |
| feature_or_score_mean / cuda:0 | 0.36 | 0.36 | 0.36 | 0.37 | 0.36 |
| feature_scale_concatenation / gpu_wall | 2.74 | 1.27 | 1.30 | 0.59 | 0.56 |
| fpfh_gpu_kdtree_scale_120 / Open3D-derived Float32 SPFH+FPFH CUDA; k=120 | 28.31 | 14.37 | 14.98 | 7.09 | 7.54 |
| fpfh_gpu_kdtree_scale_40 / Open3D-derived Float32 SPFH+FPFH CUDA; k=40 | 8.21 | 4.18 | 4.56 | 2.00 | 2.27 |
| fpfh_gpu_kdtree_scale_80 / Open3D-derived Float32 SPFH+FPFH CUDA; k=80 | 17.93 | 9.11 | 9.58 | 4.42 | 4.86 |
| fps / cuda:0 | 1182.83 | 595.27 | 595.14 | 301.37 | 302.33 |
| fps_gather / cuda:0 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| gpu_kdtree_all_points_k120 / gpu_wall | 1324.41 | 708.85 | 699.75 | 414.89 | 390.76 |
| gpu_kdtree_build / gpu_wall | 17.93 | 10.95 | 11.00 | 7.93 | 8.00 |
| image_top80 / cpu | 0.73 | 0.35 | 0.36 | 0.18 | 0.18 |
| memory_bank_cdist / cuda:0 | 4.84 | 3.14 | 3.12 | 3.12 | 3.13 |
| nearest_neighbor_min / cuda:0 | 0.57 | 0.40 | 0.39 | 0.39 | 0.41 |
| neighborhood_feature_gather / gpu_wall | 0.83 | 0.80 | 0.80 | 0.78 | 0.79 |
| normal_estimation_including_neighbor_search / cpu | 102.45 | 47.91 | 48.28 | 23.24 | 23.15 |
| point_score_interpolation / cuda:0 | 323.26 | 161.31 | 161.37 | 80.84 | 80.91 |
| positions_normals_upload / gpu_wall | 1.69 | 1.00 | 0.98 | 0.68 | 0.67 |
| scoremap_reorganization_cpu_gpu / cpu_gpu_wall | 1.01 | 0.73 | 0.72 | 0.55 | 0.54 |

## 같은 위치에서 점수와 이웃 범위가 바뀌었는지

표의 점수 차이는 변형 점수−원본 점수임. 각 샘플의 평균 차이를 구하고 샘플 간 중앙값을 취함. 결함 점수는 결함이 남은 샘플만 사용함. 가림 경계는 남은 점 중 절단면에 가장 가까운 X 상위 10%이며, 무작위 조건도 같은 방식의 X 상위 10%를 대조군으로 사용함. 이웃 거리는 각 조건에서 뽑은 동일한 점 256개의 원본·변형 비교임.

| 조건 | 결함 점수 변화 | 정상 경계 점수 변화 | 경계 밖 정상 점수 변화 | 40번째 이웃 거리 배수 | 80번째 | 120번째 |
|---|---:|---:|---:|---:|---:|---:|
| occluded_50 | 1.80 | 0.07 | 0.05 | 1.00 | 1.00 | 1.00 |
| random_removed_50 | 2.62 | 2.26 | 1.25 | 1.41 | 1.41 | 1.41 |
| occluded_75 | 2.16 | 0.44 | 0.03 | 1.00 | 1.00 | 1.00 |
| random_removed_75 | 4.26 | 5.13 | 3.74 | 1.98 | 1.99 | 1.99 |

## 자원 사용량

GPU 사용률·전력은 테스트 구간 표본 평균, 전체 VRAM·CPU RSS는 표본 최대. Torch 할당은 샘플별 최대치 중 최댓값이며 CuPy 할당 제외. 전체 VRAM은 다른 작업과 화면 출력 포함.

| 조건 | GPU 사용률 (%) | 전체 VRAM 최대 (MiB) | 전력 (W) | CPU RSS 최대 (GiB) | Torch 할당 최대 (GiB) |
|---|---:|---:|---:|---:|---:|
| full | 88.46 | 10769.00 | 132.54 | 3.01 | 4.91 |
| occluded_50 | 77.29 | 10747.00 | 132.26 | 3.01 | 4.91 |
| random_removed_50 | 88.36 | 10694.00 | 131.86 | 3.01 | 4.91 |
| occluded_75 | 83.32 | 10701.00 | 124.10 | 3.01 | 4.91 |
| random_removed_75 | 83.67 | 10694.00 | 124.86 | 3.00 | 4.91 |

## 원본과 원인 확인 근거

- 아래 기록에서 남은 결함 점, 같은 위치 점수 변화, 정상 절단 경계 주변의 점수 상승, 같은 위치에서 40·80·120번째 이웃 거리 변화를 함께 확인함. 한 카테고리·한 절단 방향 결과로 일반화하지 않음.
- [최종 지표](../source/results/piggy_occlusion_gpu_kdtree_20261107/metrics.csv), [샘플별 추론과 결함 보존](../source/results/piggy_occlusion_gpu_kdtree_20261107/test_timings.csv), [단계별 요약](../source/results/piggy_occlusion_gpu_kdtree_20261107/stage_summary.csv), [단계 호출 원본](../source/results/piggy_occlusion_gpu_kdtree_20261107/stage_timings.csv).
- [같은 점의 점수·이웃 거리 진단](../source/results/piggy_occlusion_gpu_kdtree_20261107/paired_diagnostics.csv), [자원 시계열](../source/results/piggy_occlusion_gpu_kdtree_20261107/resources.csv), [실행 검증](../source/results/piggy_occlusion_gpu_kdtree_20261107/progress.json).
- [실행 코드](../source/measure_simple3d_piggy_occlusion.py), [실행 스크립트](../source/run_simple3d_piggy_occlusion.sh).

## 캡슐과 piggy 결과 비교

같은 제거 규칙과 GPU KD-tree 구현을 사용했으나, 서로 다른 물체와 메모리뱅크임. 카테고리별 원본 대비 변화를 함께 봄.

| 조건 | 캡슐 물체 AUROC (%) | piggy 물체 AUROC (%) | 캡슐 원본 대비 (%p) | piggy 원본 대비 (%p) | 캡슐 추론 (ms) | piggy 추론 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| full | 91.25 | 62.83 | +0.00 | +0.00 | 3045.05 | 3208.23 |
| occluded_50 | 73.25 | 63.75 | -18.00 | +0.92 | 1581.08 | 1679.26 |
| random_removed_50 | 96.75 | 59.42 | +5.50 | -3.42 | 1572.68 | 1669.03 |
| occluded_75 | 62.50 | 46.33 | -28.75 | -16.50 | 857.02 | 930.52 |
| random_removed_75 | 90.08 | 47.42 | -1.17 | -15.42 | 849.05 | 904.21 |

- [캡슐 가림 실험 보고서](W46_Simple3D_capsule_occlusion_GPU_KDTree.md).
