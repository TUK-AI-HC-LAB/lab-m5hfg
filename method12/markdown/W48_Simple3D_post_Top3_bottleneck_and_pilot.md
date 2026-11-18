# W48 Simple3D — Top-3 이후 병목과 후속 실험

## 확인한 문제

52개 카테고리 비교 완료 결과의 선택 Top-3판만 집계함. 각 단계의 카테고리별 테스트 총 시간을 테스트 수로 나눈 뒤 카테고리 동일 가중 평균을 계산함.
FPFH용 120-NN이 두 데이터셋 모두 가장 큰 계측 구간임. 구조 생성과 조회를 분리해서 개선 후보를 검토함.
비중은 단계 평균/전체 추론 평균의 참고값임. 하위 구간이 상위 구간과 중첩되며 입력 읽기처럼 전체 추론 밖의 단계도 있으므로 모든 행을 합산하면 안 됨.

| 데이터셋 | 단계 | 구분 | 평균 (ms/샘플) | 전체 추론 대비 참고 비중 (%) |
|---|---|---|---:|---:|
| minishift | gpu_kdtree_all_points_k120 |  | 596.003 | 38.88 |
| minishift | step_query_k120 |  | 301.776 | 19.69 |
| minishift | step_cluster_index_build |  | 293.409 | 19.14 |
| minishift | fps | output_count=1024 | 215.449 | 14.06 |
| minishift | point_score_interpolation | 4096 centers | 170.166 | 11.10 |
| minishift | fps | output_count=4096 | 111.202 | 7.25 |
| minishift | normal_estimation_including_neighbor_search |  | 107.304 | 7.00 |
| minishift | image_top80 |  | 66.917 | 4.37 |
| minishift | point_score_interpolation | 1024 centers | 57.573 | 3.76 |
| minishift | coordinate_knn_gpu_kdtree | CuPy exact GPU KDTree k=128 | 47.113 | 3.07 |
| minishift | fpfh_gpu_kdtree_scale_120 |  | 24.259 | 1.58 |
| minishift | gpu_kdtree_build |  | 16.263 | 1.06 |
| minishift | fpfh_gpu_kdtree_scale_80 |  | 15.147 | 0.99 |
| minishift | fpfh_gpu_kdtree_scale_40 |  | 6.905 | 0.45 |
| minishift | coordinate_knn_gpu_kdtree | CuPy exact GPU KDTree k=12 | 6.384 | 0.42 |
| shapenet | gpu_kdtree_all_points_k120 |  | 118.889 | 44.14 |
| shapenet | step_cluster_index_build |  | 95.006 | 35.27 |
| shapenet | coordinate_knn_gpu_kdtree | CuPy exact GPU KDTree k=128 | 37.235 | 13.82 |
| shapenet | fps | output_count=1024 | 25.898 | 9.61 |
| shapenet | point_score_interpolation | 4096 centers | 24.318 | 9.03 |
| shapenet | step_query_k120 |  | 23.664 | 8.79 |
| shapenet | fps | output_count=4096 | 16.280 | 6.04 |
| shapenet | normal_estimation_including_neighbor_search |  | 12.805 | 4.75 |
| shapenet | image_top80 |  | 8.755 | 3.25 |
| shapenet | point_score_interpolation | 1024 centers | 6.648 | 2.47 |
| shapenet | gpu_kdtree_build |  | 6.544 | 2.43 |
| shapenet | coordinate_knn_gpu_kdtree | CuPy exact GPU KDTree k=12 | 4.956 | 1.84 |

## 원인 가설과 선택한 시험

- H1: STEP 계열 독립 구현에서 쿼리마다 512개 군집 하한을 완전 정렬하는 비용 중 일부가 불필요함.
- H2: 완전 정렬을 없애면 더 먼 군집을 일찍 방문해 후보 거리 계산이 늘어날 수 있어 오히려 느려질 수 있음.
- 선택: 가장 작은 하한의 군집을 먼저 방문한 뒤 나머지를 군집 번호순으로 검사하는 시험. 현재 120번째 거리보다 군집 하한이 큰 경우에만 건너뜀.
- 유지: STEP 512·Lloyd 50회·동일 인덱스·Float64 거리/하한·거리/원본 인덱스 동점 순서·FPFH 40/80/120·FlashFPS·집계/평활화 설정. 공식 STEP 코드 자체를 재현한 구현은 아님.
- 기대: 이웃 인덱스와 FPFH가 완전히 같고 실제 조회 시간이 감소하면 전체 파이프라인 시험으로 진행함. 조회가 느려지면 이 후보는 제외하고 이유를 남김.
- 독립 연산 시험은 전체 추론·AUROC/AP 결과가 아니며 속도 개선도 인덱스 생성 시간을 포함해 해석함.

## 재현 근거

- 기준 commit: `66956cf11839902944dffc5082210cb859f9b5f8` (새 파일은 아직 미커밋, 실행 소스 해시를 별도 기록).
- [분석 코드](../source/analyze_simple3d_post_top3.py) · [실행 스크립트](../source/run_simple3d_post_top3_pilot.sh)
- [전체 단계 집계 CSV](../source/results/post_top3_bottleneck_20261118/stage_ranking.csv)
- [52개 원본 결과](../source/results/all_selective_top3_20261118/stage_comparison.csv)
- [후속 연산 시험](../source/results/post_top3_bottleneck_20261118/pilot/)
- 새로움은 미확인. 실험 결과만으로 논문 기여라고 주장하지 않음.

## 실측 결과（연산 시험）

| 데이터셋/물체 | 샘플 | 기존 조회 (ms) | 시험 조회 (ms) | Δ (%) | 이웃 순서 불일치 |
|---|---:|---:|---:|---:|---:|
| minishift/capsule | 0 | 310.484 | 396.778 | +27.79 | 0 |
| minishift/capsule | 50 | 312.041 | 399.536 | +28.04 | 0 |
| minishift/capsule | 129 | 310.662 | 396.776 | +27.72 | 0 |
| shapenet/vase9 | 0 | 64.724 | 91.356 | +41.15 | 0 |
| shapenet/vase9 | 18 | 16.626 | 37.727 | +126.92 | 0 |
| shapenet/vase9 | 36 | 63.867 | 88.271 | +38.21 | 0 |

- 현재 판단: 모든 샘플에서 빨라지지 않아 일괄 대체는 보류함. 방문 순서 변경에 따른 추가 탐색 비용을 먼저 진단함.
- 이 표는 동일 인덱스를 공유한 조회 연산 시험이며 전체 추론 개선이나 AUROC/AP 유지를 직접 검증한 결과는 아님.

## 실패 원인 계측

별도 계측 커널로 쿼리당 평균 작업량을 측정함. 계측 오버헤드가 있으므로 이 실행의 시간은 비교하지 않음.

| 물체/샘플 | 방문 순서 | 방문 군집/쿼리 | 후보 거리 계산/쿼리 | 후보 병합/쿼리 |
|---|---|---:|---:|---:|
| capsule/0 | sorted | 2.267 | 2225.868 | 7.908 |
| capsule/0 | unsorted | 2.355 | 2312.356 | 8.065 |
| vase9/18 | sorted | 6.458 | 492.403 | 5.133 |
| vase9/18 | unsorted | 14.289 | 1082.985 | 12.553 |

- [작업량 원본 CSV](../source/results/post_top3_bottleneck_20261118/traversal_diagnosis/counters.csv)

- vase9/18에서는 후보 거리 계산이 약 2.20배, 후보 병합이 약 2.45배로 증가함. 가까운 군집부터 처리하지 않으면 임시 120번째 거리 기준이 늦게 좁혀져 더 많은 후보를 처리한다는 설명을 지지함.
- capsule/0에서는 후보 거리 계산 증가가 약 3.9%뿐임. 이것만으로 조회 시간 약 28% 증가를 전부 설명할 수 없음. 정렬판은 하한 초과 시 군집 순회를 종료하지만, 비정렬판은 남은 512개 군집을 끝까지 검사하는 제어 비용도 후보 원인임. 각 원인의 시간 기여율은 아직 분리 측정하지 않음.

## 두 번째 후보: 정렬 순서를 유지한 CUB Radix 교체

정렬 제거 실패를 바탕으로 512개 하한의 정렬만 NVIDIA CUB stable block radix sort로 교체함. 후보 병합 정렬과 탐색 순서·인덱스 구성은 유지함. 기존 라이브러리 교체이며 그 자체를 새로운 방법으로 주장하지 않음.

| 물체/샘플 | 기존 조회 (ms) | Radix 조회 (ms) | Δ (%) | 이웃 불일치 |
|---|---:|---:|---:|---:|
| capsule/0 | 301.980 | 251.229 | -16.81 | 0 |
| capsule/50 | 298.479 | 250.505 | -16.07 | 0 |
| capsule/129 | 299.544 | 251.220 | -16.13 | 0 |
| vase9/0 | 64.940 | 48.790 | -24.87 | 0 |
| vase9/18 | 15.894 | 12.326 | -22.44 | 0 |
| vase9/36 | 62.667 | 48.628 | -22.40 | 0 |

- 판단: 전체 파이프라인 소규모 비교로 진행할 후보임.
- 동일 인덱스의 조회 연산 시험이므로 아직 전체 추론·AUROC/AP 유지 결과는 아님.
- [원본 결과](../source/results/post_top3_bottleneck_20261118/radix_pilot/samples.csv)

## 현재 진행 단계와 확장 조건

- 완료: 52개 카테고리 병목 집계, 정렬 제거 후보 6개 샘플 시험, 2개 샘플 작업량 진단, Radix 후보 6개 샘플·4회 교차 순서 조회 시험.
- 완료: [capsule·vase9 전체 파이프라인의 새 뱅크·점수·AUROC/AP 비교](W48_Simple3D_STEP_radix_end_to_end.md). 전체 평균 추론은 capsule 5.00%, vase9 2.27% 감소했으며 뱅크·점별 점수·전체 및 난이도별 지표가 일치함. 각 조건 1회 결과이며 반복 측정의 안정성은 미검증. 연산 시험의 개선율을 전체 추론 개선율로 쓰지 않음.
- 다음 조건: 전체 파이프라인에서도 점수·뱅크·지표 일치와 시간 감소를 확인한 뒤에만 반복 측정과 데이터 규모 변화 시험을 검토함. 원래 점군의 점 수를 바꾸면 모델 출력도 바뀔 수 있으므로, 같은 입력에서 기준판과 시험판을 쌍으로 비교함.
- 미실시: 전체 52개 Radix 재실험, 다양한 GPU 반복 검증, 다른 탐지 모델 적용. 이 항목들을 완료한 것처럼 기록하지 않음.
- 논문 관점: 현재까지는 병목 원인과 구현 최적화 근거임. CUB 정렬 교체 자체는 기존 연산 재사용이므로 새로운 알고리즘이라고 주장하지 않음. 탐지 과제에 특화된 중복 탐색 제거 원리는 별도 설계·선행연구 확인이 필요함.
