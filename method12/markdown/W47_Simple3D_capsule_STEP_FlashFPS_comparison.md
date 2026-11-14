# W47 Simple3D 캡슐 — FlashFPS와 STEP을 함께 사용한 전체 추론 비교

MiniShift 캡슐, RTX 5080, 원본 50만 점/샘플. 같은 정상 54개에서 각 방식의 메모리뱅크를 새로 만들고, 같은 테스트 130개를 각각 평가함.
두 방식 모두 집계 대표점 4,096개는 같은 FlashFPS 혼합판(후보 25%, FPS 2,048개 + 채움 2,048개). FPFH 40·80·120, 집계 128-NN, 평활화의 기존 FPS 1,024개와 12-NN, 보간과 점수 규칙은 유지함.
STEP 기반 구현은 FPFH용 전체 120-NN만 대체함(512군집·Lloyd 50회, 동일 거리에서는 원본 인덱스 우선). 두 방식 모두 집계 및 평활화에 기존 GPU KD-tree를 쓰므로 STEP 쪽에도 KD-tree 구성 비용이 남음.
공식 STEP 코드를 재현한 것이 아니라 [논문의 Indexing STEP 원리](https://doi.org/10.1016/j.jpdc.2025.105187)를 가져온 별도 구현임. 앞선 [구간 파일럿](W47_Simple3D_capsule_STEP_120NN.md)에서 FPFH 40·80 값 일부가 달랐으므로, 비트 단위 일치를 가정하지 않음.

## 전체 시간과 탐지 성능

추론은 특징 추출과 점수 계산 합계로, 입력 대기·점수 저장·지표 계산은 제외함. 버전별 1회 연속 실행 결과임.

| 항목 | FlashFPS + KD-tree | FlashFPS + STEP | 차이 (STEP − 기존) |
|---|---:|---:|---:|
| 메모리뱅크 구성 (초) | 95.142 | 62.335 | -32.806 |
| 평균 추론 (ms) | 2,137.976 | 1,608.194 | -529.782 |
| 중앙값 추론 (ms) | 2,118.146 | 1,600.927 | -517.219 |
| p95 추론 (ms) | 2,329.477 | 1,712.673 | -616.804 |
| 실행 전체 경과 (초; 입출력·평가 포함) | 433.024 | 329.400 | -103.624 |
| 테스트 특징 추출 (ms) | 1,597.916 | 1,062.028 | -535.888 |
| 테스트 점수 계산 (ms) | 540.060 | 546.166 | +6.107 |

평균 추론 시간 변화: **-24.78%**.

| 구간 / 지표 | FlashFPS + KD-tree (%) | FlashFPS + STEP (%) | 차이 (%p) |
|---|---:|---:|---:|
| easy / 물체 AUROC | 100.000 | 100.000 | +0.000 |
| easy / 포인트 AUROC | 98.647 | 98.647 | -0.000 |
| easy / 포인트 AP | 39.736 | 39.736 | -0.000 |
| medium / 물체 AUROC | 95.000 | 95.000 | +0.000 |
| medium / 포인트 AUROC | 87.748 | 87.749 | +0.000 |
| medium / 포인트 AP | 21.974 | 21.975 | +0.001 |
| hard / 물체 AUROC | 84.500 | 84.500 | +0.000 |
| hard / 포인트 AUROC | 61.538 | 61.470 | -0.068 |
| hard / 포인트 AP | 6.860 | 6.858 | -0.002 |
| ALL / 물체 AUROC | 93.167 | 93.167 | +0.000 |
| ALL / 포인트 AUROC | 81.843 | 81.818 | -0.025 |
| ALL / 포인트 AP | 22.302 | 22.301 | -0.001 |

## 단계별 테스트 (ms/샘플)

두 번 호출되는 보간 등은 샘플당 합계로 표시함. 바깥 단계와 안쪽 단계의 계측이 중첩되므로 표 전체를 합산할 수 없음. STEP의 군집 생성·조회는 120-NN 전체 구간에 이미 포함됨.

| 단계 | FlashFPS + KD-tree | FlashFPS + STEP | 차이 |
|---|---:|---:|---:|
| 평활화용 12-NN | 5.556 | 5.602 | +0.046 |
| 집계용 128-NN | 33.102 | 33.559 | +0.457 |
| 텐서 연결 호출 합계 | 2.458 | 2.483 | +0.024 |
| 상위 80점 평균 | 0.028 | 0.026 | -0.001 |
| 평활화 점수 평균 | 0.046 | 0.044 | -0.001 |
| 집계 특징 평균 | 0.375 | 0.379 | +0.004 |
| 40·80·80·120 특징 연결 | 2.512 | 2.528 | +0.016 |
| FPFH 120 계산 | 22.161 | 22.565 | +0.404 |
| FPFH 40 계산 | 6.068 | 6.225 | +0.157 |
| FPFH 80 계산 | 13.636 | 13.795 | +0.159 |
| 평활화 대표점 1,024개 (기존 FPS) | 218.147 | 220.366 | +2.220 |
| 집계 대표점 4,096개 (FlashFPS) | 112.459 | 113.849 | +1.390 |
| FPS 좌표 가져오기 (1,024점) | 0.007 | 0.008 | +0.001 |
| FPS 좌표 가져오기 (4,096점) | 0.094 | 0.094 | -0.000 |
| FPFH 120-NN 전체 구간 (STEP 쪽은 군집 생성 포함) | 1,147.103 | 606.180 | -540.923 |
| GPU KD-tree 구축 (128/12 조회에도 필요) | 16.581 | 16.731 | +0.150 |
| 물체 점수용 상위 80점 선택 | 0.727 | 0.643 | -0.084 |
| 입력 대기 (추론 시간 제외) | 2.189 | 2.551 | +0.362 |
| 정상 메모리뱅크 거리 계산 | 3.408 | 2.371 | -1.037 |
| 가장 가까운 정상 특징 선택 | 0.217 | 0.212 | -0.004 |
| 집계 특징 텐서 가져오기 | 0.621 | 0.626 | +0.005 |
| CPU 법선 계산 (이웃 탐색 포함) | 105.570 | 107.821 | +2.252 |
| 평활화 뒤 보간 (1,024점 → 원본) | 58.070 | 59.257 | +1.188 |
| 1차 점수 보간 (4,096점 → 원본) | 239.425 | 243.605 | +4.180 |
| 좌표·법선 GPU 전송 | 1.168 | 1.178 | +0.010 |
| 점수 원본 점군 순서로 정리 | 0.951 | 0.864 | -0.087 |
| ↳ STEP 군집 인덱스 생성 | — | 300.523 | — |
| ↳ STEP 120-NN 탐색 | — | 304.795 | — |

## 단계별 정상 특징 추출 (ms/샘플)

두 번 호출되는 보간 등은 샘플당 합계로 표시함. 바깥 단계와 안쪽 단계의 계측이 중첩되므로 표 전체를 합산할 수 없음. STEP의 군집 생성·조회는 120-NN 전체 구간에 이미 포함됨.

| 단계 | FlashFPS + KD-tree | FlashFPS + STEP | 차이 |
|---|---:|---:|---:|
| 집계용 128-NN | 34.842 | 33.227 | -1.615 |
| 텐서 연결 호출 합계 | 2.446 | 2.340 | -0.106 |
| 집계 특징 평균 | 0.437 | 0.422 | -0.015 |
| 40·80·80·120 특징 연결 | 2.560 | 2.433 | -0.126 |
| FPFH 120 계산 | 22.810 | 21.851 | -0.960 |
| FPFH 40 계산 | 6.208 | 5.895 | -0.313 |
| FPFH 80 계산 | 13.914 | 13.546 | -0.368 |
| 집계 대표점 4,096개 (FlashFPS) | 115.436 | 111.482 | -3.954 |
| FPS 좌표 가져오기 (4,096점) | 0.105 | 0.091 | -0.014 |
| FPFH 120-NN 전체 구간 (STEP 쪽은 군집 생성 포함) | 1,180.096 | 585.203 | -594.893 |
| GPU KD-tree 구축 (128/12 조회에도 필요) | 17.372 | 16.721 | -0.651 |
| 집계 특징 텐서 가져오기 | 0.629 | 0.620 | -0.009 |
| CPU 법선 계산 (이웃 탐색 포함) | 106.459 | 103.155 | -3.303 |
| 좌표·법선 GPU 전송 | 1.456 | 1.347 | -0.110 |
| ↳ STEP 군집 인덱스 생성 | — | 287.927 | — |
| ↳ STEP 120-NN 탐색 | — | 296.428 | — |

## 코어셋·자원 사용량·저장량

GPU 표본은 약 0.5초 간격으로 수집한 장치 전체 값(화면·다른 작업 포함). PyTorch 최대치는 CuPy 할당을 포함하지 않음. 저장량은 점별 점수/정답과 뱅크 파일의 합으로, 데이터셋 및 환경은 제외함.

| 항목 | FlashFPS + KD-tree | FlashFPS + STEP | 차이 |
|---|---:|---:|---:|
| 코어셋 구성 (초) | 5.660 | 5.576 | -0.084 |
| PyTorch 최대 할당 (GiB) | 4.767 | 4.767 | +0.000 |
| PyTorch 최대 예약 (GiB) | 5.975 | 5.975 | +0.000 |
| fit / GPU 전체 VRAM 표본 최대 (MiB) | 4,462.000 | 3,638.000 | -824.000 |
| fit / GPU 사용률 평균 (%) | 79.348 | 71.481 | -7.867 |
| fit / GPU 메모리 컨트롤러 사용률 평균 (%) | 17.006 | 18.365 | +1.359 |
| fit / GPU 평균 전력 (W) | 155.663 | 165.308 | +9.645 |
| fit / GPU 최대 온도 (°C) | 64.000 | 59.000 | -5.000 |
| fit / CPU RSS 표본 최대 (GiB) | 2.569 | 2.920 | +0.351 |
| coreset / GPU 전체 VRAM 표본 최대 (MiB) | 4,464.000 | 3,638.000 | -826.000 |
| coreset / GPU 사용률 평균 (%) | 52.636 | 50.909 | -1.727 |
| coreset / GPU 메모리 컨트롤러 사용률 평균 (%) | 35.636 | 35.636 | +0.000 |
| coreset / GPU 평균 전력 (W) | 159.875 | 160.449 | +0.575 |
| coreset / GPU 최대 온도 (°C) | 55.000 | 53.000 | -2.000 |
| coreset / CPU RSS 표본 최대 (GiB) | 1.854 | 1.945 | +0.091 |
| test / GPU 전체 VRAM 표본 최대 (MiB) | 10,133.000 | 9,491.000 | -642.000 |
| test / GPU 사용률 평균 (%) | 86.716 | 77.199 | -9.517 |
| test / GPU 메모리 컨트롤러 사용률 평균 (%) | 24.082 | 27.327 | +3.246 |
| test / GPU 평균 전력 (W) | 154.312 | 162.027 | +7.715 |
| test / GPU 최대 온도 (°C) | 56.000 | 58.000 | +2.000 |
| test / CPU RSS 표본 최대 (GiB) | 3.255 | 3.494 | +0.239 |
| 메모리뱅크 파일 (MiB) | 5.570 | 5.570 | +0.000 |
| 점수·정답 저장 (MiB) | 195.904 | 195.903 | -0.001 |
| 합계 (MiB) | 201.474 | 201.473 | -0.001 |

## 점수 대조

- 130개 전부에서 점별 정답 배열이 일치함. 점수가 바뀐 점은 492,265개이며, 점 점수의 최대 절대 차이는 1.1348473임.
- 물체 점수의 최대 절대 차이는 0임.
- 동점 처리뿐 아니라 두 방식에서 각각 새로 구성한 메모리뱅크의 효과도 포함하므로, 차이의 원인을 이 표만으로 분리할 수 없음.

## 재현 자료

- kdtree: [요약](../source/results/capsule_step_flashfps_e2e_20261114/kdtree/gpu_kdtree/minishift/capsule/summary.json) · [평가지표](../source/results/capsule_step_flashfps_e2e_20261114/kdtree/gpu_kdtree/minishift/capsule/metrics.csv) · [샘플별 시간](../source/results/capsule_step_flashfps_e2e_20261114/kdtree/gpu_kdtree/minishift/capsule/test_timings.csv) · [단계별 시간](../source/results/capsule_step_flashfps_e2e_20261114/kdtree/gpu_kdtree/minishift/capsule/stage_timings.csv) · [자원 표본](../source/results/capsule_step_flashfps_e2e_20261114/kdtree/gpu_kdtree/minishift/capsule/resources.csv) · [자원 요약](../source/results/capsule_step_flashfps_e2e_20261114/kdtree/gpu_kdtree/minishift/capsule/resource_summary.csv)
- step: [요약](../source/results/capsule_step_flashfps_e2e_20261114/step/gpu_kdtree/minishift/capsule/summary.json) · [평가지표](../source/results/capsule_step_flashfps_e2e_20261114/step/gpu_kdtree/minishift/capsule/metrics.csv) · [샘플별 시간](../source/results/capsule_step_flashfps_e2e_20261114/step/gpu_kdtree/minishift/capsule/test_timings.csv) · [단계별 시간](../source/results/capsule_step_flashfps_e2e_20261114/step/gpu_kdtree/minishift/capsule/stage_timings.csv) · [자원 표본](../source/results/capsule_step_flashfps_e2e_20261114/step/gpu_kdtree/minishift/capsule/resources.csv) · [자원 요약](../source/results/capsule_step_flashfps_e2e_20261114/step/gpu_kdtree/minishift/capsule/resource_summary.csv)
- [샘플별 점수 차이](../source/results/capsule_step_flashfps_e2e_20261114/paired_score_differences.csv) · [검증 기록](../source/results/capsule_step_flashfps_e2e_20261114/verification.json) · [실행 코드](../source/run_simple3d_capsule_step_flashfps.py)
