# W46 Simple3D — 캡슐 GPU KD-tree / FlashFPS / FastPoint 비교

**FastPoint가 가장 빨랐지만, 탐지 성능은 기존 FPS가 가장 높았음.** 평균 추론은 기존 대비 FlashFPS에서 **30.8%**, FastPoint에서 **34.4%** 감소했음.

## 비교 조건

- **G:** GPU KD-tree + 기존 PointNet2 CUDA FPS. CPU 공식 구현체와의 비교가 아님.
- **F:** 같은 GPU KD-tree에서 FPS 두 곳을 FlashFPS의 **FPS-Prune(75% 가지치기)**으로 교체함. FPS-Cache는 사용하지 않음.
- **P:** 같은 GPU KD-tree에서 FPS 두 곳을 **FastPoint MDPS**로 교체함. ScanNet 사전학습 거리 예측기와 공식 CUDA 샘플링 커널을 사용함. 기존 120-NN 거리·인덱스를 보관해 MDPS의 제외 목록에 재사용했으며, 선택점이 부족할 때는 남은 점을 기존 FPS 방식으로 채움. FastPoint 논문의 전체 이웃 탐색 최적화를 그대로 적용한 실험은 아님.
- 공통: RTX 5080, MiniShift `capsule` 원본 50만 점/샘플. 세 실행의 정상 54개·테스트 130개 목록 및 순서, Simple3D 설정·공식 호환 코드 해시가 일치함. 각각 정상 데이터에서 메모리뱅크를 새로 구성함.
- 특징 집계 중심점 4,096개·이웃 128개, 점수 평활화 중심점 1,024개·이웃 12개, FPFH 40·80·120을 유지함. 각 결과는 실행 시점이 다른 단일 측정이며, 다른 범주로 일반화할 수 없음.

## 전체 성능과 시간

추론 시간은 특징 추출과 점수 계산의 합이며, 데이터 읽기·결과 저장·평가지표 계산은 제외함. 차이는 각 방식에서 G를 뺀 값이며, % 지표의 차이는 %p임. GPU 전체 VRAM 표본은 다른 프로세스 사용량을 포함함.

| 항목 | G: 기존 FPS | F: FlashFPS | F − G | P: FastPoint | P − G |
|---|---:|---:|---:|---:|---:|
| 평균 추론 (ms/샘플) | 2779.21 | 1923.76 | −855.45 | 1823.13 | −956.08 |
| 중앙값 (ms/샘플) | 2755.90 | 1895.44 | −860.46 | 1804.79 | −951.11 |
| p95 추론 (ms/샘플) | 2894.77 | 2052.02 | −842.75 | 1937.26 | −957.52 |
| 메모리뱅크 구성 (초) | 128.78 | 91.67 | −37.11 | 89.36 | −39.41 |
| FPS 두 단계 합계 (ms/샘플) | 1027.45 | 72.06 | −955.39 | 70.17 | −957.28 |
| 물체 AUROC (%) | 91.25 | 62.83 | −28.42 | 80.92 | −10.33 |
| 포인트 AUROC (%) | 80.75 | 66.81 | −13.93 | 73.16 | −7.59 |
| 포인트 AP (%) | 22.77 | 4.38 | −18.40 | 12.19 | −10.58 |
| 최대 PyTorch 할당 (GiB) | 4.77 | 4.77 | 0.00 | 4.77 | 0.00 |
| 최대 PyTorch 예약 (GiB) | 5.97 | 5.97 | 0.00 | 6.97 | +1.00 |
| 테스트 중 GPU 전체 VRAM 표본 최대 (GiB) | 8.25 | 9.26 | +1.01 | 10.34 | +2.09 |
| 테스트 중 GPU 사용률 평균 (%) | 87.81 | 83.50 | −4.31 | 81.02 | −6.79 |

## 단계별 평균 추론 시간 (ms/샘플)

테스트 130개 기준. 일부 단계는 계측 범위가 겹치므로 표의 값을 더해 전체 추론 시간을 계산할 수 없음.

| 항목 | G: 기존 FPS | F: FlashFPS | F − G | P: FastPoint | P − G |
|---|---:|---:|---:|---:|---:|
| 법선 추정 및 이웃 탐색 (CPU) | 109.956 | 103.016 | −6.941 | 102.914 | −7.043 |
| 좌표·법선 GPU 전송 | 1.187 | 1.112 | −0.074 | 1.143 | −0.044 |
| GPU KD-tree 구성 | 15.605 | 16.907 | +1.302 | 16.203 | +0.598 |
| FPFH용 전체 점 120-NN 조회 | 1106.807 | 1189.015 | +82.207 | 1110.376 | +3.569 |
| FPFH 계산 k=40 | 5.735 | 6.176 | +0.442 | 5.945 | +0.211 |
| FPFH 계산 k=80 | 12.568 | 13.896 | +1.328 | 13.174 | +0.605 |
| FPFH 계산 k=120 | 20.632 | 22.865 | +2.233 | 21.751 | +1.119 |
| 특징 집계 FPS (4,096점) | 821.961 | 57.489 | −764.472 | 52.625 | −769.336 |
| 점수 평활화 FPS (1,024점) | 205.492 | 14.576 | −190.916 | 17.545 | −187.947 |
| 선택점 좌표 가져오기 | 0.015 | 0.148 | +0.133 | 0.155 | +0.140 |
| 집계용 128-NN 조회 | 31.902 | 31.361 | −0.541 | 33.190 | +1.287 |
| 평활화용 12-NN 조회 | 5.269 | 5.079 | −0.190 | 5.367 | +0.097 |
| 이웃 특징 가져오기 | 0.805 | 0.515 | −0.291 | 0.653 | −0.152 |
| 정상 메모리뱅크와 거리 계산 | 4.031 | 3.304 | −0.727 | 1.937 | −2.093 |
| 가장 가까운 특징 거리 선택 | 0.428 | 0.209 | −0.219 | 0.209 | −0.219 |
| 포인트 점수 보간 | 278.043 | 303.757 | +25.715 | 291.270 | +13.228 |
| 이미지 점수용 상위 80개 선택 | 0.612 | 0.568 | −0.044 | 0.738 | +0.125 |

## 난이도별 성능 (%)

각 난이도는 해당 결함 샘플과 정상 샘플을 사용함. 차이 단위는 %p임.

| 항목 | G: 기존 FPS | F: FlashFPS | F − G | P: FastPoint | P − G |
|---|---:|---:|---:|---:|---:|
| easy · 물체 AUROC | 99.50 | 58.75 | −40.75 | 90.75 | −8.75 |
| easy · 포인트 AUROC | 98.15 | 66.86 | −31.29 | 84.39 | −13.76 |
| easy · 포인트 AP | 41.23 | 7.84 | −33.39 | 20.46 | −20.77 |
| medium · 물체 AUROC | 90.50 | 66.25 | −24.25 | 81.75 | −8.75 |
| medium · 포인트 AUROC | 82.32 | 70.94 | −11.38 | 78.30 | −4.02 |
| medium · 포인트 AP | 21.69 | 4.94 | −16.75 | 12.55 | −9.14 |
| hard · 물체 AUROC | 83.75 | 63.50 | −20.25 | 70.25 | −13.50 |
| hard · 포인트 AUROC | 64.10 | 62.98 | −1.12 | 58.33 | −5.78 |
| hard · 포인트 AP | 7.06 | 1.81 | −5.26 | 3.97 | −3.09 |

## 해석

- FastPoint는 FlashFPS보다 평균 추론이 **100.63ms/샘플** 짧았음. 두 방식의 FPS 합계는 약 70ms로 비슷했으며, 관찰된 전체 시간 차이를 FPS만으로 설명할 수 없음. FPFH용 120-NN 조회가 두 변형에서도 가장 오래 걸렸음.
- 전체 기준 FastPoint는 FlashFPS보다 물체 AUROC가 **18.08%p**, 포인트 AUROC가 **6.34%p** 높았음. 다만 두 변형 모두 기존 FPS보다 낮았고, hard의 포인트 AUROC는 FlashFPS가 FastPoint보다 높았음.
- 두 FPS를 동시에 바꿨으므로, 특징 집계용 중심점과 평활화용 중심점 중 어느 쪽이 성능 하락에 더 크게 기여했는지는 구분할 수 없음.

## 원본 기록

- **G: GPU KD-tree:** [결과 JSON](../../method12/source/results/official_vs_gpu_kdtree_20261104/gpu_kdtree/minishift/capsule/summary.json) · [단계별 시간 CSV](../../method12/source/results/official_vs_gpu_kdtree_20261104/gpu_kdtree/minishift/capsule/stage_summary.csv) · [자원 사용량](../../method12/source/results/official_vs_gpu_kdtree_20261104/gpu_kdtree/minishift/capsule/resource_summary.json) · [실행 설정](../../method12/source/results/official_vs_gpu_kdtree_20261104/gpu_kdtree/minishift/capsule/environment.json)
- **F: FlashFPS:** [결과 JSON](../../method12/source/results/capsule_gpu_kdtree_flashfps_20261109/gpu_kdtree/minishift/capsule/summary.json) · [단계별 시간 CSV](../../method12/source/results/capsule_gpu_kdtree_flashfps_20261109/gpu_kdtree/minishift/capsule/stage_summary.csv) · [자원 사용량](../../method12/source/results/capsule_gpu_kdtree_flashfps_20261109/gpu_kdtree/minishift/capsule/resource_summary.json) · [실행 설정](../../method12/source/results/capsule_gpu_kdtree_flashfps_20261109/gpu_kdtree/minishift/capsule/environment.json) · [검증 기록](../../method12/source/results/capsule_gpu_kdtree_flashfps_20261109/verification.json)
- **P: FastPoint:** [결과 JSON](../../method12/source/results/capsule_gpu_kdtree_fastpoint_20261109/gpu_kdtree/minishift/capsule/summary.json) · [단계별 시간 CSV](../../method12/source/results/capsule_gpu_kdtree_fastpoint_20261109/gpu_kdtree/minishift/capsule/stage_summary.csv) · [자원 사용량](../../method12/source/results/capsule_gpu_kdtree_fastpoint_20261109/gpu_kdtree/minishift/capsule/resource_summary.json) · [실행 설정](../../method12/source/results/capsule_gpu_kdtree_fastpoint_20261109/gpu_kdtree/minishift/capsule/environment.json) · [검증 기록](../../method12/source/results/capsule_gpu_kdtree_fastpoint_20261109/verification.json)
- [FlashFPS 단독 비교표](W46_Simple3D_capsule_GPU_KDTree_FlashFPS_comparison.md) · [FlashFPS 적용 원본](https://github.com/Yuzhe-Fu/FlashFPS/tree/f32a4b7886501873f2d7d380dc84f07ae77492df) · [FastPoint 적용 원본](https://github.com/SNU-ARC/FastPoint/tree/7bebca723d5ba87891a9f221e3f0cf96962fab50)
