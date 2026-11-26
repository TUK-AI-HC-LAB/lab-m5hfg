# W46 Simple3D 캡슐 — FlashFPS 후보 선택 방식 비교

FlashFPS에서 후보 수·반복 수는 유지하고, 후보를 입력의 앞 25%에서 전체에 걸쳐 무작위 추출한 25%로 바꾸자 물체 AUROC가 **62.83% → 87.42%**, 포인트 AUROC가 **66.81% → 78.63%**로 회복됐음. 평균 추론은 기존 GPU KD-tree의 원래 FPS보다 **29.66% 단축**됐으나 전체 탐지 지표는 아직 기존에 못 미침.

## 실험 조건

- RTX 5080, MiniShift `capsule` 50만 점/샘플, 정상 54개로 각 메모리뱅크를 새로 구성하고 동일 테스트 130개 평가.
- 기존 FPS는 GPU KD-tree 버전의 PointNet2 CUDA FPS. 두 FlashFPS 변형은 모두 75% 가지치기, FPS-Cache 미사용, 중심점 4,096개·1,024개, 집계·평활화 이웃 128개·12개, FPFH 40·80·120 유지.
- 새 변형의 후보는 전체 인덱스에서 비복원 무작위 추출한 25%. 시드 42, 동일 점 개수에서는 같은 후보 인덱스 집합을 재사용함. 후보 인덱스를 원래 상대 순서대로 정렬하여 FPS 초기점과 미선택 후보로 채우는 기존 규칙을 유지함.
- 이전 두 실행과 실행 시점이 다름. 시간 차이의 전부를 후보 선택 변경에 귀속하지 않음.

## 전체 결과

시간 차이는 표에 표시한 단위, 지표 차이는 %p. 추론 시간은 특징 추출과 점수 계산의 합으로, 입력 읽기·결과 저장·AUROC 계산은 제외.

| 항목 | 기존 GPU KD-tree FPS | 앞 25% | 무작위 25% | Δ 무작위−앞 | Δ 무작위−기존 |
|---|---:|---:|---:|---:|---:|
| 물체 AUROC (%) | 91.25 | 62.83 | 87.42 | +24.58 | −3.83 |
| 포인트 AUROC (%) | 80.75 | 66.81 | 78.63 | +11.82 | −2.11 |
| 포인트 AP (%) | 22.77 | 4.38 | 16.00 | +11.63 | −6.77 |
| 평균 추론 (ms/샘플) | 2779.21 | 1923.76 | 1954.92 | +31.16 | −824.28 |
| 추론 p95 (ms/샘플) | 2894.77 | 2052.02 | 2107.91 | +55.89 | −786.86 |
| 메모리뱅크 구성 (초) | 128.78 | 91.67 | 90.00 | −1.67 | −38.78 |
| 두 FPS 합계 (ms/샘플) | 1027.45 | 72.06 | 73.52 | +1.46 | −953.93 |
| 특징 집계 FPS · 4,096점 (ms/샘플) | 821.96 | 57.49 | 58.52 | +1.04 | −763.44 |
| 점수 평활화 FPS · 1,024점 (ms/샘플) | 205.49 | 14.58 | 15.00 | +0.42 | −190.50 |
| FPFH용 전체 점 120-NN (ms/샘플) | 1106.81 | 1189.01 | 1207.72 | +18.70 | +100.91 |
| 최대 PyTorch 할당 (GiB) | 4.77 | 4.77 | 4.77 | +0.00 | +0.00 |
| 테스트 중 GPU 전체 VRAM 표본 최대 (MiB) | 8448.00 | 9485.00 | 9892.00 | +407.00 | +1444.00 |

GPU 전체 VRAM은 다른 프로세스가 쓰는 메모리도 포함함. 단계별 전체 기록은 아래 CSV에 연결함.

## 난이도별 결과 (%)

| 항목 | 기존 GPU KD-tree FPS | 앞 25% | 무작위 25% | Δ 무작위−앞 | Δ 무작위−기존 |
|---|---:|---:|---:|---:|---:|
| easy · 물체 AUROC | 99.50 | 58.75 | 94.75 | +36.00 | −4.75 |
| easy · 포인트 AUROC | 98.15 | 66.86 | 98.36 | +31.50 | +0.21 |
| easy · 포인트 AP | 41.23 | 7.84 | 30.58 | +22.73 | −10.66 |
| medium · 물체 AUROC | 90.50 | 66.25 | 89.50 | +23.25 | −1.00 |
| medium · 포인트 AUROC | 82.32 | 70.94 | 79.27 | +8.33 | −3.06 |
| medium · 포인트 AP | 21.69 | 4.94 | 16.51 | +11.57 | −5.18 |
| hard · 물체 AUROC | 83.75 | 63.50 | 78.00 | +14.50 | −5.75 |
| hard · 포인트 AUROC | 64.10 | 62.98 | 61.50 | −1.48 | −2.60 |
| hard · 포인트 AP | 7.06 | 1.81 | 3.91 | +2.11 | −3.15 |

## 해석

- 정상 10개와 불량 120개의 1,200쌍에서 불량이 정상보다 낮은 점수를 받은 경우는 기존 FPS **105쌍**, 앞 25% **446쌍**, 무작위 25% **151쌍**이었음. 동점은 없었음.
- 두 FlashFPS 변형의 후보 수와 반복 횟수가 같고 후보를 뽑는 위치만 바꿨는데 지표가 회복됨. 캡슐에서는 기존 성능 하락의 상당 부분이 입력 순서에 치우친 후보 선택과 관련 있다는 근거임.
- 원래 FPS보다 물체 AUROC **3.83%p**, 포인트 AUROC **2.11%p**, 포인트 AP **6.77%p** 낮음. 남은 격차가 후보 축소와 반복 축소 중 어느 요인 때문인지는 분리하지 않았음. 단일 범주·단일 시드 결과임.

## 근거 자료

- **기존 FPS:** [요약](../source/results/official_vs_gpu_kdtree_20261104/gpu_kdtree/minishift/capsule/summary.json) · [단계별 시간](../source/results/official_vs_gpu_kdtree_20261104/gpu_kdtree/minishift/capsule/stage_summary.csv) · [샘플별 점수](../source/results/official_vs_gpu_kdtree_20261104/gpu_kdtree/minishift/capsule/test_timings.csv)
- **앞 25%:** [요약](../source/results/capsule_gpu_kdtree_flashfps_20261109/gpu_kdtree/minishift/capsule/summary.json) · [단계별 시간](../source/results/capsule_gpu_kdtree_flashfps_20261109/gpu_kdtree/minishift/capsule/stage_summary.csv) · [샘플별 점수](../source/results/capsule_gpu_kdtree_flashfps_20261109/gpu_kdtree/minishift/capsule/test_timings.csv)
- **무작위 25%:** [요약](../source/results/capsule_gpu_kdtree_flashfps_random25_20261111/gpu_kdtree/minishift/capsule/summary.json) · [단계별 시간](../source/results/capsule_gpu_kdtree_flashfps_random25_20261111/gpu_kdtree/minishift/capsule/stage_summary.csv) · [샘플별 점수](../source/results/capsule_gpu_kdtree_flashfps_random25_20261111/gpu_kdtree/minishift/capsule/test_timings.csv)
- [검증 기록](../source/results/capsule_gpu_kdtree_flashfps_random25_20261111/verification.json) · [후보 선택 코드](../source/simple3d_flashfps_random25.py) · [실행 스크립트](../source/run_simple3d_capsule_flashfps_random25.sh)
