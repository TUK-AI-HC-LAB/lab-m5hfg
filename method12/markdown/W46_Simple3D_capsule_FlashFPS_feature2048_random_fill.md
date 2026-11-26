# W46 Simple3D 캡슐 — 특징 집계 중심점 무작위 채우기

FPS로 직접 고른 2,048개를 그대로 두고 나머지 2,048개만 무작위로 채웠음. 전체 물체 AUROC **+0.08%p**, 전체 포인트 AUROC **−0.25%p**로 두 AUROC를 동시에 높이지는 못했음. Hard 포인트 AUROC는 **+2.39%p**였으나 medium은 **−3.40%p**였음.

## 조건

- 기준은 CPU 공식 구현체가 아니라 **GPU KD-tree 버전의 원래 PointNet2 CUDA FPS**임. 기존은 전체 점군에서 특징 집계 중심점 4,096개·평활화 중심점 1,024개를 각각 FPS로 직접 선택함. 아래 두 채우기 방식도 같은 GPU KD-tree를 사용함.
- RTX 5080, MiniShift 캡슐 원본 50만 점. 정상 54개로 뱅크를 새로 구성하고 동일 테스트 130개를 사용함.
- 무작위 후보 125,000개(25%, 시드 42), 특징 집계에서 FPS로 직접 뽑은 2,048개 및 최종 4,096개를 유지함. 예비 실행에서 직접 고른 인덱스가 순서 채우기와 같고 나머지만 다름을 확인함.
- 미선택 후보에서 시드 43의 무작위 우선순위를 사용해 2,048개를 중복 없이 채웠음. GPU KD-tree, FPFH 40·80·120, 집계 이웃 128개 및 평활화용 FPS·이웃 12개 유지.
- 측정 시점이 다름. 추론 시간은 특징 추출과 점수 계산의 합계이며, 입력 읽기·저장·평가 제외. 지표 차이는 %p.

## 전체 결과

| 항목 | 기존 GPU KD-tree FPS | FlashFPS 순서 채움 | FlashFPS 무작위 채움 | Δ 무작위−순서 | Δ 무작위−기존 |
|---|---:|---:|---:|---:|---:|
| 물체 AUROC (%) | 91.25 | 88.00 | 88.08 | +0.08 | −3.17 |
| 포인트 AUROC (%) | 80.75 | 79.17 | 78.92 | −0.25 | −1.83 |
| 포인트 AP (%) | 22.77 | 15.91 | 16.25 | +0.35 | −6.52 |
| 평균 추론 (ms/샘플) | 2779.21 | 1900.80 | 1942.30 | +41.49 | −836.91 |
| p95 추론 (ms/샘플) | 2894.77 | 2030.37 | 2068.80 | +38.42 | −825.98 |
| 메모리뱅크 구성 (초) | 128.78 | 92.55 | 89.87 | −2.67 | −38.90 |
| 특징 집계 FPS (ms/샘플) | 821.96 | 109.58 | 113.27 | +3.69 | −708.69 |
| 점수 평활화 FPS (ms/샘플) | 205.49 | 14.03 | 14.51 | +0.48 | −190.98 |
| 두 FPS 합계 (ms/샘플) | 1027.45 | 123.61 | 127.79 | +4.17 | −899.67 |
| FPFH용 전체 점 120-NN (ms/샘플) | 1106.81 | 1133.50 | 1152.52 | +19.02 | +45.72 |

## 난이도별 결과 (%)

| 항목 | 기존 GPU KD-tree FPS | FlashFPS 순서 채움 | FlashFPS 무작위 채움 | Δ 무작위−순서 | Δ 무작위−기존 |
|---|---:|---:|---:|---:|---:|
| easy · 물체 AUROC | 99.50 | 94.75 | 95.00 | +0.25 | −4.50 |
| easy · 포인트 AUROC | 98.15 | 98.51 | 98.56 | +0.04 | +0.41 |
| easy · 포인트 AP | 41.23 | 30.42 | 30.39 | −0.03 | −10.84 |
| medium · 물체 AUROC | 90.50 | 91.50 | 91.50 | 0.00 | +1.00 |
| medium · 포인트 AUROC | 82.32 | 84.00 | 80.61 | −3.40 | −1.72 |
| medium · 포인트 AP | 21.69 | 16.15 | 16.52 | +0.37 | −5.17 |
| hard · 물체 AUROC | 83.75 | 77.75 | 77.75 | 0.00 | −6.00 |
| hard · 포인트 AUROC | 64.10 | 57.91 | 60.30 | +2.39 | −3.80 |
| hard · 포인트 AP | 7.06 | 3.88 | 4.22 | +0.34 | −2.84 |

## 해석

- 기존 GPU KD-tree FPS 대비 무작위 채우기의 평균 추론 시간은 **30.11% 단축**됐지만, 물체 AUROC와 포인트 AUROC는 각각 **3.17%p**, **1.83%p** 낮았음.
- 물체 AUROC 차이는 정상·불량 1,200쌍 중 불량의 점수가 낮았던 쌍이 144개에서 143개로 한 쌍 줄어든 정도임. 동점 없음.
- 포인트 AP는 +0.35%p, hard 포인트 AUROC는 +2.39%p였으나 전체 포인트 AUROC는 −0.25%p였음. 하나의 시드·범주 결과이므로 무작위 채우기를 일관된 개선으로 보지 않음.
- 특징 집계 FPS는 +3.69ms. 전체 추론은 +41.49ms였지만 변경하지 않은 FPFH 조회도 느리게 측정돼 전체 시간 차이를 채우기 변경만으로 설명할 수 없음.

## 근거 자료

- **기존 GPU KD-tree FPS:** [요약](../source/results/official_vs_gpu_kdtree_20261104/gpu_kdtree/minishift/capsule/summary.json) · [단계별 시간](../source/results/official_vs_gpu_kdtree_20261104/gpu_kdtree/minishift/capsule/stage_summary.csv) · [샘플별 점수](../source/results/official_vs_gpu_kdtree_20261104/gpu_kdtree/minishift/capsule/test_timings.csv)
- **순서 채우기:** [요약](../source/results/capsule_gpu_kdtree_flashfps_random25_feature50_20261111/gpu_kdtree/minishift/capsule/summary.json) · [단계별 시간](../source/results/capsule_gpu_kdtree_flashfps_random25_feature50_20261111/gpu_kdtree/minishift/capsule/stage_summary.csv) · [샘플별 점수](../source/results/capsule_gpu_kdtree_flashfps_random25_feature50_20261111/gpu_kdtree/minishift/capsule/test_timings.csv)
- **무작위 채우기:** [요약](../source/results/capsule_gpu_kdtree_flashfps_random25_feature50_randomfill_20261111/gpu_kdtree/minishift/capsule/summary.json) · [단계별 시간](../source/results/capsule_gpu_kdtree_flashfps_random25_feature50_randomfill_20261111/gpu_kdtree/minishift/capsule/stage_summary.csv) · [샘플별 점수](../source/results/capsule_gpu_kdtree_flashfps_random25_feature50_randomfill_20261111/gpu_kdtree/minishift/capsule/test_timings.csv)
- [검증 기록](../source/results/capsule_gpu_kdtree_flashfps_random25_feature50_randomfill_20261111/verification.json) · [변경 코드](../source/simple3d_flashfps_random25_feature50_randomfill.py) · [실행 스크립트](../source/run_simple3d_capsule_flashfps_random25_feature50_randomfill.sh)
