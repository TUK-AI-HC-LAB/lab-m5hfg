# W46 Simple3D 캡슐 — 특징 집계 FlashFPS와 기존 평활화 FPS 조합

직전 2,048개 순서 채우기 메모리뱅크와 특징 집계를 고정하고 평활화 FPS만 되돌리자, 물체 AUROC **+5.17%p**, 포인트 AUROC **+2.67%p**, 포인트 AP **+6.40%p**. 전체 두 AUROC는 원래 GPU KD-tree 버전보다 높았으며, 평균 추론 시간은 원래 버전보다 **22.5% 짧았음**.

## 조건

- RTX 5080, MiniShift `capsule` 50만 점/샘플. 직전 FlashFPS 순서 채우기가 정상 54개에서 생성한 메모리뱅크를 파일에서 그대로 불러옴. 재학습이나 재구성 없이 동일 테스트 130개를 평가함. 파일 경로·SHA-256 검증.
- 특징 집계는 무작위 후보 125,000개(25%, 시드 42)에서 2,048개를 FPS로 고르고, 나머지 2,048개를 순서대로 채워 최종 4,096개를 사용함. 집계 이웃 128개, FPFH 40·80·120, GPU KD-tree 유지.
- **평활화만 변경:** 후보 25%에서 256개를 직접 고르고 나머지를 채우는 대신, 원본 50만 점에서 최종 1,024개 모두를 기존 PointNet2 CUDA FPS로 선택함. 평활화 이웃 12개 및 점수 계산·보간은 유지함.
- 측정 시점은 다름. 추론은 특징 추출과 점수 계산의 합이며 입력 읽기·결과 저장·평가 제외. 지표 차이는 %p임.

## 전체 결과

| 항목 | 기존 GPU KD-tree FPS | 집계·평활화 FlashFPS | 집계만 FlashFPS · 평활화 기존 FPS | Δ 직전 대비 | Δ 기존 GPU KD-tree 대비 |
|---|---:|---:|---:|---:|---:|
| 물체 AUROC (%) | 91.25 | 88.00 | 93.17 | +5.17 | +1.92 |
| 포인트 AUROC (%) | 80.75 | 79.17 | 81.84 | +2.67 | +1.10 |
| 포인트 AP (%) | 22.77 | 15.91 | 22.30 | +6.40 | −0.47 |
| 평균 추론 (ms/샘플) | 2779.21 | 1900.80 | 2154.50 | +253.69 | −624.71 |
| p95 추론 (ms/샘플) | 2894.77 | 2030.37 | 2289.37 | +259.00 | −605.40 |
| 특징 집계 FPS (ms/샘플) | 821.96 | 109.58 | 113.58 | +4.00 | −708.38 |
| 점수 평활화 FPS (ms/샘플) | 205.49 | 14.03 | 220.17 | +206.14 | +14.68 |
| 두 FPS 합계 (ms/샘플) | 1027.45 | 123.61 | 333.75 | +210.14 | −693.70 |
| FPFH용 전체 점 120-NN (ms/샘플) | 1106.81 | 1133.50 | 1155.28 | +21.78 | +48.48 |
| 포인트 점수 보간 (ms/샘플) | 278.04 | 288.82 | 299.78 | +10.96 | +21.74 |

## 난이도별 결과 (%)

| 항목 | 기존 GPU KD-tree FPS | 집계·평활화 FlashFPS | 집계만 FlashFPS · 평활화 기존 FPS | Δ 직전 대비 | Δ 기존 GPU KD-tree 대비 |
|---|---:|---:|---:|---:|---:|
| easy · 물체 AUROC | 99.50 | 94.75 | 100.00 | +5.25 | +0.50 |
| easy · 포인트 AUROC | 98.15 | 98.51 | 98.65 | +0.13 | +0.50 |
| easy · 포인트 AP | 41.23 | 30.42 | 39.74 | +9.32 | −1.50 |
| medium · 물체 AUROC | 90.50 | 91.50 | 95.00 | +3.50 | +4.50 |
| medium · 포인트 AUROC | 82.32 | 84.00 | 87.75 | +3.74 | +5.42 |
| medium · 포인트 AP | 21.69 | 16.15 | 21.97 | +5.82 | +0.28 |
| hard · 물체 AUROC | 83.75 | 77.75 | 84.50 | +6.75 | +0.75 |
| hard · 포인트 AUROC | 64.10 | 57.91 | 61.54 | +3.63 | −2.56 |
| hard · 포인트 AP | 7.06 | 3.88 | 6.86 | +2.98 | −0.20 |

## 해석

- 정상·불량 1,200쌍 중 불량이 정상보다 낮게 나온 쌍이 144개 → 82개로 감소함(동점 없음). 메모리뱅크가 같으므로 평활화 FPS가 최종 물체 점수의 순위에도 영향을 미쳤다는 근거임.
- 전체 물체 AUROC와 포인트 AUROC는 원래 버전보다 각각 +1.92%p·+1.10%p. 그러나 전체 포인트 AP는 −0.47%p, hard 포인트 AUROC는 −2.56%p로 모든 지표가 개선된 것은 아님.
- 직전 대비 평균 추론 +253.69ms 중 평활화 FPS 증가량이 +206.14ms였음. 단계 계측 범위는 일부 겹칠 수 있으며 측정 시점도 다름.

## 근거 자료

- **기존 GPU KD-tree FPS:** [요약](../source/results/official_vs_gpu_kdtree_20261104/gpu_kdtree/minishift/capsule/summary.json) · [단계별 시간](../source/results/official_vs_gpu_kdtree_20261104/gpu_kdtree/minishift/capsule/stage_summary.csv) · [샘플별 점수](../source/results/official_vs_gpu_kdtree_20261104/gpu_kdtree/minishift/capsule/test_timings.csv)
- **직전 FlashFPS 순서 채우기:** [요약](../source/results/capsule_gpu_kdtree_flashfps_random25_feature50_20261111/gpu_kdtree/minishift/capsule/summary.json) · [단계별 시간](../source/results/capsule_gpu_kdtree_flashfps_random25_feature50_20261111/gpu_kdtree/minishift/capsule/stage_summary.csv) · [샘플별 점수](../source/results/capsule_gpu_kdtree_flashfps_random25_feature50_20261111/gpu_kdtree/minishift/capsule/test_timings.csv)
- **평활화만 기존 FPS:** [요약](../source/results/capsule_gpu_kdtree_flashfps_random25_feature50_smooth_original_20261111/gpu_kdtree/minishift/capsule/summary.json) · [단계별 시간](../source/results/capsule_gpu_kdtree_flashfps_random25_feature50_smooth_original_20261111/gpu_kdtree/minishift/capsule/stage_summary.csv) · [샘플별 점수](../source/results/capsule_gpu_kdtree_flashfps_random25_feature50_smooth_original_20261111/gpu_kdtree/minishift/capsule/test_timings.csv)
- [검증 기록](../source/results/capsule_gpu_kdtree_flashfps_random25_feature50_smooth_original_20261111/verification.json) · [실행 코드](../source/measure_simple3d_capsule_flashfps_smooth_original.py) · [실행 스크립트](../source/run_simple3d_capsule_flashfps_random25_feature50_smooth_original.sh)
