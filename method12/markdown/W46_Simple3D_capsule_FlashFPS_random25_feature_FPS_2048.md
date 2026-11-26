# W46 Simple3D 캡슐 — FlashFPS 특징 집계의 FPS 반복 수 비교

무작위 후보 25%를 고정한 상태에서 특징 집계용 FPS 선택을 1,024개에서 2,048개로 늘리자 전체 물체 AUROC **+0.58%p**, 포인트 AUROC **+0.54%p**. 그러나 전체 포인트 AP **−0.10%p**, hard 포인트 AUROC **−3.59%p**. 전반적인 성능 개선으로 결론 내리기 어려움.

## 조건

- RTX 5080, MiniShift `capsule` 50만 점/샘플. 정상 54개에서 메모리뱅크를 각각 새로 구성하고 동일한 테스트 130개를 평가함.
- 두 무작위 설정은 후보 125,000개(25%, 시드 42), 후보 순서, 미선택 점으로 채우는 규칙을 유지함. 특징 집계용 최종 4,096개 중 FPS로 뽑는 점만 1,024개에서 2,048개로 변경함.
- 점수 평활화는 최종 1,024개 중 기존대로 256개를 FPS로 고름. GPU KD-tree, FPFH 40·80·120, 이웃 수 128·12, FPS-Cache 미사용 조건도 유지함.
- 이전 기록과 측정 시점이 다르므로 전체 추론 시간 차이를 FPS 변경의 인과 효과로 해석하지 않음.

## 전체 결과

지표 차이는 %p, 시간 차이는 표에 표시한 단위임. 추론은 특징 추출과 점수 계산 합계로 입력 읽기·저장·지표 계산은 제외함.

| 항목 | 기존 GPU KD-tree FPS | 무작위 후보 · 집계 FPS 1,024개 | 무작위 후보 · 집계 FPS 2,048개 | Δ 2,048−1,024 |
|---|---:|---:|---:|---:|
| 물체 AUROC (%) | 91.25 | 87.42 | 88.00 | +0.58 |
| 포인트 AUROC (%) | 80.75 | 78.63 | 79.17 | +0.54 |
| 포인트 AP (%) | 22.77 | 16.00 | 15.91 | −0.10 |
| 평균 추론 (ms/샘플) | 2779.21 | 1954.92 | 1900.80 | −54.12 |
| 추론 p95 (ms/샘플) | 2894.77 | 2107.91 | 2030.37 | −77.54 |
| 메모리뱅크 구성 (초) | 128.78 | 90.00 | 92.55 | +2.55 |
| 두 FPS 합계 (ms/샘플) | 1027.45 | 73.52 | 123.61 | +50.09 |
| 특징 집계 FPS (ms/샘플) | 821.96 | 58.52 | 109.58 | +51.06 |
| 점수 평활화 FPS (ms/샘플) | 205.49 | 15.00 | 14.03 | −0.96 |
| FPFH용 전체 점 120-NN (ms/샘플) | 1106.81 | 1207.72 | 1133.50 | −74.22 |
| 포인트 점수 보간 (ms/샘플) | 278.04 | 310.76 | 288.82 | −21.94 |

## 난이도별 성능 (%)

| 항목 | 기존 GPU KD-tree FPS | 무작위 후보 · 집계 FPS 1,024개 | 무작위 후보 · 집계 FPS 2,048개 | Δ 2,048−1,024 |
|---|---:|---:|---:|---:|
| easy · 물체 AUROC | 99.50 | 94.75 | 94.75 | 0.00 |
| easy · 포인트 AUROC | 98.15 | 98.36 | 98.51 | +0.16 |
| easy · 포인트 AP | 41.23 | 30.58 | 30.42 | −0.16 |
| medium · 물체 AUROC | 90.50 | 89.50 | 91.50 | +2.00 |
| medium · 포인트 AUROC | 82.32 | 79.27 | 84.00 | +4.74 |
| medium · 포인트 AP | 21.69 | 16.51 | 16.15 | −0.36 |
| hard · 물체 AUROC | 83.75 | 78.00 | 77.75 | −0.25 |
| hard · 포인트 AUROC | 64.10 | 61.50 | 57.91 | −3.59 |
| hard · 포인트 AP | 7.06 | 3.91 | 3.88 | −0.04 |

## 해석

- 물체 AUROC는 정상·불량 1,200쌍 중 불량에 더 낮은 점수를 준 쌍이 151개에서 144개로 줄며 상승했음. 동점은 없었음.
- medium 포인트 AUROC는 **+4.74%p**, hard는 **−3.59%p**였음. 전체 AUROC만 보면 hard 성능 악화를 놓칠 수 있음.
- 집계 FPS 자체는 **+51.06ms** 걸렸음. 전체 추론 시간이 이번 실행에서 더 짧았지만, 변경하지 않은 FPFH용 이웃 조회와 보간도 더 짧게 측정됐으므로 FPS 반복을 늘리면 전체 추론이 빨라진다고 볼 수 없음.
- 무작위 후보 방식의 성능을 더 높이려면 반복 수 증가보다, FPS로 고르지 않고 채우는 나머지 중심점의 분포나 평활화용 FPS의 영향을 분리해 검증할 필요가 있음.

## 근거 자료

- **기존 FPS:** [요약](../source/results/official_vs_gpu_kdtree_20261104/gpu_kdtree/minishift/capsule/summary.json) · [단계별 시간](../source/results/official_vs_gpu_kdtree_20261104/gpu_kdtree/minishift/capsule/stage_summary.csv) · [샘플별 점수](../source/results/official_vs_gpu_kdtree_20261104/gpu_kdtree/minishift/capsule/test_timings.csv)
- **직전 무작위 방식:** [요약](../source/results/capsule_gpu_kdtree_flashfps_random25_20261111/gpu_kdtree/minishift/capsule/summary.json) · [단계별 시간](../source/results/capsule_gpu_kdtree_flashfps_random25_20261111/gpu_kdtree/minishift/capsule/stage_summary.csv) · [샘플별 점수](../source/results/capsule_gpu_kdtree_flashfps_random25_20261111/gpu_kdtree/minishift/capsule/test_timings.csv)
- **특징 집계 FPS 2,048개:** [요약](../source/results/capsule_gpu_kdtree_flashfps_random25_feature50_20261111/gpu_kdtree/minishift/capsule/summary.json) · [단계별 시간](../source/results/capsule_gpu_kdtree_flashfps_random25_feature50_20261111/gpu_kdtree/minishift/capsule/stage_summary.csv) · [샘플별 점수](../source/results/capsule_gpu_kdtree_flashfps_random25_feature50_20261111/gpu_kdtree/minishift/capsule/test_timings.csv)
- [검증 기록](../source/results/capsule_gpu_kdtree_flashfps_random25_feature50_20261111/verification.json) · [FPS 변경 코드](../source/simple3d_flashfps_random25_feature50.py) · [실행 스크립트](../source/run_simple3d_capsule_flashfps_random25_feature50.sh)
