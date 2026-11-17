# W47 Simple3D 캡슐 — FlashFPS·STEP·Top-3 기여도 비교

동일 RTX 5080, MiniShift 정상 54개·테스트 130개. 각 실행 별도 예열·메모리뱅크 재구성. 저장·파일 로딩·지표 집계는 추론 시간에서 제외함.
STEP은 FPFH용 120-NN만 교체(512군집, Lloyd 50회). 집계 128-NN과 평활화 12-NN은 전부 GPU KD-tree. Top-3는 4,096점 집계 보간에만 적용하고 동점 행은 전체 정렬로 처리함.
한 세션 네 구성 각 1회, 실행 순서 B→D→A→C. 각 효과는 앞 구성을 기준으로 측정한 추가 변화이며, 독립적인 효과나 순서 영향까지 추정한 것은 아님.
완료 구성: 4/4.

| 구성 | 평균 추론 (ms) | p95 (ms) | 뱅크 (초) | 물체 AUROC (%) | 포인트 AUROC (%) | 포인트 AP (%) |
|---|---:|---:|---:|---:|---:|---:|
| A · GPU KD-tree + 기존 FPS + 전체 정렬 | 3,013.27 | 3,218.99 | 135.78 | 91.25000 | 80.74655 | 22.77132 |
| B · GPU KD-tree + FlashFPS + 전체 정렬 | 2,221.09 | 2,458.42 | 101.66 | 93.16667 | 81.84305 | 22.30246 |
| C · STEP 512 + FlashFPS + 전체 정렬 | 1,782.36 | 1,847.12 | 68.28 | 93.16667 | 81.81789 | 22.30099 |
| D · STEP 512 + FlashFPS + 4,096점 보간만 Top-3 | 1,596.94 | 1,697.05 | 64.56 | 93.16667 | 81.81789 | 22.30099 |

## 핵심 단계별 시간 (ms/샘플)

| 단계 | A | B | C | D |
|---|---:|---:|---:|---:|
| 집계 대표점 선택 | 903.74 | 116.40 | 123.07 | 117.05 |
| 평활화 대표점 선택 | 225.80 | 224.71 | 239.19 | 226.26 |
| FPFH 120-NN 전체 | 1,196.79 | 1,200.75 | 682.89 | 627.27 |
| ↳ STEP 군집 생성 | — | — | 348.53 | 312.45 |
| ↳ STEP 120-NN 조회 | — | — | 333.38 | 313.95 |
| 집계 점수 보간 | 248.78 | 247.52 | 266.83 | 180.40 |
| 평활화 점수 보간 | 61.22 | 60.07 | 64.26 | 61.00 |

STEP 군집 생성과 조회는 FPFH용 120-NN 전체에 포함되므로 중복 합산하지 않음.

## 이전 구성 대비 변화

- **A→B**: 전체 추론 -792.18ms (-26.29%), 점수 불일치 64,999,985/65,000,000점, 포인트 점수 최대 절대 차이 251.86963, 물체 점수 최대 절대 차이 150.02654, 뱅크 일치 False, 난이도별 지표 일치 False.
- **B→C**: 전체 추론 -438.74ms (-19.75%), 점수 불일치 492,265/65,000,000점, 포인트 점수 최대 절대 차이 1.1348473, 물체 점수 최대 절대 차이 0, 뱅크 일치 False, 난이도별 지표 일치 False.
- **C→D**: 전체 추론 -185.41ms (-10.40%), 점수 불일치 0/65,000,000점, 포인트 점수 최대 절대 차이 0, 물체 점수 최대 절대 차이 0, 뱅크 일치 True, 난이도별 지표 일치 True.
- **A→D의 전체 추론 시간 변화:** -1416.33ms (-47.00%). GPU KD-tree를 사용한 기준판 대비이며 공식 CPU 구현체 대비 수치가 아님.

## 실행 근거

- [완료 기록](../source/results/capsule_optimization_ladder_20261117/progress.json) · [검증 JSON](../source/results/capsule_optimization_ladder_20261117/verification.json)
- A: [요약](../source/results/capsule_optimization_ladder_20261117/A/kdtree/gpu_kdtree/minishift/capsule/summary.json) · [단계별 시간](../source/results/capsule_optimization_ladder_20261117/A/kdtree/gpu_kdtree/minishift/capsule/stage_summary.csv) · [GPU·메모리](../source/results/capsule_optimization_ladder_20261117/A/kdtree/gpu_kdtree/minishift/capsule/resource_summary.csv)
- B: [요약](../source/results/capsule_optimization_ladder_20261117/B/kdtree/gpu_kdtree/minishift/capsule/summary.json) · [단계별 시간](../source/results/capsule_optimization_ladder_20261117/B/kdtree/gpu_kdtree/minishift/capsule/stage_summary.csv) · [GPU·메모리](../source/results/capsule_optimization_ladder_20261117/B/kdtree/gpu_kdtree/minishift/capsule/resource_summary.csv)
- C: [요약](../source/results/capsule_optimization_ladder_20261117/C/step/gpu_kdtree/minishift/capsule/summary.json) · [단계별 시간](../source/results/capsule_optimization_ladder_20261117/C/step/gpu_kdtree/minishift/capsule/stage_summary.csv) · [GPU·메모리](../source/results/capsule_optimization_ladder_20261117/C/step/gpu_kdtree/minishift/capsule/resource_summary.csv)
- D: [요약](../source/results/capsule_optimization_ladder_20261117/D/step/gpu_kdtree/minishift/capsule/summary.json) · [단계별 시간](../source/results/capsule_optimization_ladder_20261117/D/step/gpu_kdtree/minishift/capsule/stage_summary.csv) · [GPU·메모리](../source/results/capsule_optimization_ladder_20261117/D/step/gpu_kdtree/minishift/capsule/resource_summary.csv)
- [A→B의 샘플별 점수·시간 차이](../source/results/capsule_optimization_ladder_20261117/paired_A_B.csv)
- [B→C의 샘플별 점수·시간 차이](../source/results/capsule_optimization_ladder_20261117/paired_B_C.csv)
- [C→D의 샘플별 점수·시간 차이](../source/results/capsule_optimization_ladder_20261117/paired_C_D.csv)
