# W48 Simple3D — STEP 군집 하한 정렬의 Radix 교체 검증

완료한 비교: 2/2개 카테고리. 각 버전·카테고리 1회.

## 질문과 사전 가설

- 질문: 군집 방문 순서는 그대로 두고 512개 하한 정렬만 CUB stable radix로 교체하면 전체 추론이 빨라지는가?
- 가설: 이웃 순서·뱅크·이상 점수가 같고 조회 시간이 감소함. 전체 시간에서는 감소가 작거나 다른 단계 변동에 가려질 수 있음.
- 고정: FlashFPS hybrid, STEP512/Lloyd50, 4,096점 보간 Top-3, 1,024점 보간 전체 정렬, 집계128, 평활화12, seed0.
- 변경: 독립 STEP 계열 구현의 군집 하한 정렬만 bitonic에서 NVIDIA CUB BlockRadixSort로 변경. 후보 병합 정렬 유지. 새 알고리즘의 최초 제안이라고 주장하지 않음.
- 두 버전 모두 정상 데이터로 뱅크를 새로 구성함. capsule은 기존→Radix, vase9는 Radix→기존 순서. 입력/저장/지표 집계는 추론 시간 제외.
- Δ = Radix−기존. ms/샘플, 음수가 빠름. 카테고리별 1회이므로 반복 분산이나 통계적 유의성은 미확인.

| 카테고리 | 평균 기존 (ms) | 평균 Radix (ms) | Δ (ms) | Δ (%) | p95 기존 (ms) | p95 Radix (ms) | 점수 불일치/전체 | 뱅크·지표 일치 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| capsule | 1516.746 | 1440.950 | -75.795 | -5.00 | 1656.514 | 1543.002 | 0/65,000,000 | True/True |
| vase9 | 410.907 | 401.592 | -9.314 | -2.27 | 476.433 | 463.213 | 0/4,564,539 | True/True |

## 원본 근거

- 기준 commit: `66956cf11839902944dffc5082210cb859f9b5f8`. 새 파일은 미커밋이며 plan.json에 소스 해시 기록.
- [실행 스크립트](../source/run_simple3d_step_radix_e2e.sh) · [코드](../source/run_simple3d_step_radix_e2e.py)
- [진행 상태](../source/results/step_radix_e2e_20261118/progress.json) · [계획·소스 해시](../source/results/step_radix_e2e_20261118/plan.json)
- [비교 CSV](../source/results/step_radix_e2e_20261118/category_comparison.csv) · [전체 단계별 시간](../source/results/step_radix_e2e_20261118/stage_comparison.csv)
- [전체·난이도별 AUROC/AP](../source/results/step_radix_e2e_20261118/metrics_comparison.csv) · [자원 계측](../source/results/step_radix_e2e_20261118/resource_comparison.csv)
- [검증 JSON](../source/results/step_radix_e2e_20261118/verification.json)
- 각 물체 폴더의 paired_samples.csv에 샘플별 점수·시간 차이를 저장함. raw 뱅크와 점수는 WSL에만 저장.
- 단계별 시간은 중첩되어 단순 합산하지 않음. GPU 전체 사용량에는 다른 프로세스 포함, Torch 할당에는 CuPy 제외.
- 순서: 병목 집계 → 정렬 제거 후보 실패 → 작업량 진단 → 정렬 유지 Radix 연산 시험 → 현재 전체 파이프라인 검증. 이후 규모 변화·반복 측정·다른 모델 검증은 아직 미실시.
