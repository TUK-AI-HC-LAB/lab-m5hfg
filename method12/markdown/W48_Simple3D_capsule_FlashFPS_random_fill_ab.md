# W48 Simple3D 캡슐 — FlashFPS 순서 채움 vs 무작위 채움

**결과:** 나머지 2,048점을 무작위로 채우자 캡슐 전체 물체 AUROC는 **+1.50%p**, 포인트 AUROC는 **+0.54%p**, 포인트 AP는 **+0.50%p**였음. 평균 추론은 **2,121.04 → 2,092.74ms/샘플**로 1.33% 짧았음. 단, 단일 시드·카테고리의 1회씩 측정임.

## 비교 조건

- MiniShift 캡슐, RTX 5080, 샘플당 50만 점, 정상 54개·동일 테스트 130개. 같은 세션에서 순서판 → 무작위판으로 실행하고, 정상 메모리뱅크는 각각 새로 구성함.
- 두 버전 모두 후보 25%(시드 42)에서 FPS로 먼저 2,048점을 고름. 남은 2,048점만 **미선택 후보 인덱스 순서**에서 **미선택 후보 중 무작위 비복원 추출(시드 43)**로 바꿈. 예비 검증에서 먼저 고른 2,048점이 같고, 변경판의 4,096점에 중복이 없음을 확인함.
- 최근 W48 FlashFPS 기준과 동일하게 GPU KD-tree, FPFH 40·80·120, 집계 128-NN, 평활화의 기존 FPS 1,024점·12-NN, 전체 정렬 보간을 유지함. STEP 미적용. 평활화에도 다른 FPS를 사용하던 W46 무작위 채움 비교와는 별도임.

## 결과

차이 = 무작위 − 순서. 추론은 특징 추출+점수 계산이며, 입력 읽기·저장·지표 계산은 제외함.

| 항목 | 순서 채움 | 무작위 채움 | 차이 |
|---|---:|---:|---:|
| 전체 물체 AUROC (%) | 93.17 | 94.67 | +1.50%p |
| 전체 포인트 AUROC (%) | 81.84 | 82.39 | +0.54%p |
| 전체 포인트 AP (%) | 22.30 | 22.81 | +0.50%p |
| 평균 추론 (ms/샘플) | 2,121.04 | 2,092.74 | −28.30ms |
| p95 추론 (ms/샘플) | 2,272.33 | 2,246.00 | −26.33ms |
| 메모리뱅크 구성 (초) | 89.62 | 90.58 | +0.96초 |
| 집계 대표점 선택 (ms/샘플) | 112.35 | 109.06 | −3.29ms |
| 평활화 대표점 선택 (ms/샘플) | 217.54 | 211.85 | −5.69ms |
| FPFH 120-NN (ms/샘플) | 1,133.39 | 1,128.46 | −4.93ms |

난이도별 포인트 AUROC는 **easy +0.17%p, medium −4.16%p, hard +5.62%p**였음. 전체가 올랐지만 모든 난이도에서 오른 것은 아님. 변경하지 않은 단계에서도 시간이 줄었으므로 전체 28.30ms 차이를 무작위 채움의 효과로 단정하지 않음. 차이는 반올림 전 값으로 계산함.

## 실행 근거

- [검증 결과·소스 해시](../source/results/capsule_flashfps_fill_ab_20261123/verification.json)
- 순서판: [지표](../source/results/capsule_flashfps_fill_ab_20261123/ordered/gpu_kdtree/minishift/capsule/metrics.csv) · [단계별 시간](../source/results/capsule_flashfps_fill_ab_20261123/ordered/gpu_kdtree/minishift/capsule/stage_summary.csv)
- 무작위판: [지표](../source/results/capsule_flashfps_fill_ab_20261123/random/gpu_kdtree/minishift/capsule/metrics.csv) · [단계별 시간](../source/results/capsule_flashfps_fill_ab_20261123/random/gpu_kdtree/minishift/capsule/stage_summary.csv)
