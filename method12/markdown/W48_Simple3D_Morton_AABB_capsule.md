# W48 Simple3D 캡슐 — Morton 묶음의 경계를 구에서 직육면체로 교체

## 왜 했나

이전 Morton 실험에서 빠른 인덱스 생성의 이득이 느슨한 구 경계와 후보 검사 증가로 상쇄됨. 같은 묶음에서 경계만 축에 평행한 직육면체(AABB)로 바꾸면, 정확한 이웃과 FPFH 값을 유지하면서 후보 검사 및 생성+조회 시간을 줄일 수 있는지 검증함.

## 비교 조건과 결과

MiniShift capsule 테스트 0·65·129번, 각 50만 점. 표본마다 각 방식 6회, 가능한 실행 순서 6개를 사용함. 원본 좌표, 거리·동점 규칙, Radix 정렬 및 이웃 120개 유지. 두 Morton 방식의 묶음과 내부 인덱스 순서 일치 확인. 직육면체가 모든 소속 점을 포함하는지 검사하고, 합성 3종은 전체 거리 정렬과 대조함.

| 방식 | 구조 생성 (ms) | 120-NN 조회 (ms) | 생성+조회 (ms) | 방문 묶음/쿼리 | 후보 거리 계산/쿼리 |
|---|---:|---:|---:|---:|---:|
| 기존 K-means 50회 + 구 | 292.55 | 249.23 | 546.18 | 2.27 | 2,223.93 |
| Morton + 구 | 1.65 | 913.22 | 915.03 | 14.11 | 13,777.93 |
| Morton + 직육면체 | 1.68 | 436.13 | 437.90 | 5.57 | 5,438.10 |

시간은 각 샘플 6회 중앙값의 샘플 간 평균. 합계 중앙값은 생성·조회 각각의 중앙값과 따로 계산함. 방문·후보 수는 별도 계측 실행에서 얻었으며, 계측 커널의 시간은 표에서 제외함.

- 세 표본에서 순서까지 포함한 120-NN 및 FPFH 40·80·120 값 불일치 0개.
- 같은 Morton 묶음에서 직육면체로 바꾸니 후보 거리 계산이 약 60.5% 감소함. 구 경계가 이전 손해의 큰 원인이었다는 가설을 지지함.
- 직육면체판은 기존 K-means보다 **조회 자체는 여전히 느리지만**, 구조 생성이 빨라 **생성+조회 합계는 19.8% 감소**함. 이는 전체 추론 감소율이 아님.
- 일반적인 AABB 사용 자체가 논문 기여는 아님. 캡슐 3개에서의 결과이며 다른 물체에 바로 일반화하지 않음.

## 전체 캡슐 검증

정상 54개로 버전별 메모리뱅크를 새로 구성하고 테스트 130개를 비교함. 각 버전 1회. 두 방식 모두 Radix 정렬을 사용함.

| 지표 | K-means 50회 + 구 | Morton + 직육면체 |
|---|---:|---:|
| 인덱스 생성 (ms/샘플) | 329.185 | 1.883 |
| 120-NN 조회 (ms/샘플) | 248.441 | 432.880 |
| 전체 평균 추론 (ms/샘플) | 1,498.845 | 1,376.168 |
| 전체 p95 추론 (ms/샘플) | 1,576.438 | 1,429.028 |
| 뱅크 구성 (초) | 62.427 | 56.919 |
| 물체 AUROC (%) | 93.1667 | 93.1667 |
| 포인트 AUROC (%) | 81.8179 | 81.8179 |
| 포인트 AP (%) | 22.3010 | 22.3010 |

전체 평균 추론은 **8.18% 감소**함. 6,500만 점에서 이상 점수 불일치 0개. 물체 점수·정상 뱅크·전체 및 난이도별 평가 지표도 일치함. 조회 자체는 느려졌으나 인덱스 생성 비용 감소가 이를 상쇄함. 캡슐 외 카테고리에서는 검증하지 않음.

## 근거

- [표본별 기록](../source/results/morton_aabb_capsule/samples.csv) · [반복별 시간](../source/results/morton_aabb_capsule/repeats.csv) · [통과 여부](../source/results/morton_aabb_capsule/summary.json)
- [전체 캡슐 검증 JSON](../source/results/morton_aabb_capsule_e2e/verification.json) · [샘플별 비교](../source/results/morton_aabb_capsule_e2e/capsule/paired_samples.csv)
- [K-means 단계별 시간](../source/results/morton_aabb_capsule_e2e/capsule/baseline/step/gpu_kdtree/minishift/capsule/stage_summary.csv) · [AABB 단계별 시간](../source/results/morton_aabb_capsule_e2e/capsule/radix/step/gpu_kdtree/minishift/capsule/stage_summary.csv). 기존 측정 도구의 폴더명은 `baseline`/ `radix`이며, AABB CSV의 인덱스 생성 설명에 남은 ‘50 Lloyd updates’는 기존 계측 이름일 뿐 실제 AABB 실행에는 Lloyd 갱신이 없음.
