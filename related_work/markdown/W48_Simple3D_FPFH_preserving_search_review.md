# W48 FPFH를 유지하는 탐색 비용 절감 검토

조사일: 2026-11-19. 검토만 수행함. 새 탐색 구현·설치·실험은 진행하지 않음.

## 1. 결론

**다음 후보로는 ‘재사용을 위한 전체 128-NN 확대’보다, 현재 STEP 계열 구현의 비싼 인덱스 생성 절차를 가벼운 공간 분할로 바꾸는 방향을 우선 검토할 가치가 있음.**

목표는 FPFH 자체를 바꾸는 것이 아니라, 같은 원본 좌표에서 같은 최근접 이웃을 더 적은 준비 비용으로 찾는 것임. 빨라진다는 결과나 새 알고리즘의 새로움은 아직 확인하지 않음.

## 2. 현재 비용이 어디서 발생하는가

아래 수치는 이전 capsule의 **Radix 적용 전체 실행 1회**에서 측정한 평균임. SHOT 실행과 혼합한 값이 아님.

| 단계 | 현재 구현 | 평균 ms/샘플 |
|---|---|---:|
| 법선 | CPU Open3D, 최대10개 이웃 | 105.254 |
| 별도 GPU KD-tree 구성 | 집계128-NN·평활화12-NN에서 재사용 | 16.180 |
| STEP 군집 인덱스 구성 | 512개 군집, Lloyd 갱신50회 | **290.613** |
| FPFH용120-NN 조회 | STEP 계열 탐색 + Radix 하한 정렬 | **245.037** |
| 대표점 집계 이웃 조회 | 같은 GPU KD-tree, 4096개 쿼리·k128 | 32.354 |
| 평활화 이웃 조회 | 같은 GPU KD-tree, 1024개 쿼리·k12 | 5.460 |

- [단계별 원본 CSV](../../method12/source/results/step_radix_e2e_20261118/capsule/radix/step/gpu_kdtree/minishift/capsule/stage_summary.csv).
- [전체 비교](../../method12/markdown/W48_Simple3D_STEP_radix_end_to_end.md).
- 현재 군집 생성은 초기 배정과50회 갱신 후 최종 배정을 포함함. 단순히 색인을 한 번 정렬하는 방식이 아님. [indexing 함수](../../method12/source/measure_simple3d_capsule_step.py)
- 현재 경로는 CPU 법선 검색, GPU KD-tree, STEP 군집이라는 서로 다른 준비 비용을 부담함. 그러나 여러 구조가 있다는 사실만으로 없앨 수 있는 비용이 모두 중복이라고 단정하지 않음.

## 3. 이미 하고 있는 재사용과 실패한 후보

| 아이디어 | 확인한 사실 | 이번 판단 |
|---|---|---|
| FPFH40·80·120 탐색 공유 | 현재 이미120-NN 한 번의 앞부분을 재사용함 | 새 실험으로 다시 제안하지 않음 |
| 집계128-NN과 평활화12-NN이 같은 KD-tree 사용 | 현재 이미 하나의 GPU KD-tree를 사용함 | 이 둘을 공유한다는 것만으로 새 이득 없음 |
| 전체 이웃을128개로 늘려 대표점 행 재사용 | 과거 두 조회 합계1156.557→1275.899ms로 악화 | 그대로 재시도하지 않음 |
| 쿼리를 Morton 순서로 정렬 | 과거 집계 조회32.927→34.398ms로 악화 | 동일한 쿼리 재정렬은 우선 제외 |
| 법선용 이웃 재사용 | 과거 CPU↔GPU 공분산 왕복 구현은226.26→323.42ms | 왕복 구현은 제외. GPU 내부 완결은 별도 후보 |
| 직접 사용하는 FPFH만 계산 | 표본상 SPFH 의존성이 거의 전체 점을 덮음 | 주요 탐색 제거 후보로 미채택 |

과거 결과는 당시 기준판에서 나온 값이며 현재 Radix판의 예상 시간으로 쓰지 않음.

근거: 128-NN 재사용 (GitHub 미첨부: `W47_Simple3D_capsule_FlashFPS_neighbor_reuse128.md`), Morton 쿼리 순서 (GitHub 미첨부: `W47_Simple3D_capsule_FlashFPS_Morton_128NN.md`), 법선 재사용 (GitHub 미첨부: `W47_Simple3D_capsule_Faster_PFH_normal_reuse.md`), [FPFH 의존성](../../method12/markdown/W48_Simple3D_FPFH_dependency_and_avoidable_work.md).

## 4. 해결 후보별 판단

### 후보 A. 저비용 공간 분할로 STEP 인덱스 구성 — 우선 검토

**원인 가설:** 정확한120-NN을 찾으려는 현재 경로에서, 좋은 군집을 얻기 위한50회 반복 비용이 실제 조회 절감 이득에 비해 클 수 있음.

검토할 구체적 형태:

```text
현재: 원본 점 → K-means50회 → 군집·반경 → 정확한120-NN 탐색
후보: 원본 점 → 공간 키 정렬·512개 묶음 → 묶음의 보수적 경계 → 정확한120-NN 탐색
```

- Morton 공간 키는 점의 **묶음 순서만** 정함. 양자화한 좌표로 거리나 FPFH를 계산하지 않음.
- 원본 점을 모두 포함하고 원본 인덱스를 유지함. 묶음 경계가 모든 소속 점을 포함하도록 계산함.
- 처음에는 현재512개 묶음·Radix 탐색·FPFH·FlashFPS·후처리를 유지하고 인덱스 구성만 교체하는 것이 원인 분리에 유리함.
- 공간 정렬 후 일정 크기로 묶는 시제품은 완전한 LBVH 구현과 다름. LBVH 논문 재현이라고 부르면 안 됨.
- **실패 가능성:** 공간적으로 멀리 떨어진 점들이 같은 묶음에 들어가 경계가 커지면 조회 후보가 늘어날 수 있음. 생성 시간만 줄고 전체는 느려질 수 있음.
- 판정은 반드시 **구성+조회 합계**로 내림. 290.613ms가 전부 사라진다고 미리 예상하지 않음.
- 과거 Morton 실험은4,096개 쿼리의 방문 순서를 바꾼 것임. 이번 후보인 원본 점의 검색 인덱스 분할 변경과는 다름.

빠른 공간 계층 생성과 정확한 GPU kNN을 함께 다룬 LBVH 연구가 관련 근거임 [1]. 다만 해당 논문의 성능이 현재 데이터·k120에서도 재현된다고 보장하지 않음.

### 후보 B. STEP 인덱스를 집계·평활화에도 공유 — 후순위

- 현재 STEP은120-NN에만 사용하고,128/12-NN은 CuPy KD-tree에서 처리함.
- 하나의 구조로 통일하면 별도 GPU KD-tree 구성 약16.180ms를 줄일 여지는 있음. 그러나128/12 조회가 느려지면 이득이 사라짐.
- 현재 STEP 커널은 쿼리=원본 점, k120 고정임. k128/12와 외부 쿼리 배열을 지원하도록 별도 일반화해야 함.
- 반대로 kNN 목록을 무조건128개까지 늘리는 것은 이미 실패한 실험과 유사하므로 구분해야 함.
- CPU 법선 계산까지 자동으로 공유되는 것도 아님. 복잡도에 비해 우선 제거 가능한 구조 구성 비용이 A보다 작아 후순위로 둠.

### 후보 C. 법선까지 GPU 내부에서 이웃 재사용 — 보류

- Faster-PFH는 법선·특징 추출의 이웃 검색을 공유하는 선행연구임 [3]. 재사용 자체는 새 기여가 아님.
- 현재의 반경10000/max_nn10 법선과 k120 목록의 앞10개가 같은지, 동점·법선 부호·퇴화 처리까지 확인해야 함.
- 과거 실패는 공분산을 CPU로 돌려보내는 구현의 결과임. GPU 내부 완결이 무조건 느리다는 뜻은 아님.
- 다만 법선의 수치·부호가 바뀌면 FPFH bin과 이상 점수가 달라질 수 있어 ‘정확도 유지’를 우선하는 이번 후보 A보다 검증 위험이 큼.

### 우선 제외할 잘못된 재사용

- 다른 이미지의 KD-tree/이웃 인덱스를 그대로 재사용하지 않음. 같은 제품이어도 좌표·점 순서·결함이 달라질 수 있음.
- 원본 점에 대한120-NN을, 대표점4096개에 대한 점수 보간3-NN으로 그대로 대체하지 않음. 검색 대상 집합이 다름.
- 인접 Morton 셀만 보고 멈추지 않음. 경계 바깥에 더 가까운 점이 없다는 확인 없이 멈추면 근사 탐색으로 바뀜.

## 5. 정확한 결과를 유지하기 위한 조건

공간 분할의 모양이 달라도 정확한 이웃을 찾을 수 있는 구조인지 확인해야 함. 아래는 구현 시 검증할 조건이며 이미 증명·실측을 끝낸 결과가 아님.

1. 모든 점이 인덱스에 포함되고 원본 점 인덱스와 대응함.
2. 각 경계가 소속 점을 보수적으로 포함함. 부동소수 오차로 실제 후보를 잘라내면 안 됨.
3. 현재120번째 거리보다 하한이 **엄격히 큰 경우에만** 가지치기함. 거리 동점 후보를 버리지 않음.
4. 거리 정의·정밀도·동점 시 원본 인덱스 순서를 현재 STEP 기준과 동일하게 유지함.
5. 이웃 순서와 FPFH40/80/120 값이 같은지 검사하고, 통과 후 뱅크·점수·AUROC/AP까지 검증함.

처음 검증할 범위는 정상·결함 capsule 소수 샘플과 작은 점군1종이면 충분함. 반복 순서를 교차하고, 생성 시간·조회 시간·방문 묶음 수·후보 거리 계산량·메모리를 기록해야 함. 검사에 실패하면 전체 카테고리 실행으로 확대하지 않음.

## 6. 관련 연구 메타데이터

### [1] LBVH 기반 GPU kNN

| Item | Content |
|---|---|
| Title | Optimizing LBVH-Construction and Hierarchy-Traversal to accelerate kNN Queries on Point Clouds using the GPU |
| Authors | Johannes Jakob, Michael Guthe |
| Conference / Journal | Computer Graphics Forum, 40(1), pp. 124–137 |
| Year | 2021 (online first 2020) |
| Paper link | [출판사](https://onlinelibrary.wiley.com/doi/10.1111/cgf.14177), [공개 원문](https://d-nb.info/1227444893/34) |
| GitHub / Official code | 저자 공식 구현 unverified. [cupy-knn](https://github.com/mortacious/cupy-knn)은 README가 해당 논문에 inspired라고 명시한 구현이며 공식 코드라고 단정하지 않음 |
| Reason for investigation | 반복 군집화 대신 공간 계층 생성과 정확한GPU kNN을 함께 검토하기 위함 |

논문은 공간 가속 구조의 생성과 계층 탐색을 최적화해 정확한 점군 이웃 문제를 다룸. 현재의 간단한 STEP 군집 커널과 같은 알고리즘은 아님. 구조 생성·조회 양쪽 비용을 보자는 근거로 사용함. [원문](https://d-nb.info/1227444893/34)

### [2] STEP

| Item | Content |
|---|---|
| Title | A General-Purpose K-Nearest Neighbor Method with an Efficient Pruning Strategy for GPUs |
| Authors | Jue Wang, Fumihiko Ino |
| Conference / Journal | Journal of Parallel and Distributed Computing, 207, 105187 |
| Year | 2026 journal issue; online2025 |
| Paper link | [DOI](https://doi.org/10.1016/j.jpdc.2025.105187), [저자 기관 원문](https://ir.library.osaka-u.ac.jp/repo/ouka/all/103516/) |
| GitHub / Official code | unverified — 현재 로컬 코드는 논문에서 영감을 얻은 독립 구현임 |
| Reason for investigation | 현재 인덱스·하한 가지치기의 출처와 독립 구현의 비용을 구분하기 위함 |

정확한 kNN을 대상으로 하는 범용 GPU 탐색 연구임. 우리 로컬의512군집·50회 갱신이 모든 경우의 최적 설정이거나 논문 전체 구현과 같다는 뜻은 아님. 로컬 코드의 과도한 준비 비용을 논문 자체의 한계로 일반화하지 않음. [논문](https://doi.org/10.1016/j.jpdc.2025.105187)

### [3] KISS-Matcher / Faster-PFH

| Item | Content |
|---|---|
| Title | KISS-Matcher: Fast and Robust Point Cloud Registration Revisited |
| Authors | Hyungtae Lim, Daebeom Kim, Gunhee Shin, Jingnan Shi, Ignacio Vizzo, Hyun Myung, Jaesik Park, Luca Carlone |
| Conference / Journal | ICRA |
| Year | 2025 (arXiv2024) |
| Paper link | [저자 원문](https://arxiv.org/html/2409.15615v2) |
| GitHub / Official code | [공식 저장소](https://github.com/MIT-SPARK/KISS-Matcher) |
| Reason for investigation | 법선·FPFH 이웃 검색을 공유하는 기존 원리와 현재 적용 제약 확인 |

원문§3.3은 큰 반경의 검색 결과를 법선·FPFH 계산에 재사용하고 불안정한 점을 필터링함. 전체 파이프라인을 그대로 도입하면 반경 규칙과 유지되는 점 집합도 달라질 수 있으므로, 현재의 k기반 FPFH를 유지하는 실험과 동일하지 않음. [원문§3.3](https://arxiv.org/html/2409.15615v2)

## 7. 논문 관점에서의 판단

- 후보 A는 ‘정확한 FPFH를 유지하면서 큰 준비 비용을 줄일 수 있는가’를 확인하기 위한 구현·연산 후보임. Morton 정렬·공간 분할·인덱스 공유는 기존 연구이므로, 성공해도 그 자체로 새로운 논문 기여라고 주장하지 않음.
- 이 단계에서 얻을 수 있는 것은 빠르고 정확한 기준선과 비용 원인에 대한 근거임. 이후 과제에 특화된 새로운 원리와 기존 방법 대비 차이를 따로 확보해야 함.
- 검토 요청 범위를 지켜 새 코드 변경·설치·실험 실행·GitHub 업로드는 하지 않음. 문서만 생성하고 Obsidian 탐색 목록을 갱신함.

## 참고문헌

[1] Jakob, Johannes, and Michael Guthe. “Optimizing LBVH-Construction and Hierarchy-Traversal to accelerate kNN Queries on Point Clouds using the GPU.” *Computer Graphics Forum*, vol. 40, no. 1, 2021, pp. 124–137.

[2] Wang, Jue, and Fumihiko Ino. “A General-Purpose K-Nearest Neighbor Method with an Efficient Pruning Strategy for GPUs.” *Journal of Parallel and Distributed Computing*, vol. 207, 2026, article 105187.

[3] Lim, Hyungtae, et al. “KISS-Matcher: Fast and Robust Point Cloud Registration Revisited.” *IEEE International Conference on Robotics and Automation*, 2025.

확인 범위: 현재 코드와 과거 로컬 실험 보고, 논문 방법 설명·출판 정보·공식 프로젝트 페이지를 대조함. 논문 전체 구현의 재현이나 모든 후속 연구에 대한 전수 조사는 아님.
