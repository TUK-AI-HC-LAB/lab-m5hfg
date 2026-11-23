# W48 Simple3D — capsule의 SHOT 지역 특징과 FPFH 비교

## 질문·가설

- 질문: 전체 점의 FPFH를 집계하는 대신 대표 위치에서 SHOT를 직접 계산하면, 미세 결함 탐지와 비용이 어떻게 달라지는가?
- 가설: 전체 점의 120-NN·SPFH/FPFH를 없애 특징 추출 비용을 줄일 여지가 있지만, 특징 변화로 작은 결함의 위치 성능이 낮아질 수 있음. CPU SHOT·LRF 비용 때문에 더 느려질 가능성도 있음.
- 이는 기존 기술자 교체 기준선이며 새로운 알고리즘을 제안한 결과가 아님.

## 고정·변경 조건

| 구분 | FPFH 기준판 | SHOT 시험판 |
|---|---|---|
| 데이터 | 동일 정상54·테스트130 | 동일 |
| 대표점 | 동일 FlashFPS hybrid4096 | 동일 좌표 SHA256 확인 |
| 특징 | FPFH40/80/80/120 평균132차원 | PCL SHOT352 직접 계산 |
| 특징 이웃 | 집계128-NN + 전체 점 FPFH120-NN | 반경 패치·동일 반경 LRF |
| 법선 | Open3D CPU radius10000/max_nn10 | 동일 |
| 특징 장치 | STEP512+Radix 및 FPFH CUDA | 공식 PCL CPU OMP4 |
| 뱅크 | 정상 데이터로 새5% coreset | 정상 데이터로 별도 새5% coreset |
| 후처리 | 4096 Top-3 보간,1024/12 평활화,상위80 평균 | 동일 |

SHOT는 PCL에서 K-search를 지원하지 않아 반경 기반으로 계산함. 반경은 정상 train0의4096 대표점에서128번째 이웃까지 거리의 중앙값으로 한 번 정하고 고정함. 테스트 mask로 조정하지 않음.
특징 차원·이웃 정의·장치가 달라지는 비교임. 전체 시간 차이를 알고리즘만의 효과로 해석하지 않음. 기존 FPFH 뱅크를 재사용하지 않으며 bitwise 점수 일치를 요구하지 않음.
NaN/0벡터가 발생하면 즉시 실패로 기록하고 대표점을 삭제하거나 임의0값으로 대체하지 않음.

- 고정 SHOT/LRF 반경: 0.201053961 (입력 좌표 단위), 정상 보정 시간 0.713s. 반경 보정은 일회성 준비 비용으로 별도 기록함.

완료한 전체 실행: 2/2. 같은 장치에서 기준판→SHOT 순서로 각1회. 반복 분산은 미검증.

| 버전 | 평균 추론(ms) | p95(ms) | 뱅크 구성(s) | Torch 최대할당(GiB) |
|---|---:|---:|---:|---:|
| fpfh | 1622.795 | 1697.510 | 69.894 | 2.481 |
| shot | 1041.443 | 1133.624 | 39.977 | 2.493 |

## 탐지 지표

| 버전 | 난이도 | 물체 AUROC(%) | 포인트 AUROC(%) | 포인트 AP(%) |
|---|---|---:|---:|---:|
| fpfh | easy | 100.0000 | 98.6466 | 39.7360 |
| fpfh | medium | 95.0000 | 87.7486 | 21.9750 |
| fpfh | hard | 84.5000 | 61.4701 | 6.8575 |
| fpfh | ALL | 93.1667 | 81.8179 | 22.3010 |
| shot | easy | 67.2500 | 59.9874 | 5.5314 |
| shot | medium | 57.7500 | 61.9876 | 2.1218 |
| shot | hard | 53.0000 | 49.4941 | 0.9048 |
| shot | ALL | 59.3333 | 57.2480 | 2.8130 |

- 입력·대표점·후처리 일치 확인. 뱅크 크기: FPFH [11059, 132], SHOT [11059, 352]. 뱅크 tensor 바이트: [5839152, 15571072].

## 결과를 해석할 때의 제한

- 전체 추론은 특징 추출·점수 계산 합계이며 파일 입력·저장·평가 계산 제외. 단계 시간은 중첩되므로 단순 합산하지 않음.
- GPU 전체 메모리는 다른 프로세스 포함, Torch 최대할당은 CuPy/PCL/CPU 메모리 제외. CPU RSS도 자원 표에서 확인해야 함.
- 미세 결함 사례 그림의 percentile은 이미지 내 순위이며 두 기술자의 raw 점수 단위가 같다는 뜻이 아님.
- 반경과 이웃 규칙 차이 때문에 결과가 달라도 원인을 SHOT 표현 하나로 단정하지 않음.

## 원본 기록

- 실행 코드 (GitHub 미첨부: `run_simple3d_shot_comparison.py`) · 실행 스크립트 (GitHub 미첨부: `run_simple3d_shot_comparison.sh`) · PCL 래퍼 (GitHub 미첨부: `simple3d_shot_pcl.cpp`)
- [진행 상태](../source/results/capsule_shot_vs_fpfh_20261119/progress.json) · [고정 반경](../source/results/capsule_shot_vs_fpfh_20261119/shot_config.json) · [계획·소스 해시](../source/results/capsule_shot_vs_fpfh_20261119/plan.json)
- [단계별 시간](../source/results/capsule_shot_vs_fpfh_20261119/stage_comparison.csv) · [자원 사용량](../source/results/capsule_shot_vs_fpfh_20261119/resource_comparison.csv) · [지표 표](../source/results/capsule_shot_vs_fpfh_20261119/metrics_comparison.csv)
- [동일 조건 검증](../source/results/capsule_shot_vs_fpfh_20261119/verification.json) · 미세 결함 사례 이미지 미생성
- [공식 PCL SHOT](https://pointclouds.org/documentation/shot_8hpp_source.html). 원본 PCL·데이터·모델·가상환경은 저장소에 복사하지 않음.
