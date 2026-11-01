# Simple3D capsule 측정 결과

## Paper Metadata

| Item | Content |
|---|---|
| Title | Towards High-Resolution 3D Anomaly Detection: A Scalable Dataset and Real-Time Framework for Subtle Industrial Defects |
| Authors | Yuqi Cheng, Yihan Sun, Hui Zhang, Weiming Shen, Yunkang Cao |
| Conference / Journal | AAAI |
| Year | 2026 |
| Paper link | https://doi.org/10.1609/aaai.v40i5.37328 |
| GitHub / Official code | https://github.com/hustCYQ/MiniShift-Simple3D |
| Reason for investigation | CUDA FPFH 경로로 실행한 capsule 단일 범주의 성능·시간·메모리·저장 용량을 별도로 정리 |

## 측정 범위

- 상태: **완료**. 실행일: 2026-11-01.
- MiniShift의 **capsule 한 범주 전체**를 평가함. 물체 한 샘플만 검사한 결과가 아님.
- 정상 학습 점군 54개로 새 메모리뱅크 구성, 테스트 점군 130개 전체 평가.
- 점군당 500,000포인트 유지. 평가한 포인트는 총 65,000,000개임.
- GPU: NVIDIA GeForce RTX 5080, WSL Ubuntu.
- 기존 메모리뱅크·테스트 특징·이상 점수는 재사용하지 않음.
- 전체 실행 소요 시간: **5,198.85초, 약 1시간 26분 39초**.

## 결과

GB는 10억 바이트 기준임. AUROC는 높을수록 좋으며, 시간·용량은 이번 로컬 실행에서 측정한 값임.

| 물체 AUROC (%) | 포인트 AUROC (%) | 학습·메모리뱅크 구성 시간 | 평균 추론 (ms/샘플) | 최대 GPU 메모리 (GB) | 총 저장 용량 (GB) |
|---:|---:|---:|---:|---:|---:|
| 91.5000 | 81.0244 | 1,452.621초 (24.21분) | 27,980.113 | 11.4689 | 20.8739 |

### 표의 측정 기준

- **학습·메모리뱅크 구성 시간:** 정상 54개의 로딩·특징 추출·coreset 선택을 포함함. 신경망 optimizer 학습은 없음. warmup과 메모리뱅크 파일 저장은 제외함.
- **평균 추론:** 로딩된 점군의 법선·FPFH 계산부터 특징 집계·메모리뱅크 검색·보간·최종 점수 수집까지 포함함. 입력 파일 로딩·결과 저장·AUROC 계산은 제외함. 약 27.98초/샘플임.
- **최대 GPU 메모리:** Open3D 할당까지 관측하기 위해 100ms 간격으로 확인한 **GPU 장치 전체 사용량의 최대값**임. 화면·드라이버·다른 프로그램도 포함하므로 모델 전용 메모리가 아니며, 관측 간격 사이의 순간 최대값을 보장하지 않음.
- **총 저장 용량:** capsule 데이터와 남아 있는 ZIP, 기본·native 환경, CUDA 도구·native 확장 소스, 공식 코드와 이번 결과를 포함한 논리 파일 크기임. 다른 범주·이전 결과·OS·드라이버·공용 캐시는 제외함. 물리 디스크 증가량과는 다름.

### 보조 측정값

| 항목 | 결과 |
|---|---:|
| 입력 로딩 포함 평균 추론 | 28,224.913 ms/샘플 |
| P95 추론 시간 | 30,039.370 ms/샘플 |
| coreset 선택 시간 | 5.545초 |
| warmup 시간 | 28.196초 |
| PyTorch 최대 allocated GPU 메모리 | 8.4020 GB |
| PyTorch 최대 reserved GPU 메모리 | 8.4389 GB |
| 프로세스 최대 RSS 메모리 | 5.9493 GB |

PyTorch 메모리 값에는 Open3D가 별도로 할당한 GPU 메모리가 포함되지 않음. 따라서 표의 장치 전체 메모리와 직접 같은 값으로 해석하면 안 됨. RSS는 CPU 측 프로세스 메모리임.

## 단계별 실행 장치

| 단계 | 실제 실행 장치·방식 |
|---|---|
| 입력 로딩·중심 정렬 | CPU |
| 법선 계산과 법선용 이웃 탐색 | GPU — Open3D tensor CUDA, max_nn=10 |
| FPFH 스케일별 이웃 탐색·계산 | GPU — Open3D tensor CUDA, max_nn=40/80/120 |
| 특징 전달·결합 | GPU — DLPack으로 Open3D와 PyTorch 사이 직접 공유 |
| 특징 집계용 FPS·gather | GPU — native PointNet2, 중심 4,096개 |
| 특징 집계용 좌표 이웃 탐색 | GPU — native KNN_CUDA, k=128 |
| LFSA 평균 집계 | GPU |
| coreset 임시 투영 시도 | CPU — sklearn. 이번에는 투영 조건 불충족으로 공식 예외 경로에 따라 원래 특징 사용 |
| coreset 대표 특징 선택 | CPU 준비 후 GPU float16 거리 갱신·선택, 5% |
| 메모리뱅크 최근접 이웃 검색 | GPU — torch.cdist + min |
| 포인트 점수 보간 | GPU — 공식 전체 정렬 방식, chunk=50,000 |
| 점수 평활화용 FPS·이웃 탐색·평균 | GPU — 중심 1,024개, k=12 |
| 점수맵 재배열 | CPU 왕복 후 CUDA 인덱스 처리를 위해 GPU로 복귀 |
| 물체 점수 계산 | GPU — 공식 코드의 상위 80개 포인트 점수 평균 |
| 정답과 비교하여 AUROC 계산 | CPU |

법선·FPFH·특징 집계는 warmup 1개, 정상 54개, 테스트 130개를 합쳐 각각 185회 관측됨. 메모리뱅크 검색과 각 보간 단계는 테스트 130개에서 관측됨.

기록상의 주의: 원시 `summary.json`의 `timing_scope` 설명에는 초기 CPU 경로 문구가 남아 있음. 실제 최종 실행의 법선·FPFH는 `environment.json`과 `device_audit.json`, `stage_timings.csv`에 기록된 **CUDA 경로**임. 원시 수치와 기록은 변경하지 않음.

## 논문과의 일치 범위·남은 차이

- 논문의 CUDA 가속 FPFH 설명에 맞춰 공개 코드의 legacy CPU API 대신 **Open3D tensor CUDA FPFH**를 사용함. 초기 CPU FPFH 점검 결과는 이 표에 포함하지 않음.
- 법선·FPFH는 Float32로 처리함. 입력 전체가 기존 반경 안에 있음을 확인한 뒤 max_nn만 지정하는 최근접 탐색을 사용함. legacy CPU API와 수치가 비트 단위로 같다는 주장은 하지 않음.
- 장치 변경 외의 공개 코드 동작은 유지함. FPFH는 40/80/120 이웃 스케일을 계산하지만 공식 코드가 두 번째 스케일을 중복 연결하므로 실제 결합은 **40/80/80/120, 총 132차원**임.
- 물체 점수도 공개 코드의 상위 80점 평균을 유지함. 따라서 논문의 단순 3스케일 연결·최대 점수 수식과 완전히 동일한 구현은 아님.
- RTX 5080용으로 CUDA 확장을 재빌드한 환경이며, 논문의 A100과 동일한 장비·환경에서 측정한 결과가 아님. **이 실행의 완료와 논문 모든 조건의 동일 재현은 구분함.**

## 원시 결과·상세 기록

- [요약 JSON](../source/results/capsule_devices_20261101_paper_cuda_float32/summary.json)
- [표 원시 CSV](../source/results/capsule_devices_20261101_paper_cuda_float32/comparison.csv)
- [환경·설정](../source/results/capsule_devices_20261101_paper_cuda_float32/environment.json)
- [학습 샘플별 시간](../source/results/capsule_devices_20261101_paper_cuda_float32/train_timings.csv)
- [테스트 샘플별 시간·물체 점수](../source/results/capsule_devices_20261101_paper_cuda_float32/test_timings.csv)
- [실제 장치 호출 기록](../source/results/capsule_devices_20261101_paper_cuda_float32/device_audit.json)
- [단계별 시간 집계](../source/results/capsule_devices_20261101_paper_cuda_float32/stage_summary.csv)
- [저장 용량 상세](../source/results/capsule_devices_20261101_paper_cuda_float32/storage.json)
- [기존 상세 측정 문서](Simple3D_capsule_device_benchmark_20261101.md)

원시 포인트 점수·정답·메모리뱅크 경로: `/home/test/simple3d-capsule-devices-20261101-paper-cuda-float32`. 데이터셋·가상환경·메모리뱅크의 새 사본을 만들거나 GitHub에 업로드하지 않음.
