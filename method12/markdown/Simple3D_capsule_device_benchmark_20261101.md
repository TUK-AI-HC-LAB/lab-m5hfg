# Simple3D capsule — 단계별 실행 장치 확인과 새 측정

## Paper Metadata

| Item | Content |
|---|---|
| Title | Towards High-Resolution 3D Anomaly Detection: A Scalable Dataset and Real-Time Framework for Subtle Industrial Defects |
| Authors | Yuqi Cheng, Yihan Sun, Hui Zhang, Weiming Shen, Yunkang Cao |
| Conference / Journal | AAAI |
| Year | 2026 |
| Paper link | https://doi.org/10.1609/aaai.v40i5.37328 |
| GitHub / Official code | https://github.com/hustCYQ/MiniShift-Simple3D |
| Reason for investigation | capsule 범주에서 native 실행 장치를 확인하고 성능·시간·메모리·저장 용량을 새로 측정 |

## 범위와 결과

정상 54개로 새 메모리뱅크 구성, 테스트 130개 전체 평가. 각 점군 500,000점 유지. 기존 메모리뱅크·특징·점수 재사용 없음. 이전 결과는 보존함.

| 물체 AUROC (%) | 포인트 AUROC (%) | 학습·메모리뱅크 구성 시간 | 평균 추론 (ms/샘플) | 최대 GPU 메모리 (GB) | 총 저장 용량 (GB) |
|---:|---:|---:|---:|---:|---:|
| 91.5000 | 81.0244 | 1452.621초 (24.21분) | 27980.113 | 11.4689 | 20.874 |

## 실행 장치

| 단계 | 실제 장치 / 방식 |
|---|---|
| 로딩·중심 정렬 | CPU, 공식 로더 |
| 법선·법선용 좌표 이웃 탐색 | GPU, Open3D tensor CUDA, max_nn=10; 반경이 전체 점군을 포함함을 검증 |
| FPFH 스케일별 이웃 탐색·계산 | GPU, Open3D tensor CUDA, max_nn=40/80/120; CPU fallback 금지 |
| 특징 결합·장치 전송 | CUDA tensor에서 DLPack으로 GPU 직접 공유, 결합 차원 132 |
| 특징 집계용 FPS / gather | GPU, native PointNet2, 중심 4096개 |
| 특징 집계용 좌표 이웃 탐색 | GPU, native KNN_CUDA, k=128 |
| LFSA 평균 집계 | GPU |
| coreset 임시 투영 | CPU, sklearn. 투영 불가 시 원본 코드의 예외 경로 유지 |
| coreset farthest-first 선택 | CPU 준비 후 GPU float16 거리 갱신·선택, 공식 5% |
| 메모리뱅크 NN | GPU, torch.cdist + min |
| 포인트 점수 1차 보간 | GPU, 공식 전체 정렬, chunk=50000, 최근접 3개 가중합 |
| 점수 평활화용 FPS·이웃·평균 | GPU, 중심 1024개, k=12 |
| 포인트 점수 2차 보간 | GPU, 공식 전체 정렬, chunk=50000 |
| 점수맵 재배열 | 공식 CPU 왕복 후 CUDA 인덱스 오류 방지를 위한 GPU 복귀 |
| 최종 물체 점수·평가 | 공식 상위 80점 평균은 GPU, 정답 비교·AUROC는 CPU |

법선 내부 이웃 탐색과 FPFH 내부 이웃 탐색은 Open3D 함수에 포함되어 별도 커널 시간으로 분리하지 않음. device_audit.json과 stage_timings.csv에 호출 시 확인한 장치·입출력 크기·횟수를 남김.

## 측정 해석

- 논문의 CUDA 가속 FPFH 설명에 맞춰 legacy CPU API를 Open3D tensor CUDA FPFH로 교체함. GPU 법선·FPFH 수치가 legacy CPU 결과와 비트 단위 동일하다고 주장하지 않음. 좌표 kNN은 CPU KD-tree 대체가 아니라 native CUDA 경로 사용. 보간도 topk 대체가 아닌 공식 full-sort 사용.
- 공식 MSND 코드는 40/80/120 스케일을 계산하지만 특징은 40/80/80/120으로 결합함. 이 중복을 그대로 유지하여 차원은 132임.
- Simple3D에는 신경망 optimizer 학습이 없음. 표의 학습 시간은 전체 정상 로딩·특징 추출·5% coreset 선택까지의 새 메모리뱅크 구성 시간임. warmup·뱅크 파일 저장 제외, 진행 기록 오버헤드는 포함함.
- 평균 추론은 입력이 로딩된 뒤 법선/FPFH 계산부터 최종 점수 수집까지 포함함. 데이터 읽기·결과 압축 저장·AUROC 계산은 제외함. CUDA 동기화된 wall time이며 단계별 CUDA event 관찰에 따른 소량의 오버헤드가 있음.
- 최대 GPU 메모리는 Open3D 할당도 포함하기 위해 100ms 주기로 관측한 장치 전체 사용량의 최대값임. 드라이버·화면·다른 프로그램도 포함하므로 모델 전용 메모리로 해석하면 안 됨. PyTorch allocated/reserved 값은 summary.json에 별도 기록함. GB=10^9 bytes이며 샘플링 사이 순간 최대값은 놓칠 수 있음.
- 총 저장 용량은 capsule 데이터(남아 있는 ZIP 포함), 기본·native 환경, native toolkit·소스, 공식 checkout, 이번 원시 결과의 논리 파일 크기임. 중복 hardlink는 한 번만 계산함. 다른 카테고리·과거 결과·OS/드라이버·공용 캐시는 제외함. 물리 디스크 증가량과는 다름.
- 구형 Torch/CUDA 바이너리를 그대로 실행한 것이 아니라 RTX5080용으로 재빌드한 로컬 측정임. 데이터 경로 오류·점수맵 장치 혼용 우회 유지. 과거 호환 백엔드 결과나 논문의 FPS와 동일 조건으로 단정하지 않음.

## 근거와 실행

- 실행: `bash method12/source/run_capsule_devices.sh --paper-cuda-fpfh`
- [실행 코드](../source/benchmark_capsule_devices.py)
- [설정·소스 해시](../source/results/capsule_devices_20261101_paper_cuda_float32/environment.json)
- [전체 샘플 목록](../source/results/capsule_devices_20261101_paper_cuda_float32/manifest.json)
- [실제 장치 호출 기록](../source/results/capsule_devices_20261101_paper_cuda_float32/device_audit.json)
- [단계별 원시 시간](../source/results/capsule_devices_20261101_paper_cuda_float32/stage_timings.csv)
- [단계별 시간 집계](../source/results/capsule_devices_20261101_paper_cuda_float32/stage_summary.csv)
- [학습 시간](../source/results/capsule_devices_20261101_paper_cuda_float32/train_timings.csv)
- [추론 시간과 물체 점수](../source/results/capsule_devices_20261101_paper_cuda_float32/test_timings.csv)
- [요약 JSON](../source/results/capsule_devices_20261101_paper_cuda_float32/summary.json)
- [표 원시 CSV](../source/results/capsule_devices_20261101_paper_cuda_float32/comparison.csv)
- [저장 용량 상세](../source/results/capsule_devices_20261101_paper_cuda_float32/storage.json)
- 원시 포인트 점수·정답 및 메모리뱅크는 summary.json의 raw_assets 경로에 보관함. 데이터셋·가상환경·메모리뱅크를 GitHub에 업로드하지 않음.

PyTorch 최대 allocated: 8.4020 GB, reserved: 8.4389 GB. 장치 초기 사용량: 1.4151 GB.

논문이 명시하지 않은 세부 장치 배치는 공개 코드로 보완함. Open3D tensor는 모든 점이 기존 반경 안에 있음을 확인한 뒤 max_nn만 사용하여 같은 최근접 이웃 목적을 계산함. GPU 수치·동률 처리 차이가 있을 수 있음. 공식 코드의 중복 스케일과 물체 top-80 집계는 장치 변경 외의 별도 차이로 남겨 두므로 논문 모든 수식과 완전히 동일하다는 주장은 하지 않음.
