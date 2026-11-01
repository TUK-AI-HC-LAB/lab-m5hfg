# Simple3D — 논문과 실제 실행 장치 비교

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

## 비교 범위

MiniShift capsule 측정의 장치 배치를 비교함. 논문에 명시된 조건, 공개 코드의 동작, 실제 실행 장치를 구분함. 성능·시간·용량 표는 [별도 측정 결과](Simple3D_capsule_measurement_results_20261101.md)에 보관함.

## 논문과 실제 실행 device 비교

**비교 원칙:** ‘논문에 미명시’는 CPU라는 뜻이 아니라, 해당 단계의 CPU/GPU 배치를 논문 본문만으로 확정할 수 없다는 뜻임. 공개 코드의 배치와 논문에 적힌 조건을 구분함. 아래 공개 코드는 upstream `5951e097ed5f6c52c1e058490f8c64f9b6323501` 기준이며, 실제 실행은 환경 설정·장치 호출 기록과 실행 코드로 확인함.

| 단계 | 논문에 명시된 device | 공개 코드의 경로 | 이번 실제 실행 | 비교 판단 |
|---|---|---|---|---|
| 실험 GPU | NVIDIA A100 | 특정 GPU 모델을 강제하지 않음 | NVIDIA GeForce RTX 5080 | **장비 다름** |
| 데이터 로딩·중심 정렬 | 단계별 미명시 | CPU — NumPy | CPU — 공식 로더 | 코드 배치 유지. 논문과의 장치 일치는 확인 불가 |
| 법선 계산·법선용 이웃 탐색 | 단계별 미명시 | CPU — Open3D legacy | **GPU — Open3D tensor CUDA**, max_nn=10 | 코드의 CPU 경로를 GPU로 변경. 논문이 GPU 법선을 명시했다고 볼 수는 없음 |
| FPFH 40 이웃 스케일 | **FPFH에 CUDA 가속 Open3D 사용 명시** | CPU에서 계산한 뒤 결과를 GPU로 전송 | **GPU — Open3D tensor CUDA** | **CUDA 사용이라는 명시 조건 일치**. 동일 API·커널이라는 뜻은 아님 |
| FPFH 80 이웃 스케일 | **동일한 CUDA FPFH 설명**, 스케일 80 명시 | CPU에서 계산한 뒤 결과를 GPU로 전송 | **GPU — Open3D tensor CUDA** | **CUDA 사용이라는 명시 조건 일치** |
| FPFH 120 이웃 스케일 | **동일한 CUDA FPFH 설명**, 스케일 120 명시 | CPU에서 계산한 뒤 결과를 GPU로 전송 | **GPU — Open3D tensor CUDA** | **CUDA 사용이라는 명시 조건 일치** |
| FPFH 내부 좌표 이웃 탐색 | FPFH 내부 탐색의 device는 별도 미명시 | CPU — Open3D legacy 탐색 | GPU — Open3D tensor 최근접 탐색 | 전체 FPFH CUDA 경로 안에서 수행. 내부 탐색까지 원문과 동일한지는 확인 불가 |
| 스케일 특징 결합 | 단계별 미명시 | GPU — torch.cat | GPU — DLPack 공유 후 torch.cat | 코드 배치 유지. 논문과의 장치 일치는 확인 불가 |
| 특징 집계용 FPS·gather | 단계별 미명시 | GPU — PointNet2 CUDA | GPU — native PointNet2 CUDA | 코드의 GPU 배치 유지. 논문은 LFSA 중심 선택을 무작위 샘플링으로 설명하므로 FPS 알고리즘까지 같다고 주장하지 않음 |
| 특징 집계용 좌표 이웃 탐색 | 단계별 미명시 | GPU — KNN_CUDA | GPU — native KNN_CUDA, k=128 | 코드 배치 유지. 논문과의 장치 일치는 확인 불가 |
| LFSA 평균 집계 | 단계별 미명시 | GPU — torch.mean | GPU — torch.mean | 코드 배치 유지. 논문과의 장치 일치는 확인 불가 |
| coreset 임시 투영 시도 | 단계별 미명시 | CPU — sklearn | CPU에서 시도했으나 투영 조건 불충족으로 원래 특징 사용 | 코드 예외 경로 유지. 투영이 적용된 실행은 아님 |
| coreset 대표 특징 선택 | 단계별 미명시 | CPU 준비 후 CUDA 사용 가능 시 GPU 선택 | CPU 준비 후 GPU float16 거리 갱신·선택 | 코드 배치 유지. 논문이 float16·이 device 배치를 명시한 것은 아님 |
| 메모리뱅크 최근접 이웃 검색 | 거리 기반 점수는 설명하지만 device는 미명시 | GPU — torch.cdist + min | GPU — torch.cdist + min | 코드 배치 유지. 논문과의 장치 일치는 확인 불가 |
| 포인트 점수 1·2차 보간 | 단계별 미명시 | GPU — 전체 정렬 기반 보간 | GPU — 공식 전체 정렬 기반 보간 | 코드 배치 유지. 논문과의 장치 일치는 확인 불가 |
| 점수 평활화용 FPS·이웃 탐색·평균 | 단계별 미명시 | GPU | GPU — 중심 1,024개, k=12 | 코드 배치 유지. 이 세부 처리의 원문 일치까지 확인된 것은 아님 |
| 점수맵 재배열 | 단계별 미명시 | CPU 결과를 CUDA 인덱스로 처리하는 혼용 부분 존재 | CPU 왕복 후 GPU 복귀 | 실행 오류를 막는 장치 수정 적용. 원본 그대로 실행한 것은 아님 |
| 최종 물체 점수 | device 미명시. 본문 수식은 최대 점수 | GPU에서 상위 80점 평균 | GPU에서 상위 80점 평균 | 코드 배치 유지. **장치와 별개로 점수 집계 수식은 논문과 다름** |
| AUROC 계산 | 지표는 명시하지만 device는 미명시 | CPU — sklearn | CPU — sklearn | 코드 배치 유지. 논문과의 장치 일치는 확인 불가 |

### 비교 결론

- **맞춘 항목:** 논문이 명시한 FPFH CUDA 가속 사용. 세 스케일 모두 실제 GPU 계산으로 확인함. 단순히 CPU 계산 결과를 GPU에 올린 것이 아님.
- **명확히 다른 항목:** 논문은 A100, 이번 실행은 RTX 5080임. 따라서 같은 GPU에서 측정한 속도·메모리 비교가 아님.
- **논문만으로 알 수 없는 항목:** 법선·좌표 이웃 탐색·FPS·LFSA·보간·coreset·메모리뱅크 검색 등의 세부 device 배치. 이번 실제 장치는 기록했지만, 이를 모두 ‘논문과 일치’로 표시하지 않음.
- **별도로 남아 있는 구현 차이:** FPS와 본문 무작위 샘플링 설명, 스케일 중복 연결, 물체 점수 집계 및 장치 혼용 수정. **‘FPFH의 CUDA 사용을 맞춤’과 ‘모든 알고리즘·환경을 논문과 동일하게 맞춤’은 다름.**

근거: 논문 PDF 4쪽(인쇄 쪽수 3330)의 Experimental Setups—Implementation Details에서 CUDA 가속 FPFH, 스케일 40/80/120, A100을 명시함. 같은 쪽 LFSA·Anomaly Detection에서 샘플링과 점수 계산을 설명함. [공식 원문](https://ojs.aaai.org/index.php/AAAI/article/download/37328/41290) · [공개 코드 FPFH](https://github.com/hustCYQ/MiniShift-Simple3D/blob/5951e097ed5f6c52c1e058490f8c64f9b6323501/feature_extractors/FPFH.py) · [공개 코드 특징·점수 계산](https://github.com/hustCYQ/MiniShift-Simple3D/blob/5951e097ed5f6c52c1e058490f8c64f9b6323501/feature_extractors/features.py) · [실제 장치 호출 기록](../source/results/capsule_devices_20261101_paper_cuda_float32/device_audit.json)
