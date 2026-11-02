# W45 Simple3D — 논문과 실제 실행 장치 비교

## Paper Metadata

| Item | Content |
|---|---|
| Title | Towards High-Resolution 3D Anomaly Detection: A Scalable Dataset and Real-Time Framework for Subtle Industrial Defects |
| Authors | Yuqi Cheng, Yihan Sun, Hui Zhang, Weiming Shen, Yunkang Cao |
| Conference / Journal | AAAI |
| Year | 2026 |
| Paper link | https://doi.org/10.1609/aaai.v40i5.37328 |
| GitHub / Official code | https://github.com/hustCYQ/MiniShift-Simple3D |
| Reason for investigation | 논문 명시 장치는 논문을 따르고, 미명시 단계는 공식 구현체 장치를 유지한 capsule 재측정 |

## 비교 범위

**논문에 실행 장치가 명시된 단계는 논문을 따르고, 명시되지 않은 단계는 공식 구현체의 CPU/GPU 배치를 유지함.** 이전에 GPU로 변경했던 법선 계산은 공식 CPU 경로로 복원함. FPFH는 논문의 CUDA 가속 설명에 맞춰 GPU에서 계산함.

- 실행: `bash method12/source/run_capsule_devices.sh --paper-device-policy`
- 현재 결과 폴더: `method12/source/results/capsule_paper_device_policy_20261102/`.
- 실행 완료: 정상 54개로 메모리뱅크 구성, 테스트 130개 전체 평가와 최종 지표 산출 완료. 전체 소요 시간 4,804.91초, 약 1시간 20분 5초임.

## 실제 실행 측정 결과 — capsule

capsule 정상 점군 54개로 새 메모리뱅크를 구성하고 테스트 점군 130개 전체 평가 완료.

| 물체 AUROC (%) | 포인트 AUROC (%) | 학습·메모리뱅크 구성 시간 | 평균 추론 (ms/샘플) | 최대 GPU 메모리 (GB)¹ | 총 저장 용량 (GB)² |
|---:|---:|---:|---:|---:|---:|
| 91.50 | 81.73 | 22.97분 | 25,729.80 | 11.40 | 20.87 |

평균 추론은 **테스트 샘플 1개당** CPU 법선·CUDA FPFH·특징 집계·검색·보간·점수 수집 시간임. 입력 파일 로딩·결과 저장·AUROC 계산·정상 메모리뱅크 구성은 제외함. 실험 전체 소요 시간과 구분함.

## 논문과 실제 실행 device 비교

**선택 원칙:** 논문 명시 장치를 우선하고 미명시 단계는 공개 코드의 장치를 유지함. ‘미명시’는 임의 장치 선택이 아니라 **공식 구현체를 따르는 기준**임. 공개 코드는 upstream `5951e097ed5f6c52c1e058490f8c64f9b6323501` 기준임. 아래 실행 배치는 완료된 실행의 설정·호출 기록을 기준으로 함.

| 단계 | 논문에 명시된 device | 공개 코드의 경로 | 이번 실제 실행 | 비교 판단 |
|---|---|---|---|---|
| 실험 GPU | NVIDIA A100 | 특정 GPU 모델을 강제하지 않음 | NVIDIA GeForce RTX 5080 | **장비 다름** |
| 데이터 로딩·중심 정렬 | 단계별 미명시 | CPU — NumPy | CPU — 공식 로더 | 코드 배치 유지. 논문과의 장치 일치는 확인 불가 |
| 법선 계산·법선용 이웃 탐색 | 단계별 미명시 | CPU — Open3D legacy | **CPU — 공식 Open3D legacy**, radius=10000, max_nn=10 | **공식 코드 유지**. 법선 결과만 CUDA FPFH 입력으로 GPU 전송 |
| FPFH 40 이웃 스케일 | **FPFH에 CUDA 가속 Open3D 사용 명시** | CPU에서 계산한 뒤 결과를 GPU로 전송 | **GPU — Open3D tensor CUDA** | **논문 명시 장치 우선**. 공식 CPU FPFH를 CUDA 함수로 변경. 논문과 동일 버전·내부 계산 코드까지 확인된 것은 아님 |
| FPFH 80 이웃 스케일 | **동일한 CUDA FPFH 설명**, 스케일 80 명시 | CPU에서 계산한 뒤 결과를 GPU로 전송 | **GPU — Open3D tensor CUDA** | **논문 명시 장치 우선** |
| FPFH 120 이웃 스케일 | **동일한 CUDA FPFH 설명**, 스케일 120 명시 | CPU에서 계산한 뒤 결과를 GPU로 전송 | **GPU — Open3D tensor CUDA** | **논문 명시 장치 우선** |
| FPFH 내부 좌표 이웃 탐색 | CUDA FPFH 설명에 포함, 내부 장치 별도 기술 없음 | CPU FPFH 함수 내부 탐색 | CUDA FPFH 함수 내부 GPU 탐색 | FPFH 전체를 CUDA 함수로 변경한 데 포함됨. 독립 단계로 떼어 장치를 임의 변경한 것이 아님 |
| 스케일 특징 결합 | 단계별 미명시 | GPU — torch.cat | GPU — DLPack 공유 후 torch.cat | 코드 배치 유지. 논문과의 장치 일치는 확인 불가 |
| 특징 집계용 FPS·gather | 단계별 미명시 | GPU — PointNet2 CUDA | GPU — native PointNet2 CUDA | 코드의 GPU 배치 유지. 논문은 LFSA 중심 선택을 무작위 샘플링으로 설명하므로 FPS 알고리즘까지 같다고 주장하지 않음 |
| 특징 집계용 좌표 이웃 탐색 | 단계별 미명시 | GPU — KNN_CUDA | GPU — native KNN_CUDA, k=128 | 코드 배치 유지. 논문과의 장치 일치는 확인 불가 |
| LFSA 평균 집계 | 단계별 미명시 | GPU — torch.mean | GPU — torch.mean | 코드 배치 유지. 논문과의 장치 일치는 확인 불가 |
| coreset 임시 투영 시도 | 단계별 미명시 | CPU — sklearn | CPU 투영 시도 후 조건 불충족으로 원래 특징 사용 | 공식 예외 처리 유지. 이번 실행에는 투영 미적용 |
| coreset 대표 특징 선택 | 단계별 미명시 | CPU 준비 후 CUDA 사용 가능 시 GPU 선택 | CPU 준비 후 GPU float16 거리 갱신·선택 | 코드 배치 유지. 논문이 float16·이 device 배치를 명시한 것은 아님 |
| 메모리뱅크 최근접 이웃 검색 | 거리 기반 점수는 설명하지만 device는 미명시 | GPU — torch.cdist + min | GPU — torch.cdist + min | 코드 배치 유지. 논문과의 장치 일치는 확인 불가 |
| 포인트 점수 1·2차 보간 | 단계별 미명시 | GPU — 전체 정렬 기반 보간 | GPU — 공식 전체 정렬 기반 보간 | 코드 배치 유지. 논문과의 장치 일치는 확인 불가 |
| 점수 평활화용 FPS·이웃 탐색·평균 | 단계별 미명시 | GPU | GPU — 중심 1,024개, k=12 | 코드 배치 유지. 이 세부 처리의 원문 일치까지 확인된 것은 아님 |
| 점수맵 재배열 | 단계별 미명시 | CPU 결과를 CUDA 인덱스로 처리하는 혼용 부분 존재 | CPU 왕복 후 GPU 복귀 | 실행 오류를 막는 장치 수정 적용. 원본 그대로 실행한 것은 아님 |
| 최종 물체 점수 | device 미명시. 본문 수식은 최대 점수 | GPU에서 상위 80점 평균 | GPU에서 상위 80점 평균 | 코드 배치 유지. **장치와 별개로 점수 집계 수식은 논문과 다름** |
| AUROC 계산 | 지표는 명시하지만 device는 미명시 | CPU — sklearn | CPU — sklearn | 코드 배치 유지. 논문과의 장치 일치는 확인 불가 |

### 비교 결론

- **논문을 따른 항목:** FPFH CUDA 가속 사용. 세 스케일을 CUDA 함수로 계산하며 단순히 CPU 계산 결과를 GPU에 올리는 방식이 아님.
- **공식 코드를 따른 항목:** 법선은 CPU, FPS·집계용 KNN·LFSA·보간·메모리뱅크 검색은 GPU. 이전 GPU 법선을 CPU로 복원함. 미명시 단계를 임의로 전부 GPU에 배치하지 않음.
- **명확히 다른 항목:** 논문은 A100, 이번 실행은 RTX 5080임. 따라서 같은 GPU에서 측정한 속도·메모리 비교가 아님.
- **해석 범위:** 장치 선택 기준을 맞춘 재측정이며, 미명시 단계는 공식 구현체를 따른 것임. 모든 단계의 장치가 논문과 동일하다고 확인한 것은 아님. CUDA FPFH는 공식 CPU 법선을 전송해 사용하며 Float32·내부 최근접 탐색의 수치 차이가 있을 수 있음.
- **별도로 남아 있는 구현 차이:** FPS와 본문 무작위 샘플링 설명, 스케일 중복 연결, 물체 점수 집계 및 장치 혼용 수정. **‘FPFH의 CUDA 사용을 맞춤’과 ‘모든 알고리즘·환경을 논문과 동일하게 맞춤’은 다름.**

## 실행과 근거

- 현재 실행 설정·소스 해시: `method12/source/results/capsule_paper_device_policy_20261102/environment.json`.
- 단계별 시간·장치: 같은 폴더의 `stage_timings.csv`, `device_audit.json`, `stage_summary.csv`.
- 실행 코드: `method12/source/benchmark_capsule_devices.py`, `method12/source/simple3d_paper_cuda_fpfh.py`.
- 정상 1개·테스트 2개 소규모 검증 통과: `method12/source/results/capsule_paper_device_policy_20261102_smoke/smoke.json`. 같은 폴더의 `device_audit.json`에서 CPU 법선·CUDA FPFH 호출 확인.
- 논문 근거: PDF 4쪽(인쇄 쪽수 3330)의 Experimental Setups—Implementation Details에서 CUDA 가속 FPFH, 스케일 40/80/120, A100 명시. 같은 쪽 LFSA·Anomaly Detection에서 샘플링과 점수 계산 설명.
- [공식 원문](https://ojs.aaai.org/index.php/AAAI/article/download/37328/41290) · [공개 코드 FPFH](https://github.com/hustCYQ/MiniShift-Simple3D/blob/5951e097ed5f6c52c1e058490f8c64f9b6323501/feature_extractors/FPFH.py) · [공개 코드 특징·점수 계산](https://github.com/hustCYQ/MiniShift-Simple3D/blob/5951e097ed5f6c52c1e058490f8c64f9b6323501/feature_extractors/features.py).
