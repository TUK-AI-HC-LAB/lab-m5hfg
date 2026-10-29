# Simple3D: 약한 카테고리와 느린 처리 구간

<details>
<summary>Paper Metadata</summary>

| Item | Content |
|---|---|
| Title | Towards High-Resolution 3D Anomaly Detection: A Scalable Dataset and Real-Time Framework for Subtle Industrial Defects |
| Authors | Yuqi Cheng, Yihan Sun, Hui Zhang, Weiming Shen, Yunkang Cao |
| Conference / Journal | AAAI 2026, 40(5), 3327–3334 |
| Year | 2026 |
| Paper link | https://doi.org/10.1609/aaai.v40i5.37328 |
| GitHub / Official code | https://github.com/hustCYQ/MiniShift-Simple3D |
| Reason for investigation | 로컬 전체 카테고리 실행에서 저성능 구간과 느린 처리 구간 식별 |

</details>

**기준: 2026-10-28에 완료한 로컬 52개 카테고리 실행.** 로컬 AUROC를 논문 보고값과 비교함. 차이는 로컬−논문(%p)이며, 양수는 로컬 값이 더 높다는 뜻임. 이 실행은 CPU cKDTree 좌표 탐색을 사용한 호환 구현이며, 현재 진행 중인 원본 CUDA 연산 재실행과 구분함.

## 1. 어느 카테고리에서 약한가?

각 데이터셋에서 물체 판정·위치 탐지 AUROC가 낮은 카테고리 3개씩을 정리함. MiniShift 카테고리 수치는 난이도 ALL 기준임.

### 물체 AUROC가 낮은 카테고리

| 데이터셋 | 카테고리 | 논문 물체 AUROC (%) | 로컬 물체 AUROC (%) | 차이 (%p) |
|---|---|---:|---:|---:|
| MiniShift | **screw** | 23.20 | **22.42** | −0.78 |
| MiniShift | toothbrush | 66.10 | 52.08 | −14.02 |
| MiniShift | light | 54.20 | 59.17 | +4.97 |
| Anomaly-ShapeNet | **headset1** | 95.70 | **66.67** | −29.03 |
| Anomaly-ShapeNet | tap1 | 59.60 | 67.78 | +8.18 |
| Anomaly-ShapeNet | helmet3 | 65.50 | 74.85 | +9.35 |

### 포인트 AUROC가 낮은 카테고리

| 데이터셋 | 카테고리 | 논문 포인트 AUROC (%) | 로컬 포인트 AUROC (%) | 차이 (%p) |
|---|---|---:|---:|---:|
| MiniShift | **toothbrush** | 45.50 | **45.57** | +0.07 |
| MiniShift | screen | 57.50 | 57.33 | −0.17 |
| MiniShift | flat_pad | 61.80 | 61.01 | −0.79 |
| Anomaly-ShapeNet | **bucket0** | 72.50 | **70.68** | −1.82 |
| Anomaly-ShapeNet | bottle1 | 72.80 | 73.07 | +0.27 |
| Anomaly-ShapeNet | bottle3 | 83.80 | 80.84 | −2.96 |

논문 기준: MiniShift ALL은 부록 Table 7, Anomaly-ShapeNet은 부록 Table 12. [공개 원문](https://arxiv.org/html/2507.07435v1). 로컬 하위 카테고리만 선정한 표이므로 논문 전체의 하위 순위와는 다를 수 있음.
AUROC는 정확도가 아님. 50% 미만인 경우 점수 순위 방향까지 점검할 대상이며, 이 수치만으로 오류 원인을 확정하지 않음.

### MiniShift는 어느 난이도에서 약한가?

12개 카테고리의 단순 평균임.

| 난이도 | 논문 물체 AUROC (%) | 로컬 물체 AUROC (%) | 차이 (%p) | 논문 포인트 AUROC (%) | 로컬 포인트 AUROC (%) | 차이 (%p) |
|---|---:|---:|---:|---:|---:|---:|
| Easy | 75.60 | 73.06 | −2.54 | 77.30 | 77.56 | +0.26 |
| Medium | 68.60 | 65.73 | −2.87 | 65.50 | 65.12 | −0.38 |
| **Hard** | **61.60** | **62.33** | +0.73 | **56.30** | **56.00** | −0.30 |
| ALL | 68.60 | 67.04 | −1.56 | 66.20 | 66.00 | −0.20 |

논문 기준: 본문 Table 2. [공개 원문](https://arxiv.org/html/2507.07435v1).
**논문과 로컬 모두 Hard에서 결함 위치를 찾는 성능이 낮음.** screw의 물체 판정·toothbrush의 위치 탐지는 논문에서도 낮지만, headset1의 물체 판정은 로컬에서 특히 크게 낮아진 항목임. 실행 환경·호환 연산 차이가 있으므로 그 원인까지 이 표로 확정하지 않음.

## 2. 어느 부분에서 느린가?

각 카테고리의 테스트 평균 시간을 구한 뒤 카테고리 간 단순 평균을 계산함. 파일 로딩·결과 저장·성능 지표 계산은 추론 합계에서 제외함.

| 데이터셋 | 특징 추출 (ms/샘플) | 점수 계산 전체 (ms/샘플) | 그중 메모리뱅크 NN (ms/샘플) | 추론 합계 (ms/샘플) |
|---|---:|---:|---:|---:|
| MiniShift | **4,558.49** | 744.50 | 3.50 | 5,302.99 |
| Anomaly-ShapeNet | **587.58** | 121.11 | 1.37 | 708.69 |

- **가장 느린 구간은 특징 추출:** 전체 추론의 약 **86.0% / 82.9%**를 차지함. 법선·FPFH·FPS·주변 점 탐색·특징 집계·전송을 포함함.
- **메모리뱅크 NN은 주요 시간 병목이 아님:** GPU의 `torch.cdist` + `torch.min`이며 FAISS는 사용하지 않음.
- 점수 계산 전체에는 NN 외에 보간·장치 이동·CPU 결과 수집 등이 포함됨. NN 시간은 점수 계산 전체의 일부이므로 합계에 다시 더하지 않음.
- 특징 추출 내부에서 어떤 연산이 가장 느린지는 이 구간별 측정만으로 확정하지 않음.

## 근거

- 전체 카테고리·난이도별 로컬 지표 CSV (로컬: `method12/source/results/all_categories_20261028/metrics.csv`)
- 52개 카테고리 완료 기록 (로컬: `method12/source/results/all_categories_20261028/progress.json`)
- 실행·시간 측정 코드 (로컬: `method12/source/all_categories_simple3d.py`)
- 진행 중인 원본 CUDA 연산 재실행 (로컬: `method12/markdown/Simple3D_native_all_categories_local.md`)