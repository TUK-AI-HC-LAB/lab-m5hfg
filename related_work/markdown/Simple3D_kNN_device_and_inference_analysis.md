# Simple3D의 메모리뱅크 탐색 장치와 추론 시간 분석

## Paper Metadata

| Item | Content |
|---|---|
| Title | Towards High-Resolution 3D Anomaly Detection: A Scalable Dataset and Real-Time Framework for Subtle Industrial Defects |
| Authors | Yuqi Cheng, Yihan Sun, Hui Zhang, Weiming Shen, Yunkang Cao |
| Conference / Journal | AAAI 2026, 40(5), 3327–3334 |
| Year | 2026 |
| Paper link | https://doi.org/10.1609/aaai.v40i5.37328 |
| GitHub / Official code | https://github.com/hustCYQ/MiniShift-Simple3D |
| Reason for investigation | 실제 실행 코드에서 두 종류의 이웃 탐색을 구분하고, FAISS 사용 여부·CPU/GPU 배치와 느린 추론 시간의 관계를 확인 |

분석 기준: **논문의 방법·실험 조건을 먼저 설명하고, 공개 공식 코드와 로컬 실행의 차이를 구분한다.** 공식 코드 commit은 `5951e097ed5f6c52c1e058490f8c64f9b6323501`이다. 확인일: 2026-10-29.

## 1. 논문에서 설명하는 방법

처리 흐름은 **FPFH 기반 여러 크기의 주변 특징 계산 → 주변 특징 집계 → 정상 특징 집합 구성 → 최근접 정상 특징과의 거리로 이상 점수 계산**이다.

- **정상 기준:** 정상 학습 데이터에서 얻은 특징 집합을 사용한다. 논문의 정상 대표 특징 집합이 코드의 메모리뱅크에 해당한다.
- **포인트 이상 점수:** 검사 특징과 가장 가까운 정상 특징 사이의 거리다. 논문 식 (9)의 핵심이다.
- **물체 이상 점수:** 논문 식 (10)은 포인트 이상 점수의 **최댓값**으로 정의한다.
- **FPFH:** Implementation Details에 Open3D의 **CUDA 가속 구현**을 사용했다고 서술한다.
- **기본 설정:** FPFH 이웃 크기 40·80·120, 특징 집계 이웃 수 128, 실험 GPU는 **NVIDIA A100**이다.
- **속도:** Runtime 분석은 4k 해상도 조건에서 **20 FPS 초과**를 보고한다.

**논문 본문만으로 FAISS 사용 여부, 특정 kNN 라이브러리, 모든 단계의 CPU/GPU 배치를 확정할 수는 없다.** 아래 구현 세부사항은 공개 코드에서 확인한 내용이다.

근거: [논문 원문](https://arxiv.org/html/2507.07435v1)의 Anomaly Detection 식 (9)–(10), Experimental Setups—Implementation Details, Analysis on Runtime. [로컬 출판 PDF](https://ojs.aaai.org/index.php/AAAI/article/download/37328/41290)의 4쪽·7쪽에서도 확인했다.

## 2. 공개 공식 코드에서 확인되는 NN 구현

### 메모리뱅크 탐색 — GPU, FAISS 없음

```python
# feature_extractors/features.py, compute_anomay_scores()
dist = torch.cdist(patch, self.patch_lib)
min_val, min_idx = torch.min(dist, dim=1)
```

검사 특징과 정상 메모리뱅크의 모든 쌍에 대해 유클리드 거리를 계산하고, 가장 가까운 하나를 선택하는 **완전 탐색 1-NN**이다. 이 경로에서 FAISS를 호출하지 않는다. 로컬 실행에서는 두 텐서를 CUDA에 올리므로 거리 계산·최솟값 선택은 GPU에서 수행한다.

```python
# 공식 coreset 선택 후
self.patch_lib = self.patch_lib[self.coreset_idx].to(self.args.device)

# 로컬 실행에서도 평가 직전 CUDA 이동 확인
m.patch_lib = m.patch_lib.cuda()
```

### 좌표 주변 점 탐색 — 별개의 연산

```python
# feature_extractors/FPFH.py
knn = KNN(k=self.args.group_size, transpose_mode=True)
idx = batched_knn(knn, unorganized_pc_no_zeros, center)
```

이 KNN은 입력의 XYZ에서 중심점 주변 점을 찾는 연산이다. **검사 특징과 메모리뱅크를 비교하는 NN과 다르다.** 공식 코드는 `knn_cuda.KNN`을 사용한다.

근거: [공식 거리 계산](https://github.com/hustCYQ/MiniShift-Simple3D/blob/5951e097ed5f6c52c1e058490f8c64f9b6323501/feature_extractors/features.py#L175), [메모리뱅크 장치 이동](https://github.com/hustCYQ/MiniShift-Simple3D/blob/5951e097ed5f6c52c1e058490f8c64f9b6323501/feature_extractors/features.py#L303), [공식 좌표 kNN 호출](https://github.com/hustCYQ/MiniShift-Simple3D/blob/5951e097ed5f6c52c1e058490f8c64f9b6323501/feature_extractors/FPFH.py#L95).

## 3. 논문·공개 코드·현재 실행 비교

현재 실행은 `native_all_categories_20261029`의 원본 CUDA 연산 재실행이다. 아래 비교에는 이 실행만 사용한다.

| 항목        | 논문 서술                         | 공개 공식 코드                                    | 현재 로컬 실행                                      |
| --------- | ----------------------------- | ------------------------------------------- | --------------------------------------------- |
| GPU       | A100                          | 장치 설정에 따름                                   | RTX 5080                                      |
| 법선·FPFH   | FPFH에 Open3D CUDA 가속 구현 사용 서술 | Legacy Open3D의 CPU 경로                       | CPU                                           |
| 좌표 이웃 탐색  | 구체 라이브러리·장치 미명시               | GPU KNN_CUDA                                | GPU KNN_CUDA 소스 보존 포크                         |
| FPS       | 중심점 선택에 FPS 사용                | PointNet2 CUDA                              | PointNet2 원본 CUDA 커널                          |
| 포인트 점수 보간 | 논문만으로 정렬 구현까지 확정 불가           | GPU에서 거리 전체 정렬 후 가까운 3개 사용                  | **공식 전체 정렬 구현**                               |
| 메모리뱅크 NN  | 최근접 정상 특징과의 거리; FAISS 미명시     | `torch.cdist` + `torch.min`                 | 동일, GPU                                       |
| 물체 점수     | 식 (10): **최댓값**               | ShapeNet·MiniShift: **상위 80개 평균**           | 공개 코드 유지                                      |
| FPFH 스케일  | 40·80·120                     | num_MSND=2에서 중간 스케일 중복 연결                   | 중복 유지                                         |
| 실행 환경     | 세부 버전은 본문만으로 확정 불가            | README 권장 Python 3.8·Torch 1.12.1·CUDA 11.3 | Python 3.11·Torch 2.11·CUDA 12.8, sm_120 재컴파일 |

**현재 실행도 논문의 모든 조건과 동일하지 않다.** 원본 연산의 CPU/GPU 배치는 사용하지만, GPU FPFH·A100·논문 식의 최댓값 집계까지 맞춘 실험은 아니다.

KNN_CUDA는 원본 저장소·배포 파일이 404여서 원본 소스를 보존한 `unique1i/knn_cuda` 포크를 사용한다. PointNet2는 지원 아키텍처 목록을 RTX 5080용 `sm_120`으로 변경해 빌드했다. 경로 치환 오류와 점수맵 CPU/CUDA 인덱싱 오류 우회도 포함된다. CPU KD-tree·대체 FPS·topk 보간은 사용하지 않는다.

근거: [공식 FPFH 경로·스케일 연결](https://github.com/hustCYQ/MiniShift-Simple3D/blob/5951e097ed5f6c52c1e058490f8c64f9b6323501/feature_extractors/FPFH.py#L63), [공식 물체 점수 집계](https://github.com/hustCYQ/MiniShift-Simple3D/blob/5951e097ed5f6c52c1e058490f8c64f9b6323501/feature_extractors/features.py#L220), 현재 실행 환경 (로컬: `method12/source/results/native_all_categories_20261029/environment.json`), 빌드·검증 로그 (로컬: `method12/source/results/native_ops_20261029/build.log`).

## 4. 현재 실행에서 느린 구간

**문서 수정 시 확인된 완료 카테고리 기준**이다. Anomaly-ShapeNet은 40개 전체, MiniShift는 완료된 5개만 포함한다. MiniShift의 전체 12개 평균은 아니다. 각 카테고리의 테스트 평균 시간을 구한 뒤 카테고리 간 단순 평균을 계산했다.

| 데이터셋·완료 범위 | 특징 추출 (ms/샘플) | 점수 계산 전체 (ms/샘플) | 그중 메모리뱅크 NN (ms/샘플) | 추론 합계 (ms/샘플) |
|---|---:|---:|---:|---:|
| MiniShift — 완료 5개 | **23,957.50** | 1,116.04 | **4.00** | 25,073.54 |
| Anomaly-ShapeNet — 전체 40개 | **2,791.38** | 120.58 | **0.99** | 2,911.96 |

MiniShift 완료 5개는 capsule·cube·spring_pad·screw·screen이다. 수치는 실행 중간 스냅샷이며 이후 완료되는 카테고리는 반영하지 않았다.

- **느린 주 구간은 특징 추출:** 위 측정에서 추론의 약 **95.5% / 95.9%**다. CPU 법선·FPFH와 GPU FPS·좌표 kNN·특징 집계, 장치 이동을 포함한다.
- **메모리뱅크 NN은 주요 시간 병목이 아님:** 현재 GPU에서 별도로 측정한 NN 시간은 위 표의 4.00ms / 0.99ms다.
- 점수 계산 전체에는 메모리뱅크 NN 외에 공식 전체 정렬 보간·장치 이동·결과 수집이 포함된다. NN은 점수 계산 전체의 일부이므로 추론 합계에 다시 더하지 않는다.
- 특징 추출 전체가 느리다는 것은 확인했지만, 그 안의 **CPU FPFH와 GPU 좌표 kNN 중 어느 쪽이 얼마나 차지하는지는 아직 세분 측정하지 않았다.** CPU FPFH만이 원인이라고 단정하지 않는다.
- 논문은 CUDA 가속 FPFH를 서술하지만 현재 공개 코드 경로는 CPU다. 이는 중요한 조건 차이이며, 동일 입력의 CUDA FPFH 비교 실험 전에는 가속 효과를 확정하지 않는다.

근거: 현재 실행 카테고리별 시간 CSV (로컬: `method12/source/results/native_all_categories_20261029/metrics.csv`), 현재 진행 기록 (로컬: `method12/source/results/native_all_categories_20261029/progress.json`), 현재 실행 코드 (로컬: `method12/source/native_all_categories.py`), 시간 측정 코드 (로컬: `method12/source/all_categories_simple3d.py`), CPU·GPU 관측 (로컬: `method12/markdown/Simple3D_native_resource_usage.md`).

## 5. 논문 속도와 비교할 때의 결론

**현재 메모리뱅크 검색은 FAISS 없이 GPU에서 수행하며, 느린 구간은 특징 추출이다.** 논문과는 GPU 종류·FPFH 구현·물체 점수 집계·실행 환경의 차이가 있다. 논문의 20 FPS와 현재 ms의 비율을 그대로 알고리즘 자체의 속도 차이라고 해석하지 않는다.