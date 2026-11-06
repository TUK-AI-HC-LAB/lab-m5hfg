# W46 3D 이상 탐지 효율화 관련 연구

## Paper Metadata

| Item | PointCore | M2P-AD |
|---|---|---|
| Title | PointCore: An efficient framework for unsupervised point cloud anomaly detection using joint local-global features | M2P-AD: Memory-to-Prototype Learning with Boundary-aware Score Refinement for 3D Anomaly Detection |
| Authors | Baozhu Zhao, Xiaohan Zhang, Jingfeng Guo, Qi Liu | Seyoung Jeong, Jong Pil Yun, Sang Jun Lee |
| Conference / Journal | Neural Networks, Vol. 197, 108446 | arXiv 프리프린트 |
| Year | 2026 저널판, 2024 최초 프리프린트 | 2026 |
| Paper link | [저널판 PDF](https://drliuqi.github.io/files/publications/PointCore.pdf) | [arXiv v1](https://arxiv.org/html/2607.13499v1) |
| GitHub / Official code | [저자 저장소](https://github.com/yuedumingz/PointCore) | unverified |
| Reason for investigation | 좌표를 활용한 메모리뱅크 검색 후보 축소 | 정상 특징 압축과 경계 부위 오탐 보정 |

## 비교 요약

| 관점             | 논문                                    | 변경한 부분                                              | 실험 근거                                                                                                                                                                                               |
| -------------- | ------------------------------------- | --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 메모리뱅크 검색       | **PointCore**                         | 정렬된 XYZ 좌표로 후보를 좁힌 뒤 후보 안에서 고차원 PointMAE 특징 비교      | Real3D-AD에서 Reg3D-AD 대비 전체 추론 **13.022 → 4.282초/샘플**. PointCore의 좌표 기반 점수 계산 **0.840초**, PointMAE 기반 점수 계산 **1.918초**. [§4.4, Table 14](https://drliuqi.github.io/files/publications/PointCore.pdf) |
| 메모리뱅크 압축       | **M2P-AD**                            | 정상 패치 특징을 K-means로 묶고 학습한 **대표 벡터 128개**를 기본 검색에 사용 | Real3D-AD, Anomaly-ShapeNet, MulSen-AD에서 탐지 성능 평가. **128개**는 기본 뱅크의 벡터 수. [§3.3–3.5, §4](https://arxiv.org/html/2607.13499v1)                                                                       |
| 특징 추출 중 중심점 선택 | **FlashFPS**                          | 최원점 샘플링에서 비교 후보와 반복 횟수를 줄이고, 계층형 모델에서 샘플링 결과를 재사용   | S3DIS·ScanNet의 의미 분할 모델에서 표준 CUDA FPS 대비 **평균 5.16배 전체 추론 가속**(가지치기 비율 75%). [Table 1, §4.3](https://arxiv.org/html/2604.17720)                                                                     |
| 국소 형상 특징 계산    | **Curvature-Fused Feature Extractor** | CALE로 저차원 형상 특징을 계산하고 SOON으로 정상 뱅크에 저장할 특징을 선별      | Real3D-AD의 Seahorse에서 **전체 추론 1.6초/샘플**, 비교 방법보다 **2.77초 단축**. [IEEE Access 논문](https://doi.org/10.1109/ACCESS.2026.3656572)                                                                        |

## PointCore · 메모리뱅크 검색

정상 특징에 XYZ 좌표를 함께 저장한다. 검사 포인트를 정상 형상과 정렬한 뒤, XYZ가 가까운 정상 후보를 먼저 찾는다. 이후 그 후보 안에서 PointMAE 특징이 가장 가까운 항목을 고른다. 위치가 비슷한 부위끼리 비교하면서 고차원 거리 계산의 후보 수를 줄이는 구조다. Table 14의 1.918초는 **PointMAE 기반 점수 계산 단계**의 시간이고, 특징 추출은 별도로 **0.424초**다. [논문 §3, Table 14](https://drliuqi.github.io/files/publications/PointCore.pdf#page=10)

## M2P-AD · 메모리뱅크 압축

정상 패치 특징을 **128개 군집**으로 나눈 뒤 각 군집의 중심을 보정해 대표 벡터로 저장한다. 검사 패치는 이 대표 벡터와 먼저 비교한다. 기본 점수가 높은 **상위 20% 패치**에만 별도 경계 뱅크를 추가로 검색하고, 두 거리 중 작은 값을 최종 점수로 사용해 정상 경계의 높은 점수를 낮춘다. [논문 §3.3–3.5](https://arxiv.org/html/2607.13499v1)

기본 뱅크의 크기는 벡터 개수로 제시된다. 전체 저장량에는 별도 경계 뱅크가 포함된다. 논문에서 보고한 128개는 압축 후 크기이며, 특징 추출·군집화·대표 벡터 보정까지 포함한 **구성 시간**과 **최대 VRAM**은 별도 측정이 필요한 지표다.

## FlashFPS · 특징 추출 중 중심점 선택

| Item | Content |
|---|---|
| Title | FlashFPS: Efficient Farthest Point Sampling for Large-Scale Point Clouds via Pruning and Caching |
| Authors | Yuzhe Fu, Hancheng Ye, Cong Guo, Junyao Zhang, Qinsi Wang, Yueqian Lin, Changchun Zhou, Hai (Helen) Li, Yiran Chen |
| Conference / Journal | DAC |
| Year | 2026 |
| Paper link | [arXiv](https://arxiv.org/abs/2604.17720) |
| GitHub / Official code | [저자 저장소](https://github.com/Yuzhe-Fu/FlashFPS) |
| Reason for investigation | Simple3D의 특징 추출에서 중심점을 고르는 FPS 연산의 가속 후보 검토 |

**FPS-Prune**은 최원점 후보와 반복 횟수를 줄인다. **FPS-Cache**는 여러 층에서 FPS를 반복하는 모델에서 앞 층의 샘플링 결과를 재사용한다. **5.16배**는 두 기법을 적용한 PointNeXt-L·PointVector-L의 의미 분할 전체 추론 가속 수치다. [논문 §3, Table 1](https://arxiv.org/html/2604.17720)

Simple3D에서 직접 대응하는 부분은 **FPFH 계산 후 특징을 모을 중심점 4,096개를 고르는 단계**다. FlashFPS 적용 효과는 MiniShift에서 FPS 자체 시간, 특징 추출 전체 시간, 물체·포인트 AUROC를 함께 측정해 판단한다. 층 간 캐싱의 효과도 Simple3D의 샘플링 구조에 맞춰 별도 확인이 필요하다.

## Curvature-Fused Feature Extractor · 국소 형상 특징 계산

| Item | Content |
|---|---|
| Title | Curvature-Fused Feature Extractor for Accelerated Point Cloud Anomaly Detection |
| Authors | Yujie Huang, Zhiquan Qi |
| Conference / Journal | IEEE Access, Vol. 14, pp. 13491–13500 |
| Year | 2026 |
| Paper link | [IEEE Access 논문](https://doi.org/10.1109/ACCESS.2026.3656572) |
| GitHub / Official code | unverified |
| Reason for investigation | Simple3D의 FPFH에 해당하는 국소 형상 특징 계산을 저차원 표현으로 바꾸는 접근 조사 |

**CALE**은 점의 위치, 표면 방향, 곡률을 이용해 주변 형상을 저차원 특징으로 표현한다. **SOON**은 복셀과 최근접 이웃 탐색을 이용해 정상 메모리뱅크에 넣을 대표 특징을 선별한다. 논문은 Real3D-AD와 Anomaly-ShapeNet에서 평가했으며, **Seahorse의 1.6초/샘플은 특징 계산과 매칭을 포함한 전체 추론 시간**이다. CALE만의 시간 단축 폭은 별도로 확인되지 않는다. [논문 §III–IV](https://doi.org/10.1109/ACCESS.2026.3656572)

## Simple3D와 연결

| 비교할 단계 | 참고할 연구 | 따로 기록할 값 |
|---|---|---|
| 정상 메모리뱅크에서 최근접 특징 검색 | PointCore | 검색 시간, 후보 수, 물체·포인트 AUROC |
| 정상 특징을 뱅크로 만드는 과정 | M2P-AD | 정상 특징 추출 시간, 압축 시간, 기본·경계 뱅크 크기, 최대 VRAM |
| 특징을 집계할 중심점 선택 | FlashFPS | FPS 시간, 특징 추출 전체 시간, 물체·포인트 AUROC |
| 점 주변의 국소 형상 특징 계산 | Curvature-Fused Feature Extractor | 형상 특징 계산 시간, 특징 추출 전체 시간, 물체·포인트 AUROC |

논문마다 데이터셋과 측정 범위가 다르므로 각 논문 안에서 제시한 비교 조건에 맞춰 수치를 읽는다. Simple3D에서는 FPS, 좌표 이웃 탐색, 메모리뱅크 검색을 분리해 기록한다. FPS는 중심점을 고르고, 좌표 이웃 탐색은 중심점 주변의 점을 찾으며, 메모리뱅크 검색은 만들어진 특징을 정상 뱅크와 비교한다.
