# W46 3D 이상 탐지 효율화 관련 연구

## 비교 요약

| 가속 대상 | 논문 | 핵심 방법 | 논문 실험 결과 |
|---|---|---|---|
| 중심점 선택 | **FlashFPS** | FPS 후보와 반복 횟수를 줄이고, 계층형 모델에서는 이전 층의 결과를 재사용 | S3DIS·ScanNet 의미 분할에서 표준 CUDA FPS 대비 전체 추론 **평균 5.16배** 가속(75% 가지치기). [Table 1](https://arxiv.org/html/2604.17720) |
| 중심점 선택 및 이후 이웃 탐색 | **FastPoint** | 중심점 간 최소 거리의 변화 추세를 예측하고, 샘플링 때 계산한 거리 정보를 이웃 탐색에 재사용 | S3DIS·ScanNet·SemanticKITTI에서 기존 FPS 기준 전체 추론 **기하평균 2.55배** 가속. [Figure 7](https://openaccess.thecvf.com/content/ICCV2025/papers/Lee_FastPoint_Accelerating_3D_Point_Cloud_Model_Inference_via_Sample_Point_ICCV_2025_paper.pdf#page=8) |

데이터셋·하드웨어·측정 범위가 다르므로 논문의 배속을 직접 비교할 수 없음.

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

Simple3D에서는 **특징 집계 중심점 4,096개와 점수 평활화 중심점 1,024개를 고르는 두 FPS**에 대응함. MiniShift 캡슐에서 두 FPS에 FPS-Prune만 적용한 결과는 [GPU KD-tree 기준 비교표](W46_Simple3D_capsule_GPU_KDTree_FlashFPS_comparison.md)에 정리함. 논문의 전체 FlashFPS 결과와 달리 해당 실험에는 FPS-Cache를 적용하지 않았음.

## FastPoint · 거리 예측으로 중심점 선택 및 이웃 탐색 가속

| Item | Content |
|---|---|
| Title | FastPoint: Accelerating 3D Point Cloud Model Inference via Sample Point Distance Prediction |
| Authors | Donghyun Lee, Dawoon Jeong, Jae W. Lee, Hongil Yoon |
| Conference / Journal | ICCV |
| Year | 2025 |
| Paper link | [CVF 원문](https://openaccess.thecvf.com/content/ICCV2025/html/Lee_FastPoint_Accelerating_3D_Point_Cloud_Model_Inference_via_Sample_Point_ICCV_2025_paper.html) |
| GitHub / Official code | [저자 저장소](https://github.com/SNU-ARC/FastPoint) |
| Reason for investigation | Simple3D에서 중심점을 고르는 FPS를 가속할 다른 접근 검토 |

FPS는 이미 고른 중심점에서 멀리 떨어진 점을 다음 중심점으로 고름. FastPoint의 **MDPS**는 FPS를 일부만 실행해 중심점 간 최소 거리의 변화 추세를 추정함. 그 추세를 이용해 기존 중심점에 너무 가까운 점을 후보에서 빼고, 샘플링 중에 계산한 거리 정보를 이후 반경 검색과 k-NN에도 재사용함. FlashFPS의 후보·반복 가지치기와 달리 **거리 예측을 이용해 표준 FPS의 선택 패턴을 근사**하는 방식임. [논문 §4](https://openaccess.thecvf.com/content/ICCV2025/papers/Lee_FastPoint_Accelerating_3D_Point_Cloud_Model_Inference_via_Sample_Point_ICCV_2025_paper.pdf#page=5)

Simple3D에서 FPS만 비교하려면 **MDPS만** 적용하고 기존 GPU KD-tree 이웃 탐색과 중심점 수를 유지하는 것이 맞음. 논문의 2.55배는 이웃 탐색 최적화까지 포함한 전체 추론 수치이며, Simple3D 측정 결과가 아님.
