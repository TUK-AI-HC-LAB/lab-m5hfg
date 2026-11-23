# W48 대표점 주변에서 직접 기하 특징을 만드는 관련 연구

조사일: 2026-11-19. 범위: 선행연구 조사만 수행함. 구현 교체·학습·추론 실험은 하지 않음.

## 1. 먼저 내릴 판단

**‘전체 FPFH 대신 대표점 주변에서 직접 특징을 만든다’는 발상 자체는 새로운 방법이 아님.** SHOT, PPF-FoldNet, SpinNet은 지역 표면을 직접 기술하는 기존 방법이며, Group3AD는 이미 3D 이상 탐지에 그룹 특징과 기하 기반 중심점 선택을 사용함 [1–4]. 곡률을 이상 탐지 단서로 쓰는 것도 CASL에서 다뤄짐 [5].

따라서 이 연구들은 우리의 제안이라고 쓰는 것이 아니라, **비교해야 할 기준선과 새로움 검토 대상**으로 사용해야 함. 아래 적용 가능성·위험·우선순위는 논문이 입증한 결과가 아니라 현재 문제에 대한 판단임.

### 현재 문제와 연결

- 현재 Simple3D 변형은 전체 점의 120-NN → SPFH/FPFH → 대표점 4,096개 주변 128-NN 특징 평균 순서임.
- 6개 카테고리·24개 표본의 의존성 분석에서 MiniShift의 직접 FPFH 사용률은 53.8465%이지만, SPFH/쿼리 필요율은 99.9403%였음. Anomaly-ShapeNet 표본에서는 SPFH/쿼리 필요율이 100%였음.
- 이는 **현재 특징 정의를 유지한 채 출력 행만 선택하는 것으로는 큰 탐색 절감이 어렵다**는 근거임. 모든 FPFH 구현이나 모든 지역 특징의 한계라는 뜻은 아님.
- [분석 문서](../../method12/markdown/W48_Simple3D_FPFH_dependency_and_avoidable_work.md) · [원본 측정](../../method12/source/results/fpfh_dependency_audit_20261119/zero_edge_refinement/counts.csv).

## 2. 공통 용어

- **대표점:** 주변 영역을 요약할 기준 위치. 예: 병 표면에서 선택한 한 점.
- **지역 패치:** 대표점 주위의 원본 점 묶음. 원본 점을 전체에서 버린다는 뜻이 아님.
- **지역 기하 특징:** 패치의 위치 관계·표면 방향·형태를 숫자 벡터로 요약한 것.
- **법선:** 표면이 향하는 방향. SHOT·PPF 계열에서 사용하므로 이웃 탐색과 법선 계산 비용이 모두 사라지는 것은 아님.
- **정합:** 두 점군에서 같은 위치를 찾아 맞추는 과제. 정합 성능이 높다고 미세 결함 탐지 성능도 높다고 보장되지는 않음.
- **수작업 특징:** 사람이 정한 계산 규칙으로 만드는 특징. ‘학습 없음’은 ‘계산 비용 없음’이 아님.

## 3. 한 번에 비교

| 방법 | 패치를 특징으로 바꾸는 방법 | 학습 | 현재 문제와의 관계 | 주의할 점 |
|---|---|---|---|---|
| SHOT [1] | 지역 좌표계와 표면 법선 방향의 공간 히스토그램 | 특징 추출기 학습 불필요 | 대표 위치에서 직접 기술자를 만드는 비학습 비교 기준 | 법선·좌표계·패치 검색 비용이 남음 |
| PPF-FoldNet [2] | 중심점과 주변 점 사이의 거리·각도 → 오토인코더 특징 | 필요 | 전체 SPFH/FPFH 확산을 거치지 않는 지역 패치 표현 사례 | 정상 뱅크만 만드는 방법이 아니라 특징 학습도 필요함 |
| SpinNet [3] | 지역 패치를 원통 공간으로 변환 → 신경망 특징 | 필요 | 패치의 표면 구조를 직접 학습하는 비교 후보 | 원통 표현·신경망 비용과 사전학습 조건을 포함해야 함 |
| Group3AD [4] | 그룹 특징 학습·기억 저장소·기하 기반 중심점 선택 | 필요 | ‘그룹 기반 이상 탐지/적응적 중심 선택’의 직접적인 선행연구 | 중심점 선택에 FPFH를 사용하므로 그대로 가져와도 병목이 없어지는 것은 아님 |
| CASL [5] | 다중 스케일 곡률을 활용한 자기지도 표현 학습 | 필요 | ‘곡률을 쓰면 새로운가’에 대한 검토 대상 | 곡률 단독 관찰과 학습한 CASL 전체 방법을 혼동하면 안 됨 |

이 표는 구조 비교임. 논문별 시간·정확도를 같은 장치·데이터·프로토콜로 측정한 성능 순위표가 아님.

## 4. 논문별 확인 내용

### 4.1 SHOT — 학습 없는 지역 특징 기준선 [1]

| Item | Content |
|---|---|
| Title | Unique Signatures of Histograms for Local Surface Description |
| Authors | Federico Tombari, Samuele Salti, Luigi Di Stefano |
| Conference / Journal | ECCV |
| Year | 2010 |
| Paper link | [원문](https://www.cvl.iis.u-tokyo.ac.jp/class2017/2017w/papers/5.3DdataProcessing/Tombari_SHOT_ECCV2010.pdf) |
| GitHub / Official code | [저자 연구실 구현](https://github.com/fedassa/SHOT), [PCL 구현 문서](https://pointclouds.org/documentation/classpcl_1_1_s_h_o_t_estimation.html) |
| Reason for investigation | 학습 없이 중심 위치의 지역 표면을 직접 기술하는 비교 기준 확인 |

- **원문 확인:** §3의 지역 좌표계, §4의 공간 분할·방향 히스토그램. 지역 표면 매칭을 위한 방법임. [원문 §3–4](https://www.cvl.iis.u-tokyo.ac.jp/class2017/2017w/papers/5.3DdataProcessing/Tombari_SHOT_ECCV2010.pdf)
- **구현 확인:** PCL은 한 점의 공간 이웃과 법선으로 SHOT를 계산하는 `computePointSHOT`을 제공함. [PCL API](https://pointclouds.org/documentation/classpcl_1_1_s_h_o_t_estimation.html)
- **적용 판단:** 중심 위치에서 기술자를 직접 계산하는 기준선으로 우선 검토할 가치가 있음. 다만 현재의 ‘전체 FPFH를 모아 평균’하는 표현과는 달라서 정상 뱅크를 새로 만들어야 함.
- **위험 가설:** 작은 결함이 히스토그램에 약하게 반영되거나 지역 좌표계가 불안정할 수 있음. 우리 데이터에서의 정확도·GPU 속도는 미검증이며 PCL 코드 존재만으로 GPU 가속을 주장하지 않음.

### 4.2 PPF-FoldNet — 중심점과 주변 점의 관계를 직접 표현 [2]

| Item | Content |
|---|---|
| Title | PPF-FoldNet: Unsupervised Learning of Rotation Invariant 3D Local Descriptors |
| Authors | Haowen Deng, Tolga Birdal, Slobodan Ilic |
| Conference / Journal | ECCV |
| Year | 2018 |
| Paper link | [공식 논문 페이지](https://openaccess.thecvf.com/content_ECCV_2018/html/Tolga_Birdal_PPF-FoldNet_Unsupervised_Learning_ECCV_2018_paper.html), [저자 원문 HTML](https://arxiv.org/html/1808.10322v1) |
| GitHub / Official code | unverified — 이번 조사에서 저자 공식 코드 저장소를 확정하지 못함 |
| Reason for investigation | 전체 점 기술자 평균 대신 중심점 기준 패치 표현을 쓰는 선행연구 확인 |

- **원문 확인:** §3.1은 중심점과 주변 점의 법선·거리·각도 관계를 PPF 집합으로 표현함. §3.2는 이를 오토인코더로 압축해 지역 기술자로 사용함. ‘모든 주변 점 쌍’이 아니라 중심 기준의 점 쌍 표현임. [원문 §3](https://arxiv.org/html/1808.10322v1)
- **적용 판단:** 중심 기준 관계를 직접 계산하는 구조는 현재의 SPFH 이웃 의존성을 피할 후보 원리를 제공함. 그러나 이 방식이 더 빠르거나 결함에 더 민감하다는 것은 별도 검증해야 함.
- **한계:** 학습과 법선이 필요함. 학습을 없애고 PPF를 임의 히스토그램으로 모으면 원래 PPF-FoldNet이 아니라 별도 기준선임. 그 자체의 새로움도 미확인임.

### 4.3 SpinNet — 지역 표면을 원통 표현으로 변환 [3]

| Item | Content |
|---|---|
| Title | SpinNet: Learning a General Surface Descriptor for 3D Point Cloud Registration |
| Authors | Sheng Ao, Qingyong Hu, Bo Yang, Andrew Markham, Yulan Guo |
| Conference / Journal | CVPR |
| Year | 2021 |
| Paper link | [공식 원문](https://openaccess.thecvf.com/content/CVPR2021/papers/Ao_SpinNet_Learning_a_General_Surface_Descriptor_for_3D_Point_Cloud_CVPR_2021_paper.pdf), [저자 원문 HTML](https://arxiv.org/html/2011.12149v1) |
| GitHub / Official code | [공식 저장소](https://github.com/QingyongHu/SpinNet) |
| Reason for investigation | 지역 패치에서 직접 표면 기술자를 학습하는 대안 확인 |

- **원문 확인:** §3에서 지역 표면을 기준축에 정렬한 뒤 원통 공간으로 변환하고, 점 기반·원통 합성곱으로 특징을 학습함. 정합 과제에서 평가한 방법임. [원문 §3](https://arxiv.org/html/2011.12149v1)
- **적용 판단:** 패치 표현 대안의 학습 기반 비교 후보임. 정합을 위한 회전 강인성이 미세 결함 탐지에도 유리하다고 단정하지 않음.
- **구현 조건:** 공식 README의 검증 환경은 Python 3.6·PyTorch 1.6·CUDA 10.2임. 현재 RTX 5080 환경에서 실행 가능 여부와 변환 비용은 확인이 필요함. [공식 코드](https://github.com/QingyongHu/SpinNet)
- **한계:** 사전학습 데이터·패치 전처리·신경망 추론 비용을 포함해야 하므로, 우선순위를 SHOT보다 뒤로 둠. 이는 구현 부담에 대한 판단이며 성능 열위라는 뜻이 아님.

### 4.4 Group3AD — 가장 직접적인 새로움 검토 대상 [4]

| Item | Content |
|---|---|
| Title | Towards High-resolution 3D Anomaly Detection via Group-Level Feature Contrastive Learning |
| Authors | Hongze Zhu, Guoyang Xie, Chengbin Hou, Tao Dai, Can Gao, Jinbao Wang, Linlin Shen |
| Conference / Journal | ACM Multimedia 2024; 원문 기재 DOI: 10.1145/3664647.3680919 |
| Year | 2024 |
| Paper link | [출판 DOI](https://doi.org/10.1145/3664647.3680919), [저자 원문 HTML](https://arxiv.org/html/2408.04604v1) |
| GitHub / Official code | [논문이 안내한 저장소](https://github.com/M-3LAB/Group3AD); 2026-11-19 확인 시 README만 있고 실행 소스는 확인되지 않음 |
| Reason for investigation | 그룹 특징·기하 기반 대표점 선택이 이미 이상 탐지에서 제안됐는지 확인 |

- **원문 확인:** §3.2에서 그룹 특징 학습과 IUN/IAN을 제시하고, §3.2.3 AGCS는 FPFH 기반 변화 단서와 FPS를 결합해 중심점을 선택함. [원문 §3.2](https://arxiv.org/html/2408.04604v1)
- **중요한 구분:** ‘기하 변화가 큰 곳에 대표점을 더 배치한다’는 방향은 이미 있음. 또한 AGCS는 FPFH를 사용하므로 이 절차를 그대로 도입하는 것은 현재 FPFH 탐색 병목의 제거와 다름.
- **현재 판단:** 무조건 실행할 후보라기보다 신규 방법을 주장하기 전에 반드시 비교해야 할 연구임. 공개 저장소 링크와 실행 가능한 코드 공개 여부도 구분함. [저장소 확인](https://github.com/M-3LAB/Group3AD)

### 4.5 CASL — 곡률 추가 자체도 새 기여가 아님 [5]

| Item | Content |
|---|---|
| Title | CASL: Curvature-Augmented Self-supervised Learning for 3D Anomaly Detection |
| Authors | Yaohua Zha, Xue Yuerong, Chunlin Fan, Yuansong Wang, Tao Dai, Ke Chen, Shu-Tao Xia |
| Conference / Journal | AAAI |
| Year | 2026 (arXiv 최초 공개 2025) |
| Paper link | [AAAI 원문](https://ojs.aaai.org/index.php/AAAI/article/download/38226/42188), [저자 원문 HTML](https://arxiv.org/html/2511.12909v1) |
| GitHub / Official code | [논문이 안내한 공식 저장소](https://github.com/zyh16143998882/CASL) |
| Reason for investigation | 지역 곡률을 결함 단서로 쓰려는 방향의 기존 연구 범위 확인 |

- **원문 확인:** 곡률 자체를 점수로 사용하는 관찰과, 다중 스케일 곡률 prompt로 좌표 복원을 유도하는 CASL 학습 방법을 구분함. 이후 pseudo-anomaly 분류 미세조정도 포함함. [원문 Methodology](https://arxiv.org/html/2511.12909v1)
- **현재 판단:** ‘곡률을 넣는다’만으로는 새롭지 않음. 곡률 사용이 모든 점의 이웃 탐색을 자동으로 없애지도 않음.
- **위험 가설:** 정상 모서리·나사산도 기하 변화가 크므로 곡률이 큰 점을 곧바로 결함이라고 처리하면 오탐 가능성이 있음. 이 문장은 현재 적용에 대한 위험 가설이지 CASL 전체의 실패를 입증한 결과가 아님.

## 5. 무엇이 이미 있고, 무엇을 추가로 증명해야 하는가

| 주장하려는 내용 | 이번 조사에서의 판단 |
|---|---|
| 대표점 주변을 직접 벡터로 만들었음 | SHOT·PPF-FoldNet·SpinNet 등과 겹침. 단독 새 기여로 쓰기 어려움 |
| 그룹 특징으로 3D 이상을 찾음 | Group3AD가 직접적인 선행연구임 |
| 곡률·기하 변화가 큰 영역에 집중함 | CASL·Group3AD와의 차이를 먼저 밝혀야 함 |
| SHOT로 바꿨더니 빨라짐 | 유용한 비교 실험이지만 기본적으로 기존 특징 교체임 |
| 같은 계산 예산에서 미세 결함을 더 잘 보존함 | 검증할 연구 질문 후보. 지금은 결과도 새로움도 미확인임 |

조사한 5편이 전체 문헌을 망라하지는 않음. 특히 새로운 특징·대표점 배치 규칙을 제안하기 전에는 최신 관련 방법과 추가 비교가 필요함. 현재 문서로 ‘아무도 안 했다’거나 최신 최고 성능이라고 주장하지 않음.

## 6. 다음 비교를 한다면 무엇을 고정해야 하는가

**추천 기준선은 대표 위치에서 직접 계산하는 SHOT 계열임. 새 알고리즘이라고 부르기 전에, 이 기존 방법만으로 문제가 해결되는지 확인하는 용도임. 아직 실행하지 않음.**

```text
현재: 원본 점 전체 → 전체 FPFH → 대표점 주변 FPFH 평균 → 정상 뱅크 비교
후보: 원본 점 전체 → 대표점 주변 패치 → 직접 지역 기술자 → 새 정상 뱅크 비교
```

- 원본 점은 검색 대상으로 유지하고, 우선 대표점 선택·개수·후처리는 고정함. SHOT의 support 반경과 지역 좌표계 계산을 128-NN 집계와 같은 것으로 취급하지 않음. 차이는 설정으로 명시해야 함.
- 특징이 달라지므로 정상 메모리뱅크를 다시 만들어야 함. 기존 FPFH 뱅크에 새 특징을 그대로 비교하면 안 됨.
- 기술자 차원·정규화·정상 뱅크 벡터 수·저장 바이트·검색 시간을 함께 기록함. 같은 coreset 비율만으로 모든 메모리·계산 조건이 같아지는 것은 아님.
- 학습 기반 후보는 사전학습 데이터·가중치 출처·추가 학습 조건을 별도로 기록함. 정상 뱅크 구성만 수행한 기준선과 학습 조건이 다름.
- CPU 구현과 GPU 구현 시간을 비교하면서 알고리즘만의 효과라고 해석하지 않음. 법선·패치 검색·지역 좌표계·특징·메모리 검색을 포함한 전체 시간을 측정해야 함.
- 원래 FPFH와 다른 특징이므로 bitwise 동일 점수가 아니라 물체 AUROC·포인트 AUROC/AP·미세 결함 위치 보존·시간·메모리의 관계로 평가함.
- 테스트 결함 mask는 평가에만 사용하고 대표점 선택·반경 선택·특징 설계의 정답 입력으로 사용하지 않음.

## 7. 이번 조사로 내릴 결론

1. ‘직접 지역 특징’은 기존 연구가 충분히 있는 계열임. 지금 제안할 새 알고리즘의 이름으로 쓰지 않음.
2. SHOT는 비학습 비교 기준선, PPF-FoldNet·SpinNet은 학습 기반 표현 사례, Group3AD·CASL은 이상 탐지에서의 차별성 검토 대상으로 구분함.
3. 아직 남은 질문은 **‘FPFH의 전체 점 의존성을 없앴을 때, 기존 지역 기술자가 놓치는 미세 결함이 무엇이며 왜 그런가’**임. 그 실패가 확인돼야 과제에 특화된 새 특징/집계 규칙을 설계할 근거가 생김.
4. 이번 작업에서는 소스 변경·설치·실험 실행·GitHub 업로드를 하지 않음. 원본 문서만 만들고 Obsidian 탐색 목록에 연결함.

## 8. 참고문헌과 확인 범위

[1] Tombari, Federico, Samuele Salti, and Luigi Di Stefano. “Unique Signatures of Histograms for Local Surface Description.” *Computer Vision — ECCV 2010*, 2010, pp. 356–369.

[2] Deng, Haowen, Tolga Birdal, and Slobodan Ilic. “PPF-FoldNet: Unsupervised Learning of Rotation Invariant 3D Local Descriptors.” *Proceedings of the European Conference on Computer Vision*, 2018, pp. 602–618.

[3] Ao, Sheng, et al. “SpinNet: Learning a General Surface Descriptor for 3D Point Cloud Registration.” *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2021.

[4] Zhu, Hongze, et al. “Towards High-resolution 3D Anomaly Detection via Group-Level Feature Contrastive Learning.” *Proceedings of the 32nd ACM International Conference on Multimedia*, 2024. DOI: 10.1145/3664647.3680919.

[5] Zha, Yaohua, et al. “CASL: Curvature-Augmented Self-supervised Learning for 3D Anomaly Detection.” *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 40, no. 15, 2026, pp. 12340–12348.

확인 범위: 원문의 문제 정의·방법 절과 공식 논문/코드 페이지를 중심으로 구조 및 메타데이터를 확인함. 모든 실험 표·부록을 전수 검토하거나 재현한 것은 아님. 일부 CVF/AAAI 페이지는 직접 열기가 제한되어 저자 공개 원문 HTML과 공식 페이지의 검색 색인도 함께 대조함. PPF-FoldNet 공식 코드 출처는 미확인으로 남김.
