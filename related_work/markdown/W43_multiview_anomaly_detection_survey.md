# 3D·멀티뷰 이상탐지 최신 논문 동향과 데이터셋

- 범위: 산업용 물체의 결함 탐지·위치 검출
- 목적: 논문 한 편의 세부 구조보다, 문제 → 해결 방법 → 검증 데이터셋을 요약하여 파악한다.
- 조사 성격: 선택한 대표 논문의 비교 조사이며, 전체 문헌을 망라한 체계적 리뷰는 아니다.

## Paper Metadata

각 행은 아래 요약에서 사용하는 논문이다. 코드가 확인되지 않은 경우 `unverified`로 표시했다. 조사 이유는 각 방법을 포함한 목적이다.

| ID  | Title                                                                                                                  | Authors                                                                             | Conference / Journal                | Year | Paper link                                                                | GitHub / Official code                                                   | Reason for investigation          |
| --- | ---------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ----------------------------------- | ---- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------ | --------------------------------- |
| [1] | Towards High-Resolution 3D Anomaly Detection: A Scalable Dataset and Real-Time Framework for Subtle Industrial Defects | Yuqi Cheng, Yihan Sun, Hui Zhang, Weiming Shen, Yunkang Cao                         | AAAI                                | 2026 | [공식](https://ojs.aaai.org/index.php/AAAI/article/view/37328)              | [코드](https://github.com/hustCYQ/MiniShift-Simple3D)                      | MiniShift와 Simple3D: 미세 결함·해상도·효율 |
| [2] | Point Cloud Diffusion with Global and Local Reconstruction for Instance-Level 3D Anomaly Detection                     | Linchun Wu, Qin Zou, Jiwen Lu, Qingquan Li                                          | arXiv preprint; 학회·저널 게재 unverified | 2026 | [본문](https://arxiv.org/html/2606.25740v1)                                 | unverified                                                               | PCDiff: 결함 합성과 국소·전역 복원           |
| [3] | AT3D-AD: Anomaly Type-Aware 3D Anomaly Detection via Hierarchical Point-Language Alignment                             | Jingyu Zeng, Haoquan Lu, Can Gao                                                    | arXiv preprint; 학회·저널 게재 unverified | 2026 | [본문](https://arxiv.org/html/2609.25930v1)                                 | unverified                                                               | 위치 검출·유형 인식과 포인트–언어 정렬            |
| [4] | CASL: Curvature-Augmented Self-supervised Learning for 3D Anomaly Detection                                            | Yaohua Zha, Xue Yuerong, Chunlin Fan, Yuansong Wang, Tao Dai, Ke Chen, Shu-Tao Xia  | AAAI                                | 2026 | [공식](https://ojs.aaai.org/index.php/AAAI/article/view/38226)              | [코드](https://github.com/zyh16143998882/CASL)                             | 곡률을 이용한 자기지도 표현 학습                |
| [5] | Physics-inspired Pseudo Anomaly Generation and Prototype Feature Guidance for 3D Anomaly Detection                     | Jian Ning, Qin Zou, Linchun Wu, Yuanhao Yue, Kunmo Li, Shoubin Chen, Zhongyuan Wang | Pattern Recognition; 온라인 공개본        | 2026 | [논문](https://arxiv.org/abs/2607.10544)                                    | [코드](https://github.com/NingxiaoJian/PA3AD)                              | PA3AD: 물리 기반 결함 합성과 정상 기준         |
| [6] | A Semantically Disentangled Unified Model for Multi-category 3D Anomaly Detection                                      | SuYeon Kim, Wongyu Lee, MyeongAh Cho                                                | CVPR                                | 2026 | [본문](https://arxiv.org/html/2603.25159v1)                                 | [공식 페이지의 Code 링크](https://visualsciencelab-khu.github.io/SeDiR_project/) | SeDiR: 다중 범주 특징 혼동                |
| [7] | MFF-M3AD: A unified reconstruction method with multi-scale feature fusion for multi-category 3D anomaly detection      | Hanzhe Liang, Chenxi Hu, Yejin Tang, Linlin Shen, Jinbao Wang, Can Gao              | Neural Networks; 온라인 공개본            | 2026 | [공식](https://www.sciencedirect.com/science/article/pii/S0893608026005927) | [저자 지정 저장소](https://github.com/hzzzzzhappy/MFF-M3AD)                     | 다중 스케일·다중 범주 재구성                  |

메타데이터 주의: 온라인 공개 연도와 권·호 발행일은 다를 수 있다.
## 1. 왜 3D·멀티뷰를 쓰는가

RGB 사진은 색·질감 결함을 관찰하기 좋지만 깊이와 형상 변화를 직접 표현하지 못한다. 3D는 찍힘·돌출·함몰 같은 기하 결함을, 멀티뷰는 단일 시점에서 가려지거나 잘 보이지 않는 부위를 관찰하는 데 활용한다. 외관과 형상을 결합하는 방법도 있다. [1, 2]

| 구분           | 사용하는 정보                    | 목적·특징                             |
| ------------ | -------------------------- | --------------------------------- |
| 순수 3D        | 포인트 좌표와 기하 특징              | 형상 결함 직접 분석. 예: Simple3D, CASL    |
| 실제 촬영 멀티뷰    | 동일 물체를 여러 카메라·시점에서 촬영한 영상  | 관측 범위 확대. 예: Real-IAD 데이터셋        |
| 3D의 멀티뷰 투영   | 포인트 클라우드를 여러 깊이 영상 등으로 렌더링 | 2D 사전학습 모델 활용. PCDiff의 렌더링 영상 활용  |
| RGB+3D·다중 센서 | 영상·포인트 클라우드·열화상 등          | 외관·형상 등 상호 보완 정보 활용               |
| 포인트–언어       | 3D 특징과 텍스트 특징              | 정상·이상 의미 연결과 결함 유형 학습. 예: AT3D-AD |

**입력 형식과 모델 종류는 별도 분류다.** ViT·CLIP 사용만으로 멀티뷰가 되는 것은 아니다. Point-MAE는 3D 자기지도 모델, PLM은 포인트와 언어를 연결하는 모델, U-Net은 인코더–디코더 구조를 뜻한다. ‘의미 특징’이라는 말이 반드시 텍스트 사용을 뜻하지도 않는다. [3, 4, 6]

## 2. 어떤 방향으로 연구하는가

### 2.1 주요 흐름

| 연구 방향 | 해결하려는 문제 | 대표 사례 |
|---|---|---|
| 고해상도·미세 결함 | 다운샘플링으로 작은 결함 소실, 고밀도 처리 비용 | MiniShift·Simple3D [1] |
| 합성 결함 활용 | 실제 불량 데이터·위치 정답 부족 | PCDiff, AT3D-AD, CASL, PA3AD [2–5] |
| 재구성·표현 개선 | 정상 부위 복원 오차, 미세 구조 학습 부족 | PCDiff, CASL [2, 4] |
| 다중 범주 통합 | 제품 종류별 모델 관리 부담, 범주 간 특징 간섭 | SeDiR, MFF-M3AD [6, 7] |
| 언어와 사전학습 모델 활용 | 결함 의미·종류 활용 | AT3D-AD [3] |

이는 선택한 논문에서 관찰한 분류다. 논문 수를 세어 분야 전체의 비중이나 우열을 검증한 결과는 아니다.

### 2.2 논문별 핵심 메모

**[Simple3D, 3D 데이터, FPFH]** 미세 결함·계산 부담 → 다중 스케일 기하 특징 추출, 국소 집계, 정상 특징과 비교. [1]
- 데이터셋: **MiniShift, Real3D-AD, Anomaly-ShapeNet, MulSen-AD**

**[PCDiff, 3D·렌더링 멀티뷰, CLIP]** 미세 결함·정상 부위 복원 오차 → 미세 결함 합성, 국소·전역 복원 결합. [2]
- 데이터셋: **Anomaly-ShapeNet, Real3D-AD**
- 멀티뷰는 3D에서 렌더링한 영상이다. 국소·전역 복원은 ‘국소 복원 후 나머지를 순서대로 복원’하는 과정이 아니다.

**[AT3D-AD, 3D 데이터, 포인트–언어 정렬]** 이상 지도 부족·위치와 유형 구분 한계 → 물리 기반 결함 합성, 전역·국소 언어 정렬, 위치·유형 공동 학습. [3]
- 데이터셋: **Anomaly-ShapeNet, Real3D-AD, MiniShift, MulSen-AD**

**[CASL, 3D 데이터, U-Net]** 좌표 복원의 기하학적 지름길 → 곡률·좌표 마스킹 사전학습, 합성 결함 분류 미세조정. [4]
- 이상탐지 데이터셋: **Real3D-AD, Anomaly-ShapeNet**
- 일반화 평가: **ScanObjectNN, ShapeNetPart**

**[PA3AD, 3D 데이터, 정상 프로토타입]** 실제 불량 부족 → 물리 기반 결함 합성, 정상 기준과의 특징 차이 학습. [5]
- 데이터셋: **Anomaly-ShapeNet, Real3D-AD**

**[SeDiR, 3D 데이터, Point-MAE·Transformer]** 여러 범주의 특징 혼동 → 범주별 특징 분리 후 특징 재구성. [6]
- 데이터셋: **Real3D-AD, Anomaly-ShapeNet**

**[MFF-M3AD, 3D 데이터, 통합 재구성]** 여러 물체 범주 탐지 어려움 → 다중 스케일 특징 융합·정제·재구성. [7]
- 데이터셋: **Real3D-AD, Anomaly-ShapeNet, MulSen-AD**

## 3. 어떤 데이터셋으로 검증하는가

데이터셋 전체가 아니라 정상 학습 분할을 기준으로 학습하는 경우가 많다. 테스트 분할에는 정상·이상 샘플과 결함 위치 정답이 포함된다. 실제 사용 모달리티·범주 수·전처리는 논문마다 별도 확인한다.

표의 **샘플 수는 정상·이상을 포함한 전체 데이터 기준**이다. 실물 개수와 스캔·합성 샘플 수는 같지 않으므로 단위를 함께 적었다. `미확인`은 이번 조사에서 총수를 확인하지 못한 경우, `해당 없음`은 그 데이터를 제공하는 구성이 아닌 경우다. 영상 수는 RGB·그레이스케일 영상 기준이며 깊이·법선·마스크 수는 따로 설명한다.

| 데이터셋             | 입력 / 실제·합성              | 물체 범주 수 | 전체 샘플 수와 단위                                    | RGB·그레이 영상 수                          | 샘플당 시점 수                           | 평가 초점           | 수치 출처                                                                                         |
| ---------------- | ----------------------- | ------: | ---------------------------------------------- | ------------------------------------- | ---------------------------------- | --------------- | --------------------------------------------------------------------------------------------- |
| MVTec 3D-AD      | RGB+정렬된 3D / 실제         |      10 | 4,147 스캔                                       | 4,147장: 스캔당 RGB 1장 기준                 | 1                                  | 외관·기하 결함        | [공식](https://www.mvtec.com/research-teaching/datasets/mvtec-3d-ad)                            |
| Real3D-AD        | 포인트 클라우드 / 실제           |      12 | 1,254 포인트 클라우드                                 | 해당 없음: 공개 벤치마크는 포인트 클라우드              | 해당 없음: 입력은 3D 점 집합; 원본 획득 시점 수 미확인 | 실제 기하 결함        | [공식](https://github.com/M-3LAB/Real3D-AD)                                                     |
| Anomaly-ShapeNet | 포인트 클라우드 / 합성           |      40 | 1,600 포인트 클라우드                                 | 해당 없음                                 | 해당 없음: 입력은 3D 점 집합                 | 다양한 합성 결함       | [원 논문 기준](https://arxiv.org/abs/2311.14897)                                                   |
| MiniShift        | 실제 정상 형상 기반 / 결함 합성     |      12 | 2,577 포인트 클라우드                                 | 해당 없음                                 | 해당 없음: 입력은 3D 점 집합                 | 고밀도·미세 결함       | [공식](https://hustcyq.github.io/MiniShift-Simple3D)                                            |
| MulSen-AD        | RGB+열화상+3D / 실제         |      15 | 2,035 멀티모달 샘플                                  | 미확인: 모달리티별 총 파일 수 별도 확인 필요            | 미확인: 센서 수를 시점 수로 세지 않음             | 센서 간 상호 보완      | [공식](https://github.com/MapleBoat/MulSen-AD)                                                  |
| Eyecandies       | RGB+깊이+법선 / 합성          |      10 | 15,500 렌더링 샘플: 비공개 테스트 포함, 공식 분할 합산            | 93,000장: 샘플당 RGB 6장으로 계산              | 1: 6장은 시점 변화가 아닌 조명 변화             | 외관·기하·조명        | [공식 분할·형식](https://eyecan-ai.github.io/eyecandies/download)                                   |
| Real-IAD         | 멀티뷰 RGB / 실제            |      30 | 약 30,000 물체 관측 세트: 약 15만 장÷5로 추산, 정확한 세트 수 미확인 | 약 150,000장: 원 논문의 반올림 규모              | 5                                  | 시점별 결함 관측       | [원 논문](https://arxiv.org/abs/2403.12580)                                                      |
| Real-IAD D³      | RGB+의사 3D+포인트 클라우드 / 실제 |      20 | 8,450 멀티모달 샘플                                  | 미확인: RGB와 조명별 촬영 영상 총수 구분 필요          | 1: 상부 관측 기준; 여러 조명은 별도 시점이 아님      | 입력 표현 효용        | [공식](https://realiad4ad.github.io/Real-IAD_D3/), [시점 설명](https://arxiv.org/html/2605.07149v1) |
| SiM3D            | 그레이 영상+3D+메시 / 실제·CAD   |       8 | 333 인스턴스: 원 논문의 실제·합성 구성 합계                    | 미확인: 범주별 시점 수와 실제·CAD 포함 범위 구분 필요     | 12–36: 범주별 상이                      | 멀티뷰 융합·단일 정상 참조 | [원 논문 Table 2](https://arxiv.org/html/2506.21549v1)                                           |
| Real-IAD MVN     | RGB+표면 법선 / 실제          |      20 | 8,450 멀티모달 샘플                                  | 42,250장: 8,450×5로 계산; 법선 맵 42,250개 별도 | 5                                  | 조밀한 의사 3D·미세 결함 | [원 논문 §3.3](https://arxiv.org/html/2605.07149v1)                                              |

- **Anomaly-ShapeNet은 원 논문 버전 기준이다.** 공개 확장 버전의 규모와 섞지 않는다.

### 평가를 읽는 기준

- **Object/Image AUROC:** 물체·영상의 정상/이상 점수 구분 능력.
- **Point/Pixel AUROC:** 점·픽셀의 정상/결함 구분 능력.
- **AUPRO:** 결함 영역별 검출 품질. 허용 오탐률 구간도 확인한다.

## 4. 무엇이 아직 어려운가

### 조사로 내릴 수 있는 판단

조사한 논문의 방향은 세 가지로 묶을 수 있다. [1–7]

- **미세 결함 탐지 개선:** 합성 결함으로 학습을 보강하고 기하 특징·복원 정확도를 개선한다.
- **여러 물체의 통합 탐지:** 범주 간 특징 혼동을 줄여 하나의 모델로 여러 제품을 다룬다.
- **3D·멀티뷰 정보 활용:** 직접적인 기하 분석과 여러 시점의 정보를 활용하는 접근이 병행된다. 

## 5. 핵심 연결

- **필요성:** 형상 결함에는 3D, 관측 누락에는 멀티뷰, 외관·기하 상호 보완에는 센서 융합을 활용한다.
- **연구 방향:** 학습 자료 부족 → 합성 결함 / 미세 구조 손실 → 기하·곡률 활용 / 범주 혼동 → 통합 모델 개선 / 결함 의미·유형 → 포인트–언어 정렬.
- **검증:** 순수 3D는 Real3D-AD·Anomaly-ShapeNet·MiniShift, RGB+ / 3D는 MVTec 3D-AD / 촬영 멀티뷰는 Real-IAD를 우선 구분한다.
