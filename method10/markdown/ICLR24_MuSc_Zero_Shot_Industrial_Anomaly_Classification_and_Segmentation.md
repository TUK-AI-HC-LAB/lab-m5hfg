# MuSc 전체 학습 노트: 라벨 없는 테스트 이미지의 상호 비교로 이상 찾기

## Paper Metadata

| Item | Content |
|---|---|
| Title | MuSc: Zero-Shot Industrial Anomaly Classification and Segmentation with Mutual Scoring of the Unlabeled Images |
| Authors | Xurui Li, Ziming Huang, Feng Xue, Yu Zhou |
| Conference / Journal | ICLR 2024 |
| Year | 2024 |
| Paper link | https://proceedings.iclr.cc/paper_files/paper/2024/file/096b1019463f34eb241e87cfce8dfe16-Paper-Conference.pdf |
| GitHub / Official code | https://github.com/xrli-U/MuSc — 논문과 공식 proceedings 페이지에서 연결 확인 |
| Reason for investigation | 정상 학습 데이터나 text prompt 대신 unlabeled test pool을 사용하는 이상 탐지의 원리, 비교 조건, 재현 설정과 한계를 이해하기 위함. |

읽은 범위: 공식 출판 PDF 26쪽 전체(본문, 참고문헌, 부록 A.1–A.8, Tables 1–18, Figures 1–16). 작성일: 2026-10-02. 논문 읽기와 그림 확인을 완료했으며 코드 재현 실험은 수행하지 않았다. 아래 숫자는 저자 보고값이며 자체 실험 결과가 아니다. 근거 원본: `../paper/ICLR24_MuSc_Zero_Shot_Industrial_Anomaly_Classification_and_Segmentation_with_Mutual_Scoring_of_the_Unlabeled_Images.pdf`.

## 1. 문제 설정과 핵심 주장

MuSc는 **정상 패치는 여러 다른 이미지에 반복되지만 이상 패치는 비슷한 대응물을 찾기 어렵다**는 관찰을 이용한다 [1, §1–3]. 이상 이미지에도 정상 영역이 많으므로, 라벨 없는 테스트 이미지 집합 자체가 정상 외관의 비교 자료가 된다. 논문이 집계한 정상 픽셀 비율은 MVTec AD 97.26%, VisA 99.45%이다(p.2). 이는 정상 **이미지** 비율과 다르다.

입력은 정상/이상 라벨이 없는 산업 이미지 집합이고, 출력은 이미지별 이상 점수(AC)와 위치별 이상 점수 지도(AS)다. 주요 평가가 category별 이미지 집합을 비교하는 설정이므로 서로 다른 제품을 무작위로 섞어도 같은 성능이 나온다고 해석하면 안 된다.

정상 target 학습 이미지, target 라벨, text prompt, 추가 최적화 학습 없이 동작한다. 단, 사전학습된 vision backbone은 사용한다. 따라서 '학습이 전혀 없는 모델'이라는 뜻이 아니라 **MuSc 적용을 위한 추가 학습이 없다**는 뜻이다.

테스트 이미지끼리 서로 점수에 영향을 주는 transductive 설정으로 해석할 수 있다. 단일 query를 고정 모델로 독립 판정하는 설정과 사용 가능한 정보가 다르다. 테스트 라벨은 평가에 쓰고 scoring에 쓰지 않는다. 집합이 바뀌면 같은 이미지의 점수도 바뀔 수 있다.

## 2. 용어와 표기

| 기호 / 용어 | 의미 |
|---|---|
| AC / AS | 이미지 단위 anomaly classification / 위치 단위 anomaly segmentation |
| $D_u=\{I_i\}_{i=1}^N$ | 라벨 없는 테스트 이미지 집합 |
| $M,C,L$ | 이미지당 패치 수, feature 차원, 사용하는 ViT stage 수 |
| $r$ | 패치 feature 격자에서 평균낼 $r\times r$ 이웃 크기 |
| $\hat p_{i,l}^{m,r}$ | 이미지 $i$, stage $l$, 패치 $m$의 이웃 집계 feature |
| $d_{ij}^{m,l,r}$ | query 패치와 이미지 $j$의 최근접 패치 사이 거리 |
| IA | Interval Average: 이미지별 거리 중 낮은 비율의 값들을 평균 |
| $c_i,\hat c_i$ | 패치 최대값 기반 이미지 점수, RsCIN 보정 점수 |
| AUROC | 정상보다 이상에 높은 점수를 부여하는 순위 구별 능력 |
| AP | precision–recall 기반 요약. 이상 픽셀이 희소할 때 AUROC와 함께 확인 |
| F1-max | 여러 threshold 중 최대 F1. 배포 threshold가 이미 결정됐다는 뜻은 아님 |
| PRO | 이상 연결 영역별 검출 overlap을 평가. 논문 표기와 AUPRO 표기를 구분해 읽을 것 |

이 노트의 거리 기호 $d$는 설명을 간소화한 표기이며 논문 Eq.1의 $a_{i,l}^{m,r}(I_j)$와 같다.

## 3. 전체 파이프라인

```text
같은 제품 범주의 unlabeled test pool
  → 고정 사전학습 ViT: 여러 stage의 patch token + 마지막 class token
  → LNAMD: 각 stage에서 r=1,3,5의 이웃 평균 feature
  → MSM: 다른 이미지마다 최근접 patch 거리 → 낮은 30% 평균
  → 4 stage × 3 aggregation degree의 점수 평균
  → patch map upsampling: segmentation
  → patch 점수 최대값: 초기 image score
  → RsCIN: class token 이웃의 image score로 보정: classification
```

### 3.1 LNAMD: 서로 다른 크기의 결함 표현 (§3.1, Fig.3)

ViT patch token을 공간 격자로 바꾸고 각 위치에서 $r\times r$ 이웃을 평균낸다. $r=1$은 원래 패치 feature이고 $r=3,5$는 주변 맥락을 포함한다. 입력 이미지를 각각 다른 해상도로 세 번 돌리는 방식과 구분해야 한다.

작은 이웃은 작은 결함을 보존하지만 큰 결함의 내부/전체 구조를 표현하기 어려울 수 있다. 큰 이웃은 큰 결함을 더 잘 표현하지만 작은 결함을 희석할 수 있다. 서로 다른 stage와 degree를 결합해 이 trade-off를 줄인다. feature를 먼저 한 벡터로 합치는 것이 아니라 **각 조합에서 얻은 anomaly score를 평균**한다.

### 3.2 MSM: 다른 이미지들이 query 패치를 평가 (§3.2, Eq.1–3)

각 다른 이미지 $I_j$에 대해:

$$d_{ij}^{m,l,r}=\min_n\|\hat p_{i,l}^{m,r}-\hat p_{j,l}^{n,r}\|_2,\qquad j\ne i.$$

비교 이미지 내부의 모든 위치에서 최근접 패치를 찾는다. 같은 좌표만 비교하지 않는다. 자기 이미지와 비교하면 자기 feature가 0 거리로 매칭되므로 query 이미지 자체를 제외한다.

이제 $N-1$개의 **이미지별 최근접 거리**를 오름차순 정렬하고, 가장 낮은 30%를 평균낸다:

$$a_i^{m,l,r}=\frac{1}{q}\sum_{j\in\mathcal J_{i,m,l,r}}d_{ij}^{m,l,r}.$$

$\mathcal J$는 낮은 거리의 이미지 집합이고 $q$는 선택한 이미지 수다. 정수 반올림이나 최소 이미지 수 처리는 실제 코드 재현 때 확인해야 한다. 각 query 패치마다 선택 이미지가 다를 수 있다.

**두 집계 단위를 혼동하면 안 된다.** 첫 번째 min은 '비교 이미지 내부의 패치'에 적용하고, 두 번째 낮은 30% 평균은 '서로 다른 이미지가 준 점수'에 적용한다. 전체 pool의 모든 patch를 한 memory bank로 합쳐 한 번의 global min을 구하는 방식과 다르다.

모든 거리 평균은 외관이 다른 정상 이미지 때문에 정상 패치 점수도 높일 수 있다. 반대로 단 하나의 가장 낮은 거리만 사용하면 우연히 비슷한 이상 패치 하나가 점수를 낮출 수 있다. IA는 여러 비교 이미지의 지지를 사용하면서 지나치게 먼 정상 변이를 제외하는 절충이다.

예시(설명용, 실험값 아님): 다른 이미지들이 준 거리가 `[0.1,0.2,0.3,0.8,0.9,1.0,1.1,1.2,1.3,1.4]`이면 낮은 30% 평균은 0.2다. 이런 선택은 **거리 순위의 30%**이지 이상 이미지 비율의 30%가 아니다.

최종 패치 점수는:

$$a_i^m=\frac{1}{3L}\sum_{l=1}^{L}\sum_{r\in\{1,3,5\}}a_i^{m,l,r}.$$

점수 격자를 원래 이미지 해상도로 upsample해 anomaly map을 얻고 $c_i=\max_m a_i^m$를 초기 이미지 점수로 사용한다. 최대값은 작은 결함에 반응하지만 정상 이미지의 국소 noise에도 민감하다.

### 3.3 RsCIN: 이미지 이웃으로 분류 점수 보정 (§3.3, Eq.4–5, A.1.2)

마지막 layer의 projected class token으로 이미지 유사도 행렬 $W$를 만들고, 각 이미지의 $k$-nearest image neighbors만 남기는 mask $M_k$를 적용한다. $P_k=D_k^{-1}(M_k\odot W)$를 행 단위 정규화된 이웃 가중치라 하면:

$$\hat C=\frac{C+\sum_{k\in\mathcal K}P_kC}{|\mathcal K|+1}.$$

자기 원래 점수와 각 window에서 계산한 유사 이미지의 가중 평균 점수를 동일한 비중으로 결합한다. 가까운 이웃은 여러 window에 겹쳐 포함되어 더 많은 영향을 줄 수 있다. 작은 정상 noise로 생긴 높은 점수를 낮추고, 비슷한 이상 이미지의 높은 점수로 약한 결함 점수를 올리는 것이 의도다.

RsCIN은 **이미지 분류 점수**를 보정한다. anomaly map 자체를 이웃과 평균내는 단계가 아니다. Eq.5는 한 번의 점수 결합이며 반복 diffusion이나 학습 loss로 해석하면 안 된다. class token 유사도 계산의 normalization, 자기 이웃 포함 여부 등 구현 세부는 논문 수식만으로 단정하지 않는다.

## 4. 재현에 필요한 설정

| 항목 | 논문 설정과 근거 |
|---|---|
| Backbone | OpenAI CLIP ViT-L/14-336, 고정 사전학습 모델 (§4) |
| 입력 | 518×518로 resize (p.7). 모델 이름의 336을 실제 실험 해상도로 혼동하지 않을 것 |
| Stage | 24 layers를 6개씩 4 stage로 나누고 각 stage 출력 사용 |
| Aggregation | $r\in\{1,3,5\}$ |
| IA | 낮은 30% |
| Image feature | 마지막 layer의 linearly projected class token |
| RsCIN window | MVTec AD: {2,3}; VisA: {8,9} |
| 평가 | MVTec AD 15 category, VisA 12 category; 정상/이상 test image 사용 |

CLIP의 text encoder와 normal/anomaly prompt를 사용하지 않는다. DINO/DINOv2 backbone 비교도 있으므로 원리는 CLIP image–text alignment에만 종속되지 않는다(A.2.1).

재현 시 추가 확인: category별 pool 구성, feature normalization, position embedding의 해상도 처리, IA 정수 선택, resize와 mask 보간, map smoothing, class token 이웃 정의, metric 구현/평균 방식, 코드 commit. 공식 코드 링크 확인과 실제 코드 분석/실행은 별개의 완료 조건이다.

## 5. 실험 결과와 무엇을 지지하는가

아래는 모두 원문 Table 1(p.7)의 값, 단위 %. 자체 재현 raw result는 없으며 **출판 표 자체가 근거**다.

| Dataset | image AUROC | image F1-max | image AP | pixel AUROC | pixel F1-max | pixel AP | PRO |
|---|---:|---:|---:|---:|---:|---:|---:|
| MVTec AD | 97.8 | 97.5 | 99.1 | 97.3 | 62.6 | 62.7 | 93.8 |
| VisA | 92.8 | 89.5 | 93.5 | 98.8 | 48.8 | 45.1 | 92.7 |

원문 Table 1에서 각 metric의 당시 비교 zero-shot 최고값 대비 MVTec AD image AUROC +6.0%p, pixel AUROC +4.8%p, pixel AP +21.9%p, PRO +21.1%p다. VisA image AUROC +14.7%p, pixel AUROC +4.6%p, pixel AP +19.4%p다. 상대 개선율과 percentage point를 구분한다.

**원문 내부 불일치:** 초록은 VisA +14.7%를 pixel-AUROC라고 적지만 Table 1의 수치에서는 image-AUROC(92.8−78.1)에 해당한다. pixel-AUROC는 98.8−94.2=4.6%p다. Table 3/4와 일부 부록은 VisA pixel AUROC 98.7을 보고하지만 주요 Table 1/18은 98.8이다. 평균/반올림/실험 조건 차이 원인은 원문만으로 확정할 수 없다.

많은 4-shot 방법과 경쟁력이 있지만 모든 방법의 모든 지표보다 높지는 않다. 예를 들어 MVTec AD GraphCore의 pixel AUROC는 97.4로 MuSc 97.3보다 높고, VisA 4-shot APRIL-GAN image AP 94.5도 MuSc 93.5보다 높다(Table 1). full-shot PatchCore는 MVTec AD image/pixel AUROC 99.6/98.2로 MuSc보다 높다(Table 2). 따라서 '정상 데이터 없이도 강력한 benchmark 성능'까지는 지지되지만 'full-shot을 완전히 대체'한다고 결론낼 수 없다.

### 핵심 ablation

| 검증 | 원문 결과 | 해석 |
|---|---|---|
| LNAMD (Table 3) | MVTec pixel AUROC r={1}:94.6 → {1,3,5}:97.3 | 다중 이웃 크기가 localization에 유효. 다만 모든 metric의 단일 최고 조합은 아님 |
| MSM (Table 4) | MVTec image/pixel AUROC 전체 mean 95.0/96.8 → 낮은 30% mean 97.8/97.3 | 먼 정상 변이를 제외하되 다수 이미지 근거를 유지하는 IA를 지지 |
| RsCIN (Table 5) | MVTec image AUROC 97.4→97.8; VisA 90.0→92.8 | image-level 보정의 기여, 특히 VisA에서 큼 |
| Pool 분할 (Table 7) | MVTec 3분할 image/pixel AUROC 96.7/97.3; VisA 92.3/98.6 | 비교 pool 축소로 효율을 얻지만 classification에 손실 가능 |

30%나 {1,3,5}가 모든 metric에서 항상 최적은 아니다. Fig.7에서 일부 10% 결과가 특정 metric에서 더 높고, Table 3에서 {1,3}의 image AUROC가 더 높다. 저자는 AC/AS와 두 dataset 사이의 종합적인 절충을 선택했다.

### 비용 (Tables 6,13; RTX 3090)

ViT-L/14-336 설정에서 전체 pool은 이미지당 998.8ms, 최대 GPU 메모리 7168MB, 3분할은 513.5ms/5026MB다. 이는 해당 실험 환경의 batch/pool 기반 평균 비용이며 단일 이미지 온라인 latency를 보장하는 값이 아니다. 단순 구현에서는 각 조합마다 이미지 쌍의 패치 거리 비교가 필요하므로 대략 $O(L|R|N^2M^2C)$ 거리 계산량이 발생할 수 있다(수식에 근거한 분석, 논문이 명시한 complexity 식은 아님). 실제 메모리는 chunking과 feature 저장 방식에 달려 있다.

## 6. 부록에서 추가로 확인한 내용

- **A.1:** MSM의 3단계와 RsCIN의 normalized neighbor score 결합을 자세히 유도한다.
- **A.2.1/Table 8:** DINO/DINOv2에서도 효과가 있고 backbone에 따라 AC/AS trade-off가 달라진다. CLIP이 모든 segmentation 결과의 최고인 것은 아니다.
- **A.2.2/Fig.9/Table 9:** 큰 r를 계속 추가하면 작은 결함이 희석되고 분류 성능이 내려간다. multi-scale을 무조건 많이 쓰는 것이 해결책은 아니다.
- **A.2.3/Table 10:** 낮은 30% 중 최하위 2–10%를 제거해 반복 이상 영향을 줄이려 했지만 대체로 성능이 악화된다. 드문 정상 패치에 생기는 false positive 증가가 문제라고 해석한다.
- **A.2.4/Table 11:** RsCIN을 다른 방법에도 붙일 수 있으나 모든 metric의 개선을 보장하지 않는다. DRAEM AUROC는 98.0→97.9이며 PatchCore의 AP/F1은 동일하다. 추가 image backbone이 필요할 수도 있다.
- **A.3/Table 12:** 방향/크기 변화에 취약하다. screw image AUROC 83.5, macaroni2 69.9로 평균만 보고 안정성을 판단하면 안 된다.
- **A.4/Table 13:** 작은 backbone과 subset 분할로 시간/메모리 절감 가능. speed–accuracy 비교에서는 backbone도 함께 명시해야 한다.
- **A.5/Table 14:** BTAD image/pixel AUROC 94.8/97.3. 저자는 MVTec hyperparameter를 그대로 적용했다고 보고한다.
- **A.6/Tables 15–16:** 정상 reference를 기존 MuSc pool에 추가하는 방식은 개선이 작다. MuSc+는 정상 reference만 사용하고 이미지 간 IA 대신 min을 적용한다. 1-shot MuSc+ image AUROC 92.7은 zero-shot MuSc 97.8보다 낮지만 full-shot MuSc+는 99.5/98.5(image/pixel)다. MuSc+를 단순 'MuSc에 reference 추가'로 설명하면 안 된다.
- **A.7/Tables 17–18:** category별 상세 지표. VisA macaroni2는 pixel AUROC 97.2지만 pixel AP 4.5, F1-max 12.4다. 높은 pixel AUROC만으로 작은 결함의 정밀한 검출을 주장하기 어렵다.
- **A.8/Figs.11–16:** 두 dataset 전체 category의 다양한 결함 크기/종류에 대한 정성 결과. 시각적 예시는 유용하지만 failure 빈도를 정량 보장하지 않는다.

## 7. 한계: 저자가 확인한 것과 추가 추론

**논문이 직접 보여준 한계:** 큰 pool의 비용 증가, 방향/scale 변화에서의 약화, 큰 aggregation degree의 작은 결함 희석, 일부 category의 낮은 AC/AP, RsCIN의 제한적 개선과 추가 backbone 비용.

**원리에서 도출한 검증 필요 가설:**

1. 같은 결함이 많은 이미지에서 비슷하게 반복되면 낮은 30%에 이상끼리의 매칭이 충분히 들어가 false negative가 늘 수 있다. 이상 이미지 비율 하나보다 결함 외관의 반복성과 정상 대응물 존재 여부가 중요하다.
2. 희귀한 정상 subtype이 pool에 충분히 없으면 정상 패치도 이상처럼 높은 점수를 받을 수 있다.
3. pose, illumination, background, 제품 variant가 섞이면 feature 이웃과 정상/이상 여부가 일치하지 않을 수 있다.
4. RsCIN의 가까운 이웃이 실제로 다른 상태라면 score averaging이 작은 결함을 희석할 수 있다.
5. singleton query는 논문의 MSM을 그대로 적용할 수 없다. streaming 서비스에서는 비교 pool을 유지하고 pool 갱신에 따른 score/threshold 안정성을 검증해야 한다.

위 가설은 이번에 실험으로 확인하지 않았다. 논문의 benchmark 성공을 반복 결함이나 category 혼합 조건의 강건성으로 확대 해석하지 않는다. Fig.6의 일부 logical anomaly 성공도 모든 논리 이상에 대한 일반적인 해결을 의미하지 않는다.

## 8. 기존 방법과의 연결 및 현재 연구에서의 의미

| 접근 | 판정 기준의 출처 | 비교 시 유지할 조건 |
|---|---|---|
| PatchCore | 라벨이 있는 정상 train image의 patch memory | reference 수, backbone, 정상 학습 split |
| WinCLIP zero-shot | 정상/이상 text prompt와 CLIP feature | prompt, 입력 해상도, window feature |
| AnomalyCLIP 계열 | 별도 학습을 거친 anomaly-aware prompt/feature | auxiliary 학습 데이터와 target 정보 사용 |
| MuSc | unlabeled test pool 내 다른 이미지의 patch 및 image score | category별 pool, pool 크기/구성, test-time 공동 비교 |

MuSc를 기존 개별 이미지 inference framework에 추가한다면, `predict(image)`만으로는 원래 프로토콜을 표현하기 어렵다. `predict_pool(images)` 또는 reference pool을 준비하는 별도 단계가 필요하다는 것이 현재 구현 검토의 출발점이다. 이것은 기존 코드에 대한 패치나 실행 검증을 완료했다는 뜻이 아니다.

향후 원리 검증의 우선순위는 (1) pool 크기를 줄일 때 정상 subtype 부족과 비용 효과를 분리, (2) 동일 결함 반복 비율을 제어해 IA의 실패 조건 확인, (3) RsCIN on/off에서 작은 결함과 pose 변화별 AC/AP 비교다. 구현 변경 전에 각 실험의 질문, 기대 결과, raw result 경로를 고정해야 한다.

## 9. 이해 확인용 핵심 답변

- 왜 abnormal image도 reference인가? 이상 이미지 대부분의 위치가 정상이고 이상 외관의 반복성은 제한적이라는 관찰 때문이다.
- 30%는 무엇의 비율인가? query 패치에 대해 다른 이미지들이 준 최근접 거리 중 낮은 순위의 비율이다.
- 자기 이미지 비교를 왜 제외하나? 자기 패치가 정확히 매칭되어 거리가 0이 되는 것을 막는다.
- spatial alignment가 필요한가? 같은 좌표로 제한하지 않지만 pose/scale 변화에 대한 feature 불변성을 보장하지 않는다.
- segmentation과 classification은 어떻게 연결되나? patch map의 max가 초기 AC score이고 RsCIN은 AC만 보정한다.
- CLIP 없이 가능한가? DINO/DINOv2 ablation이 있다. pretrained visual features는 여전히 필요하다.
- zero-shot인데 왜 데이터가 필요한가? labeled target train data는 없지만 공동 scoring용 unlabeled test pool이 입력이다.
- 성능 개선의 의미는? 논문 당시 비교표 및 그 정보 접근 조건에서의 결과이며 현재 모든 방법에 대한 우월성을 뜻하지 않는다.

## 참고문헌

[1] Li, Xurui, et al. "MuSc: Zero-Shot Industrial Anomaly Classification and Segmentation with Mutual Scoring of the Unlabeled Images." International Conference on Learning Representations, 2024.

근거의 우선순위: 제공된 ICLR 2024 공식 출판 PDF → 공식 proceedings metadata → 논문이 연결한 공식 코드 repository. 이후 확장판 MuSc-V2의 내용은 이 노트에 혼합하지 않았다.
