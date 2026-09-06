# WinCLIP: 학습 없이 CLIP으로 이상을 분류하고 위치를 찾는 방법

## Paper Metadata

| Item | Content |
|---|---|
| Title | WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation |
| Authors | Jongheon Jeong, Yang Zou, Taewan Kim, Dongqing Zhang, Avinash Ravichandran, Onkar Dabeer |
| Conference / Journal | CVPR 2023 |
| Year | 2023 |
| Paper link | https://openaccess.thecvf.com/content/CVPR2023/papers/Jeong_WinCLIP_Zero-Few-Shot_Anomaly_Classification_and_Segmentation_CVPR_2023_paper.pdf |
| GitHub / Official code | Original-author release unverified; public reproduction: https://github.com/zqhang/Accurate-WinCLIP-pytorch |
| Reason for investigation | target anomaly label 없이 CLIP의 language prior만으로 이상을 찾는 zero-shot 기준점과, 소수 정상 reference를 추가하는 few-normal-shot 확장을 구분하기 위함. |

## 한 문장 요약

WinCLIP은 정상/이상 상태를 설명하는 prompt ensemble과 여러 크기의 sliding window CLIP feature를 결합해 **학습 없이** anomaly score map을 만들고, WinCLIP+는 여기에 소수의 target 정상 reference image를 더해 성능을 보완한다 [1].

## 문제: 일반 CLIP 분류만으로는 왜 부족한가

일반적인 CLIP zero-shot 분류는 이미지 전체 embedding과 text embedding을 한 번 비교한다. 하지만 산업 이상 탐지에는 다음 문제가 있다 [1].

1. **상태의 문맥 의존성**: `normal`과 `anomalous`는 물체와 검사 맥락에 따라 뜻이 달라진다. 예를 들어 PCB의 missing component와 wood의 crack은 모두 이상이지만, 같은 한 단어로 충분히 표현하기 어렵다.
2. **전역 feature의 한계**: CLIP은 image-level vision-language alignment로 학습됐기 때문에 작은 결함의 pixel-level 위치를 직접 제공하지 않는다.
3. **text만으로 정의하기 어려운 이상**: 물체 방향 반전처럼 정상 reference와 비교해야 정의되는 결함이 있다.

WinCLIP은 1-2를 zero-shot 방식으로 해결하고, WinCLIP+는 3을 소수 정상 reference로 보완한다.

## 핵심 아이디어

![WinCLIP과 WinCLIP+의 개념도. 위쪽은 prompt ensemble과 multi-scale window feature로 language-guided anomaly map을 만드는 WinCLIP이고, 아래쪽은 normal reference feature와의 association을 더하는 WinCLIP+이다.](images/WinCLIP_pipeline.jpg)

```text
정상/이상 상태 단어 × anomaly-inspection prompt template
                  ↓
             CLIP text prototype
                  ↓
query image의 patch / small-window / mid-window feature와 비교
                  ↓
             multi-scale anomaly map (WinCLIP)
                  +
few normal reference feature와의 최근접 거리 (WinCLIP+)
```

### 1. Compositional Prompt Ensemble (CPE)

WinCLIP은 하나의 “normal object” prompt 대신, normal state word와 abnormal state word를 여러 anomaly-inspection template과 조합한다. 각 조합의 text embedding을 평균 내 normal prototype과 anomaly prototype을 만든다 [1].

| 구성 | 역할 |
|---|---|
| state word | `flawless`, `damaged`처럼 정상/이상 상태를 구체화 |
| object label | `bottle`, `cable`처럼 검사 대상 물체를 지정 |
| template | “a photo of a [state] [object] for visual inspection”처럼 inspection 맥락을 제공 |

query image feature가 두 prototype 중 anomaly prototype에 더 가까울수록 image-level anomaly score가 커진다. 이것이 WinCLIP의 **language-guided zero-shot score**다.

### 2. Window-based CLIP feature로 위치 찾기

pixel anomaly map을 만들려면 지역 feature가 필요하다. WinCLIP은 image를 여러 sliding window로 보고, 각 window를 CLIP image encoder에 통과시켜 local window embedding을 얻는다 [1].

- **multi-scale window**: 작은 결함과 큰 결함을 함께 다루기 위해 서로 다른 window 크기를 사용한다.
- **language alignment 유지**: window embedding도 CLIP image encoder의 class token에서 얻으므로 text prototype과 같은 embedding space에서 비교할 수 있다.
- **harmonic aggregation**: 겹치는 window의 score를 pixel마다 모아 anomaly map을 만든다. 정상 쪽 score를 더 민감하게 반영해 false positive를 줄이려는 집계 방식이다.

즉, global CLIP feature 하나로 판단하지 않고 “각 위치 주변을 본 CLIP feature”를 여러 크기로 합쳐 localization을 수행한다.

### 3. WinCLIP+: few-normal-shot reference association

WinCLIP+는 target 범주의 정상 image `K`장(논문에서는 1-4장)을 추가로 받는다. query feature와 normal reference feature memory의 cosine distance를 비교해 visual anomaly map을 계산한다 [1].

```text
reference anomaly score(x, p) = min_r 0.5 · (1 - cosine(feature(x, p), r))
```

- patch, small-window, mid-window feature마다 reference memory를 만든다.
- 세 visual score map을 평균하고, WinCLIP의 language-guided score와 결합한다.
- 따라서 **WinCLIP**은 target image 없이 text만 쓰고, **WinCLIP+**는 target 정상 reference도 쓴다.

## Protocol을 읽는 법

| Condition | target 정상 reference | target anomaly label | 학습 / fine-tuning |
|---|---:|---:|---:|
| WinCLIP zero-shot | 0장 | 사용 안 함 | 없음 |
| WinCLIP+ few-normal-shot | 범주별 1-4장 | 사용 안 함 | 없음 |

이 논문의 few-shot은 anomaly example이 아니라 **few-normal-shot**이다. 따라서 source anomaly dataset에서 adapter를 학습하는 AA-CLIP과도 학습 정보 조건이 다르다.

## 논문이 주장하는 결과

논문 abstract는 MVTec AD에서 WinCLIP zero-shot의 image/pixel AUROC를 91.8%/85.1%로, WinCLIP+ 1-normal-shot의 결과를 93.1%/95.2%로 보고한다 [1]. 이 수치는 WinCLIP+가 정상 reference를 받으면 특히 localization에서 크게 향상될 수 있음을 보여 준다.

## 강점과 해석상 주의점

| 항목 | 내용 |
|---|---|
| 강점 | zero-shot WinCLIP은 target data 학습과 parameter update가 필요 없다. |
| 강점 | CPE가 “이상”이라는 추상 상태를 object와 inspection 문맥으로 더 구체화한다. |
| 강점 | multi-scale window feature가 language-guided score와 local detail을 함께 제공한다. |
| 강점 | WinCLIP+는 1-4장의 정상 image만으로 reference 기반 정보를 보완한다. |
| 주의점 | zero-shot도 대상 물체 이름과 state-word/template 설계가 필요하다. |
| 주의점 | WinCLIP+는 target normal reference를 사용하므로 pure zero-shot 결과와 직접 같은 조건으로 비교하면 안 된다. |
| 주의점 | sliding window를 여러 크기와 위치에서 계산하므로 단일 global CLIP inference보다 계산량이 늘 수 있다. |

## 현재 연구와의 연결

WinCLIP은 source dataset 학습 없이 target에 바로 적용하는 **learning-free zero-shot baseline**이다. WinCLIP+는 target 정상 reference를 쓰는 **few-normal-shot baseline**이다. 따라서 source anomaly supervision을 쓰는 AA-CLIP, target normal train image로 모델을 학습하는 one-class 방법과 비교할 때는 AUROC 외에 `source anomaly supervision`, `target normal reference`, `parameter update`의 세 조건을 분리해 기록해야 한다.

## 참고문헌

[1] Jeong, Jongheon, et al. "WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2023, pp. 19606-19616.
