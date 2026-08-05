## Paper Metadata

| Item | Content |
|---|---|
| Title | *SimpleNet: A Simple Network for Image Anomaly Detection and Localization* |
| Authors | Zhikang Liu, Yiming Zhou, Yuansheng Xu, Zilei Wang |
| Conference / Journal | Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) |
| Year | 2023 |
| Paper link | https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_SimpleNet_A_Simple_Network_for_Image_Anomaly_Detection_and_Localization_CVPR_2023_paper.pdf |
| GitHub / Official code | https://github.com/DonaldRR/SimpleNet |
| Reason for investigation | PatchCore와 비교 가능한 industrial anomaly detection 접근으로, normal memory 최근접 검색 대신 feature-space synthetic anomaly와 discriminator를 사용하는 방법을 확인하기 위함. |

---

# SimpleNet

## 1. 이 논문이 다루는 문제

SimpleNet은 **정상 이미지만으로 학습**해 산업 이미지의 이상 여부와 이상 위치를 찾는 unsupervised anomaly detection 방법이다 [1].

| 구분 | 입력 | 출력 | 제약 | 평가 |
|---|---|---|---|---|
| Image-level detection | 테스트 이미지 1장 | 이미지 이상 점수 | 학습에는 정상 이미지만 사용 | I-AUROC |
| Pixel-level localization | 테스트 이미지 1장 | pixel별 anomaly map | 실제 결함 예시 없이 학습 | P-AUROC |

이 논문은 정상 특징을 직접 저장해 비교하는 방법과 달리, 정상 특징 주변의 경계를 학습하는 방식을 제안한다.

---

## 2. 핵심 아이디어

<img width="686" height="271" alt="image" src="https://github.com/user-attachments/assets/35ac9614-d83f-45c7-9f7a-a0fdc4d59de3" />


정상 이미지에서 얻은 지역 특징에 작은 Gaussian noise를 더해 **가짜 이상 특징(synthetic anomaly feature)** 을 만들고, 정상 특징과 가짜 이상 특징을 구분하도록 discriminator를 학습한다 [1].

```text
정상 이미지
  → 사전학습 backbone의 지역 특징
  → feature adapter
  → 적응된 정상 특징 q
  ├─ 정상 특징 q                 → normal label
  └─ q + Gaussian noise = q⁻    → anomaly label
                                  ↓
                           discriminator 학습
```

테스트 때는 가짜 이상을 만들지 않는다. 테스트 patch의 정상 점수를 discriminator가 계산하고, 그 부호를 반전해 anomaly score로 쓴다.

`patch anomaly score = -Dψ(q)`

---

## 3. 방법 구성

### 3.1 지역 특징 추출

- ImageNet 사전학습 WideResNet-50을 backbone으로 사용함.
- `layer2`와 `layer3`의 feature map에서 지역 특징을 만듦.
- 각 위치 주변의 `p × p` 이웃을 평균 내고, 두 계층의 특징을 결합함.
- 논문의 기본 지역 특징 차원은 1,536임 [1].

### 3.2 Feature adapter

ImageNet 특징을 산업 이미지 도메인에 맞추기 위해 feature adapter `Gθ`를 사용한다.

`q = Gθ(o)`

- `o`: backbone의 지역 특징
- `q`: adapter가 만든 적응된 정상 특징
- 기본 adapter: bias 없는 단일 fully-connected layer
- backbone은 고정하고 adapter와 discriminator를 학습함

### 3.3 Synthetic anomaly 생성

`q⁻ = q + ε`, `ε ~ N(0, σ²)`

- `q⁻`: 가짜 이상 특징
- 논문의 기본 noise standard deviation: `σ = 0.015`
- noise가 너무 크면 경계가 느슨해지고, 너무 작으면 학습이 불안정해질 수 있다고 저자들은 설명함 [1].

### 3.4 Discriminator와 추론

- discriminator는 정상 특징에 높은 정상 점수, 가짜 이상 특징에 낮은 정상 점수를 주도록 학습함.
- 구조: `Linear → Batch Normalization → Leaky ReLU → Linear`.
- 테스트 patch의 점수를 이미지 크기로 보간하고 Gaussian smoothing을 적용해 anomaly map을 만듦.
- anomaly map의 최대 patch 점수를 image-level anomaly score로 사용함.

---

## 4. 논문이 주장하는 결과

MVTec AD 전체 평균에서 논문은 다음 수치를 보고했다 [1, Table 1].

| 방법 | I-AUROC (%) | P-AUROC (%) | 표의 의미 |
|---|---:|---:|---|
| PatchCore | 99.1 | 98.1 | 비교 baseline |
| **SimpleNet** | **99.6** | **98.1** | 논문 보고값 |

- SimpleNet은 image-level I-AUROC 99.6%, pixel-level P-AUROC 98.1%를 보고함.
- 저자들은 같은 하드웨어에서 SimpleNet이 77 FPS이고 PatchCore보다 약 8배 빠르다고 보고함 [1].
- 위 숫자는 **논문 저자 환경의 보고값**이며, 이 note는 재현 결과를 포함하지 않는다.

---

## 5. PatchCore와의 비교

| 항목 | PatchCore [2] | SimpleNet [1] | 현재 연구에 주는 비교 관점 |
|---|---|---|---|
| 정상 정보를 담는 방식 | 대표 정상 patch를 memory bank에 저장 | adapter·discriminator가 정상/가짜 이상 경계를 학습 | memory 기반 vs. parameter 기반 |
| 이상 점수 | normal memory와의 최근접 거리 | discriminator 정상 점수의 반대값 | 거리 기반 vs. 판별 점수 기반 |
| 가짜 이상 사용 | 사용하지 않음 | feature space에서 Gaussian noise로 생성 | 실제 결함 없이 경계를 만드는 방식 |
| 테스트 연산 | 최근접 이웃 검색 필요 | 한 번의 network forward | 추론 비용 비교 대상 |

공통점은 둘 다 사전학습 CNN의 중간 특징을 사용하고, 정상 train 이미지만으로 범주별 탐지를 수행한다는 점이다.

---

## 6. 이 논문을 참고 연구로 남기는 이유

SimpleNet은 현재 PatchCore 중심의 이상 탐지 비교에서 다음 질문을 구체화한다.

1. normal memory의 최근접 검색을 discriminator 점수로 바꾸면 성능과 추론 비용의 trade-off는 어떻게 바뀌는가?
2. 실제 결함을 만들지 않고 정상 특징 주변에 noise를 더한 synthetic anomaly가 실제 결함 탐지에 충분한 경계를 제공하는가?
3. image-level 성능과 pixel-level 위치 추정 성능은 같은 선택에서 함께 개선되는가?

이 note가 제공하는 것은 **후보 방법의 원리와 비교 축**이다. 현재 연구에서 SimpleNet을 채택하거나 성능을 주장하는 근거는 아니며, 그런 판단에는 별도 재현·비교 experiment와 raw result가 필요하다.

---

## 7. 제한과 확인할 점

| 항목 | 논문에서 보이는 제한 또는 확인점 |
|---|---|
| synthetic anomaly의 현실성 | Gaussian-noise 특징이 실제 결함 특징을 직접 모델링하지는 않음 |
| noise 민감도 | `σ` 값에 따라 정상/이상 경계의 난이도가 달라질 수 있음 |
| 사전학습 특징 의존성 | backbone의 표현력이 탐지 성능에 영향을 줄 수 있음 |
| 비교의 공정성 | memory search 시간, 입력 크기, backbone, 데이터 전처리를 함께 통제해야 함 |

따라서 SimpleNet을 PatchCore의 대안으로 비교하려면 동일 데이터셋, 동일 backbone·입력 protocol, image/pixel metric, 그리고 추론 시간 측정이 필요하다.

---

## 8. 한 줄 결론

SimpleNet은 정상 특징 주변에 Gaussian noise로 만든 synthetic anomaly를 이용해 **feature-space 정상/이상 경계**를 학습하고, normal memory 검색 없이 anomaly score를 계산하는 industrial anomaly detection 방법이다.

---

## 참고문헌

[1] Liu, Zhikang, et al. "SimpleNet: A Simple Network for Image Anomaly Detection and Localization." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2023, pp. 20402-20411.

[2] Roth, Karsten, et al. "Towards Total Recall in Industrial Anomaly Detection." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2022, pp. 14318-14328.
