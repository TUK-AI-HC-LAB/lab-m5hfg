# SPADE BTAD 실행

## Paper Metadata

| Item | Content |
|---|---|
| Title | Sub-Image Anomaly Detection with Deep Pyramid Correspondences |
| Authors | Niv Cohen, Yedid Hoshen |
| Conference / Journal | arXiv preprint, 출판 proceedings를 확인하지 못해 arXiv v3 보존 |
| Year | 2020, v3 2021 |
| Paper link | https://arxiv.org/abs/2005.02357 |
| GitHub / Official code | 저자 공식 코드 확인되지 않음. 공개 비공식 PyTorch 코드 https://github.com/byungjae89/SPADE-pytorch |
| Reason for investigation | MuSc Table14 BTAD full-shot 비교군의 Image/Pixel AUROC를 이 PC에서 직접 측정 |

## 질문·방법

BTAD 제품 01·02·03의 정상 train 전체에서 ImageNet 특징을 추출하고, 741장 test의 이상 여부와 위치를 측정한다. 가설은 비슷한 정상 이미지 K개에서 어느 위치에도 가까운 특징을 찾지 못하는 영역이 이상이라는 것이다. SPADE는 추가 gradient 학습이 없다. 모델 학습 대신 정상 특징 gallery를 구축한다.

## 고정 조건

| 항목 | 값 |
|---|---|
| Public code commit | 077c67be21d68a38b4442db7311c87e708728286 |
| Backbone | torchvision WideResNet50×2, IMAGENET1K_V1. 공개 pretrained=True에 대응하는 V1 명시 |
| Train / Test | train [400,399,1000], test [70,230,441], 제품별 full-shot |
| Input | 공개 loader의 shorter side 256 LANCZOS, CenterCrop224, ImageNet mean/std. 제거된 Image.ANTIALIAS를 동일한 LANCZOS로 지정 |
| K / κ | 논문 K=50, dense κ=1. 공개 코드 기본 K=5를 사용하지 않음 |
| Image score | avgpool 2048차원 특징에서 K50 squared L2 평균, 논문 Eq2 |
| Pixel feature | layer1/2/3의 256/512/1024채널을 56×56에 정렬해 1792채널 concatenate, 논문 §3.4. bilinear align_corners=False |
| Pixel score | K50×56×56개의 모든 정상 위치에 대해 exhaustive squared L2 최소, Eq3. coreset/근사검색 없음 |
| Postprocess | 56→224 bilinear, 224→256 cv2.INTER_AREA, Gaussian sigma4 |
| Mask | shorter side256 NEAREST, CenterCrop224, >128. 평가256으로 INTER_AREA resize 후 >0.5 |
| Precision / GPU | RTX5080, FP32, TF32/AMP 미사용. cuDNN benchmark, GPU batch32 특징 추출·chunked matrix matching |
| Seed | local42, 고정 eval 모델이며 데이터 sampling 없음 |

## 공개 구현·논문과의 차이 및 한계

저자 코드 실행으로 표기하지 않는다. 공개 구현의 backbone 및 layer hook·loader 구성을 참조하지만 scoring을 논문 Eq2/3와 §3.4에 맞춰 별도 wrapper로 구현한다. 공개 코드가 사용하는 unsquared 이미지 거리·scale별 최소 거리 평균 대신 squared 거리·concat 특징 최근접 검색을 쓴다. 따라서 공개 코드의 결과와 동일한 구현을 실행했다는 주장도 하지 않는다. checkout은 수정하지 않고 commit·원본 파일 SHA를 기록한다.

논문은 256×256 resize와 224 crop 및 평가256을 기술하지만 특징 정렬 interpolation과 crop을 평가256에 대응하는 방식은 명확히 지정하지 않는다. 이번 실행은 공개 loader의 종횡비 보존 Resize 및 중앙 crop을 따르고, 관찰된 crop을 평가256으로 늘린다. 잘린 테두리를 복원하거나 평가에 포함하지 않는다. 이 선택은 명시적인 protocol 한계이며 MuSc 원래 실행과 수치 동등성을 보장하지 않는다. test 성능으로 조건을 조정하지 않는다.

GPU 검색은 전체 gallery를 유지하고 chunk로 메모리를 제한한다. float32 norm+matrix-product 계산은 float64 직접 차이와 bitwise 동일하지 않다. 각 제품 첫 실제 test 이미지의 3개 위치를 전체 K50 gallery의 float64 brute force와 대조하며 rtol1e-4, atol2e-3를 적용한다. 모든 dense 위치를 별도로 다시 검색하지 않으므로 검증 범위를 과장하지 않는다. 전체 이미지 score와 K50 이웃은 별도 float64 계산으로 검증하고, 모든 pixel 후처리·mask·AUROC는 재계산한다.

## 산출물·실행

- 준비: [prepare_spade.sh](../source/prepare_spade.sh)
- 실행: [run_spade.sh](../source/run_spade.sh), WSL Ubuntu의 기존 patchcore-gpu Python 환경
- 구현: [run_spade.py](../source/run_spade.py)
- 검증: [finish_spade.py](../source/finish_spade.py)
- 결과: [결과 폴더](../source/results/btad_k50_seed42_20261008/)
- Raw: `/home/test/spade_results/btad_k50_seed42_20261008`의 native 특징, prediction npz. 대용량 자료를 Git이나 Obsidian에 복사하지 않음
- Log: `/home/test/spade_results/run_20261008.log`

입력 이미지·라벨·mask는 BTAD 원본 meta.json과 독립 대조한다. raw prediction, 특징·모델 SHA, 제품별 CSV, 비가중 평균, 검증 JSON을 보존한다. 검증 passed 이후에만 MuSc 비교표를 갱신한다. 보고 수치는 논문에서 복사하지 않는다.

## 자체 측정 결과

제품 3개 비가중 평균, 단위 %. ImageNet 고정 모델이며 별도 학습과 test tuning은 없다.

| Product | Test images | Image AUROC | Pixel AUROC |
|---|---|---|---|
| 01 | 70 | 91.7396 | 97.4090 |
| 02 | 230 | 70.0333 | 96.2115 |
| 03 | 441 | 99.6768 | 99.3935 |
| Mean | 741 | 87.1499 | 97.6713 |

[검증](../source/results/btad_k50_seed42_20261008/verification.json) · [CSV](../source/results/btad_k50_seed42_20261008/category_metrics.csv)
