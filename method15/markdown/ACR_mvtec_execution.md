# ACR MVTec AD 실행

## Paper Metadata

| Item | Content |
|---|---|
| Title | Zero-Shot Anomaly Detection via Batch Normalization |
| Authors | Aodong Li, Chen Qiu, Marius Kloft, Padhraic Smyth, Maja Rudolph, Stephan Mandt |
| Conference / Journal | NeurIPS |
| Year | 2023 |
| Paper link | [공식 논문](https://proceedings.neurips.cc/paper_files/paper/2023/file/8078e8c3055303a884ffae2d3ea00338-Paper-Conference.pdf) |
| GitHub / Official code | [공식 구현](https://github.com/aodongli/zero-shot-ad-via-batch-norm) |
| Reason for investigation | MuSc Table 1의 ACR MVTec zero-shot 행을 이 PC에서 측정한 7개 지표로 채움 |

질문: 공식 ACR의 leave-one-category-out 학습과 batch normalization 기반 추론으로 전체 MVTec AD 결과를 얻을 수 있는가? 실행 전 가설은 라이브러리 호환 및 결과에 쓰이지 않는 메모리 캐시만 수정하면 공식 계산을 보존하며 학습·평가할 수 있다는 것이다. 공식 AUROC와 원시 출력의 재계산 일치를 검증한다.

## 실행 조건

각 target category를 제외한 14개 category의 정상 train 이미지로 모델을 학습하고, target category의 test 전체에 적용한다. 대상 test 라벨·mask는 평가에만 사용한다. 사전학습 WideResNet50-2 ImageNet V1 layer3 특징(1024×14×14)을 먼저 추출한다. Resize256 LANCZOS, CenterCrop224, ImageNet normalization을 사용하고 mask는 NEAREST 보간한다.

공식 기본 설정을 유지한다: 5-layer MLP hidden512/256/128/64/32, 출력32, 학습되는 center32, Adam LR0.0003, 업데이트50회, 업데이트당 task32개, task당 정상30개+Gaussian noise0.1로 오염시킨 특징30개, query anomaly ratio0.5. 특징 추출 seed1024는 공식 설정이고, 학습 seed42는 로컬 재현성을 위한 선택이다. 단일 seed이며 반복 표준편차는 측정하지 않는다.

추론에서도 모델을 train mode로 유지해 공간 위치별 target test 이미지 전체와 Gaussian-corrupted 복사본의 BN 통계를 사용한다. 공식 코드대로 매 업데이트마다 test 평가를 실행하고, 최종50회 모델만 사용한다. 맵은 224 bilinear interpolation, Gaussian sigma4, 카테고리 min-max 정규화하며 image score는 맵 최대값이다. FP32에 TF32와 cuDNN benchmark를 활성화한다. AMP는 사용하지 않는다.

## 호환 수정 및 검증 범위

leather의 Image AP가 sklearn의 부동소수점 누적 결과 `1.0000000000000002`로 계산돼 엄격한0~1 검증에서 실행이 중단됐다. 원시 값을 변경하거나 clipping하지 않고, 유한성 및 범위 검사에1e-12의 수치 오차 허용만 추가했다. [진단](../source/results/mvtec_seed42_20261007/leather/metric_diagnostic.json)에 원래 값과 범위 이탈 항목을 보존한다. 완료된6개 category는 재사용하고, 아직 완료 marker가 없던 leather부터 공식50회 학습을 다시 실행한다.

첫 실행은 특징60개 파일 생성 뒤 extraction용 `data_loader` 검색 경로가 같은 이름의 package를 가려 학습 import에서 종료됐다. 실행 wrapper에서 특징 추출 직후 해당 임시 경로를 제거하도록 수정하고, 기존 특징을 재사용해 다시 시작했다. 실패 traceback과 재시작 출력은 원시 실행 log에 보존한다.

재시작 때 AUPRO 계산에 참조하는 이전 APRIL-GAN checkout도 없어 종료됐다. 기존 측정의 pinned commit `f13b8a634e04f9fde8fa03db125b25af5695d8e1`을 다시 받아 준비 script에 의존성 준비를 추가했다. 이후 실제 bottle 학습의 CUDA 첫 step·finite gradient 기록과 반복 업데이트를 확인했다. [실제 첫 학습 기록](../source/results/mvtec_seed42_20261007/bottle/first_training_step.json)과 training_history.csv는 사전 smoke test와 구분된다.

원본 checkout은 수정하지 않는다. 원본 밖 runtime에서 Pillow의 삭제된 ANTIALIAS를 같은 LANCZOS enum으로 교체하고, 직접 생성한 feature/mask 파일에 현대 torch.load 옵션을 지정한다. feature를 mmap으로 읽고 Gaussian task sampler가 참조하지 않는 cross-category 합치기 캐시를 생략한다. 최종 layer3에 사용하지 않는 layer1/2 CPU 출력 보관도 생략한다. 공식 detect_outliers의 정의되지 않은 optional logger를 None으로 초기화하고 공식 맵·AUROC를 포착하는 기록만 추가한다. 변경 diff와 source SHA-256을 보존한다.

Image/Pixel AUROC는 공식 함수가 계산한다. AP, F1-max 및 AUPRO는 포착한 동일 맵에서 추가 계산한다. AUPRO는 APRIL-GAN의 cal_pro_score(200 thresholds, FPR<0.3) 구현을 사용한다. 최종 AUROC/AP/F1-max는 원시 NPZ에서 독립 재계산하고 image SHA·순서·라벨·체크포인트를 검증한다. 완료 전 수치를 MuSc 비교표에 넣지 않는다.

## 재현·근거

실행 전 독립 smoke test에서 실제 cable train 특징으로 GPU 학습1단계 후 bottle test83장의 공식 평가 함수를 실행했다. loss·gradient·맵의 유효성 검증을 통과했다. 이 test는 전체14개 category·50회 update 학습을 대신하지 않으며 초기 AUROC를 최종 결과로 보고하지 않는다. [GPU 사전 검증](../source/results/mvtec_seed42_20261007/cuda_preflight.json)에 범위를 기록한다.

- commit: `54f1a6026cda4a501d870e49b7d49004b38a16fe`
- sh: [준비](../source/prepare_acr.sh), [실행](../source/run_acr.sh)
- result: [환경·상태](../source/results/mvtec_seed42_20261007/environment.json), [호환 diff](../source/results/mvtec_seed42_20261007/compatibility.patch), [원본·runtime SHA](../source/results/mvtec_seed42_20261007/runtime_sources.json)

실행 명령은 `bash /mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method15/source/run_acr.sh`다. 원시 특징·가중치·맵은 `/home/test/acr_results/mvtec_seed42_20261007`, log는 `/home/test/acr_results/run_20261007.log`에 저장한다. 전체15개 완료 후 평가 검증·이 문서와 MuSc Table 1 갱신을 자동 수행한다. 중단 후에는 완료 category를 보존하고 미완료 category를 처음부터 다시 실행한다.

## 자체 측정 결과

15개 category, 전체 test 1,725장 완료. 수치는 %, Mean은 카테고리 비가중 평균이다. 단일 seed 실행으로 표준편차를 보고하지 않는다.

| Category | Images | image_auroc | image_f1_max | image_ap | pixel_auroc | pixel_f1_max | pixel_ap | aupro |
|---|---|---|---|---|---|---|---|---|
| bottle | 83 | 99.2063 | 97.6744 | 99.7459 | 96.2052 | 64.9299 | 61.4823 | 69.8703 |
| cable | 150 | 73.6694 | 81.1594 | 79.9128 | 87.5109 | 31.6613 | 23.7194 | 58.7163 |
| capsule | 132 | 77.4232 | 92.1739 | 92.7331 | 96.0574 | 41.9910 | 37.3604 | 77.3784 |
| carpet | 117 | 99.9197 | 99.4350 | 99.9753 | 98.0216 | 54.5951 | 44.4217 | 91.3916 |
| grid | 78 | 87.3851 | 88.2883 | 95.6226 | 91.7736 | 29.3960 | 22.2051 | 71.1893 |
| hazelnut | 110 | 80.3214 | 84.0764 | 86.5613 | 92.3259 | 40.1024 | 31.0490 | 73.3982 |
| leather | 124 | 100.0000 | 100.0000 | 100.0000 | 98.5525 | 37.4035 | 30.9491 | 79.9273 |
| metal_nut | 115 | 83.1378 | 91.8367 | 95.1259 | 80.8673 | 48.6869 | 40.9894 | 64.1756 |
| pill | 167 | 73.1042 | 92.4092 | 93.1290 | 91.0033 | 43.9264 | 38.0668 | 72.1242 |
| screw | 160 | 46.0545 | 85.3047 | 76.4471 | 85.5694 | 2.8981 | 1.2197 | 59.9748 |
| tile | 117 | 99.2785 | 97.0760 | 99.7217 | 91.2978 | 52.8339 | 41.7186 | 72.6702 |
| toothbrush | 42 | 98.6111 | 96.5517 | 99.4819 | 97.8792 | 59.1217 | 56.6579 | 74.6676 |
| transistor | 100 | 91.5417 | 80.5556 | 89.6584 | 97.1378 | 67.7692 | 68.0253 | 84.3633 |
| wood | 79 | 98.4211 | 97.4790 | 99.5009 | 89.2475 | 36.7640 | 31.0430 | 81.8746 |
| zipper | 151 | 90.7826 | 94.4000 | 96.0964 | 95.1466 | 52.2518 | 49.4103 | 77.6789 |
| Mean | 1725 | 86.5904 | 91.8947 | 93.5808 | 92.5731 | 44.2887 | 38.5545 | 73.9600 |

[카테고리 CSV](../source/results/mvtec_seed42_20261007/category_metrics.csv) · [검증](../source/results/mvtec_seed42_20261007/verification.json)

원시 맵·점수에서 AUROC/AP/F1-max 6개 지표를 독립적으로 재계산하고 1e-12 이내 일치를 확인했다. Image/Pixel AUROC는 공식 평가 함수 반환값과도 일치한다. AUPRO는 앞선 APRIL-GAN/DRAEM 비교와 같은 200-threshold 근사 구현이며 별도 재계산은 하지 않았다. 논문 성능 수치를 복사하지 않았고, 저자의 수치와 동등하다는 주장은 하지 않는다.
