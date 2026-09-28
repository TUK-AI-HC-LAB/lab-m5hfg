# W40 공통 프레임워크 전체 method 실행 결과 (MVTec AD `bottle`)

> GitHub 실행 근거: [초기 12-method script](../../method9/source/common_framework_validation_patch/framework_snapshot/run_all_methods_bottle_wsl.sh), [재실행 script](../../method9/source/common_framework_validation_patch/framework_snapshot/rerun_failed_methods_bottle_wsl.sh), [초기 raw 결과](../../method9/source/common_framework_validation_patch/results/all_methods_initial_20260926/), [정상 validation 재실행 결과](../../method9/source/common_framework_validation_patch/results/all_methods_validation_20260928/).

## 결론

공통 프레임워크에 등록된 12개 방법 중 **11개는 `bottle` 클래스에서 끝까지 실행되어 결과 CSV를 만들었다.** `GLASS`만 실행에 필요한 외부 DTD 텍스처 이미지가 준비되지 않아 학습 시작 단계에서 멈췄다. 이는 모델 코드의 결과가 아니라 데이터 의존성 미충족이다.

이 문서는 "각 방법이 현재 프레임워크에서 한 번 실제로 끝까지 도는가"를 확인한 실행 기록이다. 한 클래스, 한 seed 결과이므로 방법 간 성능 우열이나 논문 재현 성능을 주장하는 표는 아니다.

> 재현 설정 주의: 이 표의 실행은 정상 train 209장을 모두 학습에 썼던 2026-09-26 설정의 결과다. 2026-09-28에 공통 MVTec loader는 정상 train의 90%/10%를 학습/validation으로 나누도록 변경됐다. 따라서 이후 기본 설정 실행은 학습 188장·정상 validation 21장을 사용하며, 이 표의 수치와 직접 섞어 비교하면 안 된다.

## 1. 무엇을 같은 조건으로 실행했나

| 항목 | 설정 |
|---|---|
| 데이터셋 | MVTec AD `bottle` |
| 학습 데이터 | 정상 이미지 209장 |
| 시험 데이터 | 정상 20장 + 이상 63장 = 83장 |
| seed | `0` |
| 공통 실행 환경 | WSL의 `patchcore-gpu` 가상환경, GPU 사용 |
| 설정 기준 | 각 방법의 `configs/<method>.yaml` 기본값을 사용하고, 공통으로 `dataset=mvtec`, `category=bottle`, `num_workers=1`만 지정 |
| 결과 | [`results/`](../../method9/source/common_framework_validation_patch/results/all_methods_initial_20260926/) |

`image AUROC`은 이미지 한 장이 정상/이상인지 맞히는 능력이고, `pixel AUROC`은 이미지 안의 각 픽셀이 이상 위치인지 맞히는 능력이다. `saliency f1`은 예측한 이상 영역을 임계값으로 이진화했을 때 정답 마스크와 얼마나 겹치는지를 나타낸다. 값은 모두 높을수록 좋다.

## 2. 실행 결과

| Method | Image AUROC | Pixel AUROC | Saliency F1 | 실행 상태 | 대략 실행 시간 |
|---|---:|---:|---:|---|---:|
| PatchCore | 1.0000 | 0.9878 | 0.6735 | 완료 | 26초 |
| PaDiM | 1.0000 | 0.9872 | 0.7154 | 완료 | 1분 43초 |
| WinCLIP | 0.9992 | 0.9585 | 0.7052 | 완료 | 2분 13초 |
| COAD | 1.0000 | 0.9925 | 0.6903 | 완료 | 1분 33초 |
| SimpleNet | 0.9944 | 0.8776 | 0.5308 | 완료 | 9분 46초 |
| RD | 1.0000 | 0.9882 | 0.7533 | 완료 | 59분 02초 |
| RD-Orig | 1.0000 | 0.9885 | 0.7595 | 완료 | 2분 30초 |
| PromptAD | 1.0000 | 0.9902 | 0.8024 | 완료 | 7분 26초 |
| Dinomaly | 1.0000 | 0.9934 | 0.6053 | 완료 | 49분 04초 |
| UniAD | 1.0000 | 0.9848 | 0.6863 | 완료 | 43분 03초 |
| AnomalyCLIP (당시 CLIP 재정규화) | 0.8881 | 0.9038 | - | 완료 | 23초 |
| GLASS | - | - | - | **미완료: DTD 데이터 없음** | 시작 직후 중단 |

### 이 표를 읽을 때의 주의점

- `bottle` 하나만 평가했으므로, `1.0000`이 다른 클래스나 전체 MVTec에서도 유지된다는 뜻은 아니다.
- seed가 하나뿐이다. 결과의 안정성은 여러 seed 평균과 표준편차로 확인해야 한다.
- AnomalyCLIP은 ViSA에서 학습된 공식 prompt checkpoint를 불러와 평가했다. 정상 `bottle` 이미지로 해당 방법을 처음부터 학습한 다른 방법들과 출발 조건이 같지 않다. 따라서 이 행을 성능 순위 비교에 쓰면 안 된다.
- 각 방법은 자체 이미지 크기, backbone, epoch, 샘플링 같은 기본 설정이 다르다. 여기서 말하는 "같은 조건"은 같은 데이터 클래스·seed·실행 환경이지, 알고리즘 내부 하이퍼파라미터까지 동일하다는 뜻이 아니다.
- 2026-09-29에 AnomalyCLIP도 WinCLIP과 같이 공통 ImageNet 정규화 tensor를 그대로 입력으로 쓰도록 바뀌었다. 현재 기본 config의 `bottle` 결과는 Image AUROC `0.8897`, Pixel AUROC `0.9027`이다. 위 표의 AnomalyCLIP 행은 변경 전 CLIP 재정규화 실행 기록이므로 현재 기본값과 섞지 않는다.

### 2.1 기존 독립 실행 결과와의 비교 (`bottle`)

아래 표는 이전에 각 구현체를 **독립적으로 실행해 저장한 `bottle` 결과**와 이번 공통 framework 실행 결과를 비교한다. 두 열은 모두 Image AUROC와 Pixel AUROC이지만, 같은 알고리즘이라도 upstream revision, image/mask 전처리, backbone, 학습 epoch, seed, prompt checkpoint가 다를 수 있다. 따라서 `차이`는 성능 우열이 아니라 **공통 framework 포팅 결과가 기존 실행과 얼마나 다른지 확인하는 진단값**이다.

| Method | 기존 독립 실행 Image AUROC | 기존 독립 실행 Pixel AUROC | 공통 framework Image AUROC | 공통 framework Pixel AUROC | Image 차이 (공통 - 기존) | Pixel 차이 (공통 - 기존) | 기존 결과 근거 |
|---|---:|---:|---:|---:|---:|---:|---|
| PatchCore | 100.00% | 98.49% | 100.00% | 98.78% | +0.00%p | +0.29%p | [`PatchCore baseline CSV`](../../method1/source/results/PatchCore_MVTecAD_IM224_WR50_baseline.csv) |
| SimpleNet | 100.00% | 97.76% | 99.44% | 87.76% | -0.56%p | -10.00%p | [`논문 조건 CSV`](../../method2/source/results/SimpleNet_MVTecAD_WR50_paper_protocol_results.csv) |
| RD4AD | 100.00% | 98.70% | 100.00% | 98.82% | +0.00%p | +0.12%p | [`논문 조건 CSV`](../../method3/source/results/RD4AD_MVTecAD_WR50_paper_protocol_results.csv) |
| WinCLIP 0-shot | 98.60% | 85.70% | 99.92% | 95.85% | +1.32%p | +10.15%p | [`0-shot CSV`](../../method7/source/results/winclip_mvtec/zero_shot/results.csv) |
| AnomalyCLIP (VisA → MVTec) | 88.80% | 90.30% | 88.81% | 90.38% | +0.01%p | +0.08%p | [`VisA→MVTec CSV`](../../method6/source/results/anomalyclip_visa_to_mvtec/results.csv) |

#### 해석 범위

- PatchCore·RD4AD·AnomalyCLIP은 `bottle`에서 기존 값과 거의 같다. 특히 AnomalyCLIP은 같은 source-trained prompt checkpoint를 사용했으므로 adapter의 score/map 경로가 기존 실행과 연결되는지 확인하는 근거가 된다.
- SimpleNet과 WinCLIP의 큰 Pixel AUROC 차이는 공통 framework의 기본 config가 기존 독립 실행의 논문 조건·공개 재현 코드 설정과 같지 않기 때문이다. 이 차이를 두 방법 중 어느 쪽이 더 좋다는 근거로 사용하면 안 된다.
- PaDiM·COAD·PromptAD·Dinomaly·UniAD는 저장소에 같은 `bottle` 독립 실행 결과가 없으므로 이 표에 넣지 않았다. GLASS는 이번 공통 framework 실행도 DTD 텍스처 데이터 부재로 미완료다.

### 2.2 SimpleNet의 −10.00%p 차이: 프레임워크에서 달랐던 조건

`SimpleNet`의 Pixel AUROC `97.76% → 87.76%` 차이는 “같은 논문 조건에서 runner만 공통 프레임워크로 바꾼 결과”가 아니다. 두 실행은 정상 학습 이미지 209장, `bottle` test 83장, WideResNet-50 `layer2/layer3`, seed 0은 같지만, 아래의 실제 학습 조건이 달랐다.

| 항목 | 독립 재현: 논문 조건 | 공통 framework 초기 실행 | 결과에 미칠 수 있는 영향 |
|---|---|---|---|
| image 전처리 | resize 256 → center crop 224 | resize 288 → 288×288 입력 | feature map과 patch 위치·개수가 달라짐 |
| batch size | 4 | 8 | BatchNorm을 포함한 discriminator 학습 분포가 달라질 수 있음 |
| adapter linear bias | 없음 | 있음 (`nn.Linear` 기본값) | 논문에 맞춘 feature 변환과 달라짐 |
| adapter optimizer | Adam | AdamW | optimizer 규칙이 달라짐 |
| adapter learning rate | 0.0001 | 0.000001 (`lr=1e-5`의 0.1배) | adapter 갱신 크기가 100배 작음 |
| adapter weight decay | 0.00001 | AdamW 기본값 0.01 | 정규화 세기가 1,000배 큼 |

공통 framework의 `configs/simple.yaml`은 `resize: 288`, `imagesize: 288`, `batch_size: 8`, `lr: 1e-5`를 사용한다. `Trainer_SimpleNet`은 projection에 `AdamW(..., lr * .1)`를 사용하고, `Projection`은 `nn.Linear`의 기본 bias를 그대로 사용한다. 반면 독립 재현은 논문 설정에 맞춰 `256/224`, batch 4, bias 없는 FC, Adam `1e-4`, weight decay `1e-5`로 수정한 공식 clone을 사용했다.

따라서 현재 −10.00%p는 공통 framework 자체의 오류나 SimpleNet의 일반적 약점을 뜻하지 않는다. **서로 다른 SimpleNet protocol의 결과 차이**다. 공통 framework 포팅이 맞는지 검증하려면 위 여섯 조건을 독립 재현과 같게 맞춰 한 번 더 실행하고, 그때의 Pixel AUROC를 97.76%와 비교해야 한다.

- 독립 재현 설정: [`method2/markdown/SimpleNet_reproduction_setup.md`](../../method2/markdown/SimpleNet_reproduction_setup.md)
- 독립 재현 raw CSV: [`method2/source/results/SimpleNet_MVTecAD_WR50_paper_protocol_results.csv`](../../method2/source/results/SimpleNet_MVTecAD_WR50_paper_protocol_results.csv)
- 공통 framework config: [`method9/source/common_framework_validation_patch/framework_snapshot/configs/simple.yaml`](../../method9/source/common_framework_validation_patch/framework_snapshot/configs/simple.yaml)
- 공통 framework adapter 구현: [`trainer_simplenet.py`](../../method9/source/common_framework_validation_patch/framework_snapshot/trainer/trainer_simplenet.py), [`simplenet.py`](../../method9/source/common_framework_validation_patch/framework_snapshot/simplenet.py)

## 3. 정상 validation 분할 후 재실행 비교 (2026-09-28)

기존 실행은 정상 train 209장을 전부 학습에 썼다. 이후 loader를 수정해 같은 209장을 **학습 188장 + 정상 validation 21장**으로 seed 0에서 고정 분할했고, test 83장(정상 20 + 결함 63)은 그대로 두었다. 이 표는 method·category·seed·기본 config를 유지한 채 학습 정상 이미지 수만 바꾼 비교다.

> validation DataLoader는 생성됐지만, 현재 모든 trainer가 validation metric으로 epoch나 hyperparameter를 선택하는 것은 아니다. 따라서 이 표는 validation 기반 model selection의 성능이 아니라, **정상 학습 이미지 10%를 분리했을 때의 재실행 결과**다.

| Method | 상태 | 기존 Image AUROC | 188/21 Image AUROC | Δ Image | 기존 Pixel AUROC | 188/21 Pixel AUROC | Δ Pixel | 기존 F1 | 188/21 F1 | Δ F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PatchCore | 완료 | 1.0000 | 1.0000 | +0.0000 | 0.9878 | 0.9871 | -0.0007 | 0.6735 | 0.6730 | -0.0005 |
| PaDiM | 완료 | 1.0000 | 1.0000 | +0.0000 | 0.9872 | 0.9861 | -0.0011 | 0.7154 | 0.7106 | -0.0048 |
| WinCLIP | 완료 | 0.9992 | 0.9992 | +0.0000 | 0.9585 | 0.9568 | -0.0017 | 0.7052 | 0.6930 | -0.0122 |
| COAD | 완료 | 1.0000 | 1.0000 | +0.0000 | 0.9925 | 0.9916 | -0.0009 | 0.6903 | 0.7006 | +0.0103 |
| SimpleNet | 완료 | 0.9944 | 0.9151 | -0.0793 | 0.8776 | 0.6742 | -0.2034 | 0.5308 | 0.1569 | -0.3739 |
| RD | 완료 | 1.0000 | 1.0000 | +0.0000 | 0.9882 | 0.9894 | +0.0012 | 0.7533 | 0.7615 | +0.0082 |
| RD-Orig | 완료 | 1.0000 | 1.0000 | +0.0000 | 0.9885 | 0.9876 | -0.0009 | 0.7595 | 0.7211 | -0.0384 |
| PromptAD | 완료 | 1.0000 | 1.0000 | +0.0000 | 0.9902 | 0.9897 | -0.0005 | 0.8024 | 0.7968 | -0.0056 |
| Dinomaly | 완료 | 1.0000 | 1.0000 | +0.0000 | 0.9934 | 0.9925 | -0.0009 | 0.6053 | 0.5526 | -0.0527 |
| UniAD | 완료 | 1.0000 | 1.0000 | +0.0000 | 0.9848 | 0.9831 | -0.0017 | 0.6863 | 0.6769 | -0.0094 |
| AnomalyCLIP (당시 CLIP 재정규화) | 완료 | 0.8881 | 0.8881 | +0.0000 | 0.9038 | 0.9031 | -0.0007 | - | - | - |
| GLASS | 실패 | - | - | - | - | - | - | - | - | - |

### 해석

- GLASS를 제외한 11개 방법이 새 split에서 실제 결과 CSV를 만들었다.
- SimpleNet 외 10개 방법은 `bottle`, seed 0에서 Image AUROC 차이가 반올림 네 자리 기준으로 없다. Pixel AUROC와 F1은 방법별로 소폭 달라졌다.
- SimpleNet의 Image AUROC는 약 7.93%p, Pixel AUROC는 약 20.34%p, F1은 약 37.39%p 낮아졌다. 이는 188장 학습 조건에 민감할 가능성을 보이는 **한 번의 관찰**일 뿐, 원인이나 일반성을 확정하지 않는다.
- 모든 방법의 설정이 서로 다르고, `bottle` 한 category·seed 하나만 사용했다. 따라서 이 표는 방법 순위나 validation 분할의 일반 효과를 주장하는 근거로 사용할 수 없다.
- AnomalyCLIP의 현재 기본 입력 정책은 이 표를 만든 뒤 공통 ImageNet 정규화로 통일됐다. 해당 현재 결과는 Image AUROC `0.8897`, Pixel AUROC `0.9027`이며, 정규화 정책 비교는 `W40_AnomalyCLIP_공통_프레임워크_포팅.md`에 기록했다.

### 재현 근거

| 역할 | 경로 |
|---|---|
| 재실행 스크립트 | [`run_all_methods_bottle_validation_wsl.sh`](../../method9/source/common_framework_validation_patch/framework_snapshot/run_all_methods_bottle_validation_wsl.sh) |
| 실행 상태·시간 | [`status.tsv`](../../method9/source/common_framework_validation_patch/results/all_methods_validation_20260928/status.tsv) |
| 방법별 raw CSV | [`results/`](../../method9/source/common_framework_validation_patch/results/all_methods_validation_20260928/) |

## 4. GLASS만 왜 끝나지 않았나

GLASS는 **Generalized Latent Anomaly Synthesis and Segmentation** 방법이다. 정상 MVTec 이미지에 별도 텍스처 이미지(보통 DTD, Describable Textures Dataset)를 섞어 가짜 이상 이미지를 만든 뒤 학습한다. 따라서 MVTec `bottle` 폴더만으로는 학습을 시작할 수 없다.

현재 실행에서 `trainer/GLASS_lib/mvtec.py`가 텍스처 파일 목록을 읽었지만 빈 목록이었다. 그 뒤 `np.random.choice(self.anomaly_source_paths)`가 빈 목록에서 하나를 고르려 하며 다음 오류로 멈췄다.

```text
ValueError: 'a' cannot be empty unless no samples are taken
```

프레임워크 기본 경로는 `/datasets/dtd/images`이다. DTD 이미지가 있는 실제 경로를 준비한 뒤 `anomaly_source_path`에 그 `images` 폴더를 지정하면 GLASS를 같은 프로토콜로 재실행할 수 있다. 이 데이터는 현재 PC/WSL에서 찾지 못했으므로, 임의 다운로드나 다른 이미지 폴더 대체는 하지 않았다.

## 5. 실행 중 발견한 환경 의존성

처음에는 일부 방법이 아래 패키지 부재로 멈췄다. 필요한 패키지를 현재 WSL 환경에 설치한 뒤, 실패한 방법만 같은 조건으로 다시 실행했다.

| 영향 방법 | 부족했던 패키지 | 재실행 결과 |
|---|---|---|
| WinCLIP, COAD, PromptAD | `open_clip_torch` | 완료 |
| RD, RD-Orig | `geomloss`, 이후 `numba` | 완료 |
| GLASS | `albumentations` | 패키지 문제는 해결, DTD 데이터 부재로 미완료 |
| Dinomaly | `colorama` | 완료 |
| PromptAD | `seaborn` | 완료 |

이는 프레임워크의 모든 optional method 의존성을 한 가상환경에 처음부터 설치하지 않았기 때문에 생긴 환경 준비 문제다. 결과 표에는 패키지 설치 후 실제 학습/평가까지 완료된 재실행 결과를 기록했다.

## 6. 재현 근거 (Evidence Map)

### 실행 스크립트와 상태 파일

| 역할 | 경로 |
|---|---|
| 최초 12개 순차 실행 스크립트 | [`run_all_methods_bottle_wsl.sh`](../../method9/source/common_framework_validation_patch/framework_snapshot/run_all_methods_bottle_wsl.sh) |
| 의존성 설치 후 1차 재실행 스크립트 | [`rerun_failed_methods_bottle_wsl.sh`](../../method9/source/common_framework_validation_patch/framework_snapshot/rerun_failed_methods_bottle_wsl.sh) |
| RD/RD-Orig/PromptAD 2차 재실행 스크립트 | [`rerun_remaining_methods_bottle_wsl.sh`](../../method9/source/common_framework_validation_patch/framework_snapshot/rerun_remaining_methods_bottle_wsl.sh) |
| 최초 실행 상태 | [`status.tsv`](../../method9/source/common_framework_validation_patch/results/all_methods_initial_20260926/status.tsv) |
| 1차 재실행 상태 | [`status.tsv`](../../method9/source/common_framework_validation_patch/results/all_methods_initial_20260926/retry_after_dependencies/status.tsv) |
| 2차 재실행 상태 | [`status.tsv`](../../method9/source/common_framework_validation_patch/results/all_methods_initial_20260926/retry_after_dependencies_round2/status.tsv) |

### 결과 CSV 위치

| Method | Raw CSV |
|---|---|
| PatchCore | [CSV](../../method9/source/common_framework_validation_patch/results/all_methods_initial_20260926/patchcore/patchcore__layer2_layer3_/results_patchcore.csv) |
| PaDiM | [CSV](../../method9/source/common_framework_validation_patch/results/all_methods_initial_20260926/padim/padim__layer2_layer3_/results_padim.csv) |
| WinCLIP | [CSV](../../method9/source/common_framework_validation_patch/results/all_methods_initial_20260926/retry_after_dependencies/winclip/winclip__layer2_layer3_/results_winclip.csv) |
| COAD | [CSV](../../method9/source/common_framework_validation_patch/results/all_methods_initial_20260926/retry_after_dependencies/coad/coad__2_3_5_6_7_8_9_/results_coad.csv) |
| SimpleNet | [CSV](../../method9/source/common_framework_validation_patch/results/all_methods_initial_20260926/simple/simple__layer2_layer3_/results_simple.csv) |
| RD | [CSV](../../method9/source/common_framework_validation_patch/results/all_methods_initial_20260926/retry_after_dependencies_round2/rd/rd__layer2_layer3_/results_rd.csv) |
| RD-Orig | [CSV](../../method9/source/common_framework_validation_patch/results/all_methods_initial_20260926/retry_after_dependencies_round2/rd_orig/rd_orig__layer2_layer3_/results_rd_orig.csv) |
| PromptAD | [CSV](../../method9/source/common_framework_validation_patch/results/all_methods_initial_20260926/retry_after_dependencies_round2/promptad/promptad__layer2_layer3_/results_promptad.csv) |
| Dinomaly | [CSV](../../method9/source/common_framework_validation_patch/results/all_methods_initial_20260926/retry_after_dependencies/dinomaly/dinomaly__layer2_layer3_/results_dinomaly.csv) |
| UniAD | [CSV](../../method9/source/common_framework_validation_patch/results/all_methods_initial_20260926/uniad/uniad__layer2_layer3_/results_uniad.csv) |
| AnomalyCLIP | [CSV](../../method9/source/common_framework_validation_patch/results/all_methods_initial_20260926/anomalyclip/anomalyclip__layer24_/results_anomalyclip.csv) |

## 7. 현재 판단과 다음 1개 작업

현재 판단은 "공통 실행 흐름은 GLASS를 제외한 11개 등록 방법을 `bottle`에서 실제 결과 CSV까지 생성할 수 있다"이다. 아직 주장할 수 없는 것은 "12개 방법의 성능 비교가 공정하다" 또는 "전체 MVTec 재현이 끝났다"는 결론이다.

다음 작업은 DTD의 `images` 폴더를 준비하고 GLASS만 다시 실행하는 것이다. 그 결과가 생기면 이 문서의 GLASS 행과 상태를 갱신하면 된다.

## 8. 아직 실행·검증하지 못한 범위

| 항목 | 현재 상태 | 이유 | 다음 확인 |
|---|---|---|---|
| GLASS | 미완료 | 학습용 synthetic anomaly에 필요한 DTD texture image가 없음 | DTD `images` 경로를 준비한 뒤 동일 조건으로 재실행 |
| MVTec 전체 15개 category | 미실행 | 이번 실행은 code path 확인을 위한 `bottle` 1개 category smoke/reproduction 실행임 | category별 CSV 생성과 평균을 별도 실행 |
| 여러 seed | 미실행 | seed 0에서 먼저 split·runner가 동작하는지 확인함 | 여러 seed 평균·표준편차로 안정성 확인 |
| validation 기반 best epoch/checkpoint 선택 | 미구현 | VAL DataLoader만 추가됐고 Trainer별 학습 loop·선택 규칙은 아직 다름 | normal validation 또는 synthetic anomaly 기준을 정의한 뒤 Trainer별 연결 |
| 합성 이상 기반 후보 선택 | 제안 단계 | 합성 방식과 후보 순위가 실제 test 성능과 연결되는지 별도 검증이 필요함 | SWSA 계열 선택 실험 설계·검증 |
| AnomalyCLIP 새 prompt 학습 | 미실행 | 공식 학습은 normal/abnormal label과 pixel mask가 있는 source-labelled dataset을 요구하며, MVTec `train/good`만으로는 supervised prompt loss를 만들 수 없음 | source dataset으로 새 checkpoint를 학습한 뒤 target 평가 |
| AnomalyCLIP 공식 `test.py`와 metric 일치 | 미검증 | adapter의 score/map/CSV 생성은 확인했지만 같은 checkpoint·protocol의 수치 일치는 아직 비교하지 않음 | 공식 `test.py`와 framework adapter를 같은 조건에서 비교 |

따라서 이 문서가 보이는 범위는 "`bottle`, seed 0에서 공통 runner가 method별 결과 CSV를 만드는가"이다. 전체 dataset 성능, 여러 seed 안정성, validation 기반 model selection 효과는 아직 결론 내리지 않는다.
