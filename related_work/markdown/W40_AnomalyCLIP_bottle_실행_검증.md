# W40 AnomalyCLIP `bottle` 실행 검증

> GitHub 근거 파일: [실행 설정](../../method9/source/common_framework_validation_patch/framework_snapshot/experiment_anomalyclip.yaml), [AnomalyCLIP Trainer](../../method9/source/common_framework_validation_patch/framework_snapshot/trainer/trainer_anomalyclip.py), [전용 Dataset](../../method9/source/common_framework_validation_patch/framework_snapshot/datasets/anomalyclip_mvtec.py), [현재 raw CSV](../../method9/source/common_framework_validation_patch/results/anomalyclip_bottle_common_input_20260929/anomalyclip__layer24_/results_anomalyclip.csv). dataset·checkpoint는 용량 때문에 저장소에 포함하지 않는다.

## 1. 실행 목적

공통 framework에 추가한 AnomalyCLIP adapter가 실제 MVTec `bottle` 전체 test split에서 다음 흐름을 끝까지 수행하는지 확인했다.

```text
설정 파일 → Dataset/DataLoader → 공식 AnomalyCLIP checkpoint load
→ image score·anomaly map 생성 → AUROC 계산 → 결과 CSV 저장
```

## 2. 실행 설정

| 항목 | 값 |
|---|---|
| Method | `anomalyclip` |
| Dataset / category | MVTec AD / `bottle` |
| Test images | 83장: 정상 20장, 이상 63장 |
| Prompt checkpoint | ViSA-to-MVTec `epoch_15.pth` |
| Seed | `0` |
| Batch size | `1` |

실행 설정 파일: [`experiment_anomalyclip.yaml`](../../method9/source/common_framework_validation_patch/framework_snapshot/experiment_anomalyclip.yaml).

실행 명령:

```bash
cd /mnt/c/Users/test/Downloads/dinomaly_share_codebase/dinomaly_share_codebase
/home/test/miniforge3/envs/patchcore-gpu/bin/python main.py \
  --config experiment_anomalyclip.yaml
```

## 3. 실행 중 발견·수정한 문제

첫 실행은 `datasets.anomalyclip_mvtec` module이 공통 registry가 요구하는 `DatasetSplit`을 외부에 제공하지 않아 DataLoader 생성 단계에서 중단됐다.

```text
ComponentContractError: datasets.anomalyclip_mvtec must expose DatasetSplit
```

`datasets/anomalyclip_mvtec.py`에서 기존 MVTec module의 `DatasetSplit`을 함께 import하여 export하도록 수정했다. 이 수정은 전용 Dataset의 동작을 바꾸지 않고, 공통 Dataset registry contract만 충족한다.

또한 WSL 실행 environment에 `wandb`가 없었다. `main_run.py`는 `wan: false`여도 시작 시 `wandb`를 import하므로 `wandb`를 설치한 뒤 실행했다. 전체 `requirements.txt` 설치는 현재 평가에 쓰이지 않는 `horovod` build에서 실패했으나, AnomalyCLIP 평가에 필요한 package와 실제 실행에는 영향을 주지 않았다.

## 4. 결과

처음 실행값은 AnomalyCLIP adapter가 OpenAI CLIP 재정규화를 적용하던 시점의 기록이다. 이후 framework 입력 정책을 WinCLIP과 통일해, 현재 기본 config는 공통 ImageNet 정규화 tensor를 그대로 전달한다.

| 입력 정책 | Image AUROC | Pixel AUROC | 결과 CSV |
|---|---:|---:|---|
| 이전: OpenAI CLIP 재정규화 | `0.8880952381` | `0.9037956841` | [raw CSV](../../method9/source/common_framework_validation_patch/results/anomalyclip_bottle_clip_input_20260926/anomalyclip__layer24_/results_anomalyclip.csv) |
| 현재: 공통 ImageNet 정규화 그대로 | `0.8896825397` | `0.9027179073` | [raw CSV](../../method9/source/common_framework_validation_patch/results/anomalyclip_bottle_common_input_20260929/anomalyclip__layer24_/results_anomalyclip.csv) |

생성 파일:

```text
현재 기본 입력 정책의 [CSV](../../method9/source/common_framework_validation_patch/results/anomalyclip_bottle_common_input_20260929/anomalyclip__layer24_/results_anomalyclip.csv).
```

CSV에는 다음 한 행이 저장됐다.

```text
mvtec_anomalyclip_bottle, seed=0, auroc_mean=0.8896825396825397,
pixel_auroc_mean=0.9027179073366489
```

## 5. 해석과 범위

두 입력 정책 모두 공식 AnomalyCLIP source와 prompt checkpoint를 공통 framework의 Dataset·DataLoader·runner에 연결해, `bottle` 전체 test split의 score/map/AUROC/CSV 생성까지 수행했다. 한 category에서 지표 차이는 매우 작았으므로, 현재는 method 간 공통 입력 정책을 우선해 ImageNet 정규화 그대로를 기본값으로 사용한다.

다만 이 수치는 한 category와 한 seed의 실행 결과다. 논문 수치 재현이나 전체 MVTec 평균 성능을 주장하려면 나머지 14개 category 실행과 공식 `test.py` 결과 비교가 필요하다.

## Evidence Map

| 확인한 주장 | 근거 |
|---|---|
| 실제 framework 전체 실행 성공 | [초기 raw CSV](../../method9/source/common_framework_validation_patch/results/anomalyclip_bottle_clip_input_20260926/anomalyclip__layer24_/results_anomalyclip.csv) |
| 이전/현재 입력 정책의 AUROC·CSV 저장 성공 | [이전 CSV](../../method9/source/common_framework_validation_patch/results/anomalyclip_bottle_clip_input_20260926/anomalyclip__layer24_/results_anomalyclip.csv), [현재 CSV](../../method9/source/common_framework_validation_patch/results/anomalyclip_bottle_common_input_20260929/anomalyclip__layer24_/results_anomalyclip.csv) |
| 실행 설정 | [`experiment_anomalyclip.yaml`](../../method9/source/common_framework_validation_patch/framework_snapshot/experiment_anomalyclip.yaml) |
| Dataset registry 오류 수정 | [`datasets/anomalyclip_mvtec.py`](../../method9/source/common_framework_validation_patch/framework_snapshot/datasets/anomalyclip_mvtec.py) |
