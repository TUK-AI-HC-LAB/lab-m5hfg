# W40 AnomalyCLIP 공통 프레임워크 포팅

## 1. 목적과 결론

AnomalyCLIP을 DINOMALY shared codebase의 기존 모델 교체 구조에 등록했다. 이제 설정에서 아래처럼 선택할 수 있다.

```yaml
method: anomalyclip
```

포팅은 공식 AnomalyCLIP 모델을 다시 작성하거나 복사하는 방식이 아니다. 공식 repository의 수정 CLIP 구현과 source-trained prompt checkpoint를 그대로 사용하고, 공통 framework의 Dataset/DataLoader/runner가 요구하는 입출력 형태로 바꾸는 adapter를 추가했다.

현재 검증된 범위는 다음과 같다.

| 항목 | 상태 | 근거 |
|---|---|---|
| registry에서 `anomalyclip` 선택 | 완료 | `component_registry.py::METHOD_REGISTRY` |
| 공통 Trainer contract 검사 | 완료 | `MethodSpec.load_class()` 성공 |
| 공식 CLIP + prompt checkpoint 로드 | 완료 | WSL smoke test 성공 |
| 공통 batch 형식 한 장의 score/map 생성 | 완료 | score 1개, map `[1,518,518]` 생성 |
| 전체 `main.py → DataLoader → CSV` 실행 | 미완료 | 전체 test split 및 CSV 생성은 아직 미검증 |
| 공통 MVTec train split만으로 prompt 재학습 | 의도적으로 미지원 | 공식 학습은 labelled source anomaly data가 필요 |

## 2. 포팅 전 구조와 문제

기존 framework에는 11개 방법이 registry에 등록돼 있었지만 AnomalyCLIP은 없었다.

```text
method 문자열
  → component_registry.METHOD_REGISTRY
  → Trainer class
  → main_net.net()
  → main_run.run()
```

AnomalyCLIP은 일반 PatchCore/SimpleNet과 다른 점이 있다.

1. 일반 방법은 `backbones.py`에서 backbone을 받지만, AnomalyCLIP은 공식 source 안에서 수정된 ViT-L/14 CLIP을 직접 생성한다.
2. score는 일반 patch embedding distance가 아니라 normal/abnormal prompt text feature와 image/patch feature의 similarity다.
3. 공식 prompt learner는 source dataset의 normal/abnormal image와 pixel mask로 학습한다. 공통 MVTec training split은 normal image만 제공하므로, 그대로 재학습하면 공식 protocol을 만족하지 못한다.

따라서 “AnomalyCLIP을 Trainer 하나로 등록”하는 것만으로는 부족하다. 자체 backbone 처리, input normalization 변환, checkpoint 기반 prompt 사용, 공통 평가 tuple 반환을 함께 구현해야 한다.

## 3. 변경 파일과 역할

| 파일 | 변경 | 입력 | 처리 | 출력 |
|---|---|---|---|---|
| `component_registry.py` | `anomalyclip` MethodSpec 등록, `uses_backbone` flag 추가 | method 문자열 | 공식 trainer 위치와 자체-backbone 여부 선택 | `Trainer_AnomalyCLIP` class |
| `main_net.py` | `uses_backbone=False` 처리 | MethodSpec | 공통 backbone 대신 name/seed만 가진 placeholder 전달 | adapter가 받을 runner 호환 backbone 정보 |
| `datasets/anomalyclip_mvtec.py` | AnomalyCLIP 전용 MVTec Dataset variant | MVTec 원본 image/mask | Resize(short edge) → CenterCrop → 공통 sample dict | 공식 평가 공간 전처리에 맞는 batch |
| `trainer/trainer_anomalyclip.py` | 새 adapter Trainer | 공통 DataLoader batch | 공식 CLIP/prompt 호출, score/map 변환, AUROC 계산 | 공통 Trainer contract 결과 |
| `configs/anomalyclip.yaml` | AnomalyCLIP 전용 설정 | image size, source/checkpoint 경로 등 | method config로 args 확장 | adapter 실행 설정 |
| `README.md` | 12번째 method와 실행 전제 조건 기록 | 사용자 설정 | 공식 source/checkpoint 준비 방법 안내 | 재현 가능한 실행 안내 |

## 4. Registry와 모델 교체 연결

### 4.1 등록 코드

```python
"anomalyclip": MethodSpec(
    "trainer.trainer_anomalyclip",
    "Trainer_AnomalyCLIP",
    "anomalyclip",
    uses_backbone=False,
)
```

입력은 `method: anomalyclip` 문자열이다. registry는 module과 class name을 찾아 lazy import한다. 출력은 `Trainer_AnomalyCLIP` class다.

`uses_backbone=False`가 핵심이다. 이 값이 없으면 `main_net.py`가 일반 WideResNet 같은 공통 backbone을 먼저 load하려 한다. AnomalyCLIP은 공식 source의 ViT-L/14 모델을 `Trainer_AnomalyCLIP.load()` 안에서 직접 만들므로 그 경로를 건너뛴다.

`DATASET_METHOD_VARIANTS`는 `(mvtec, anomalyclip)` 조합을
`datasets.anomalyclip_mvtec.MVTecDataset`으로 자동 교체한다. 사용자는
`dataset: mvtec`를 유지하되 AnomalyCLIP을 선택했을 때만 공식 전처리 variant를 사용한다.

### 4.2 main_net 변경

```text
일반 method:
  main_net → method_spec.load_backbone(...) → Trainer.load(backbone=실제 backbone)

AnomalyCLIP:
  main_net → placeholder(name, seed) → Trainer_AnomalyCLIP.load(...)
          → 공식 AnomalyCLIP source가 ViT-L/14 CLIP 생성
```

placeholder는 모델 연산에 쓰이지 않는다. `main_run.py`가 모든 Trainer에 대해 `trainer.backbone.seed`를 읽는 기존 구조와 호환되도록 name/seed만 제공한다.

## 5. Adapter의 공통 Trainer contract 검토

registry가 요구하는 public API는 아래 다섯 가지다.

| 함수 | AnomalyCLIP adapter 구현 | 역할 |
|---|---|---|
| `load()` | 구현 | 공식 CLIP, DPAM, prompt learner, checkpoint 준비 |
| `train()` | 구현 | checkpoint 준비 확인 후 공통 test 평가 실행 |
| `predict()` | 구현 | image score와 anomaly map 생성 |
| `get_evaluation_metrics()` | 구현 | image/pixel AUROC dict 반환 |
| `set_model_dir()` | 구현 | 공통 runner의 저장 경로 API 대응 |

따라서 `validate_trainer_class()`는 AnomalyCLIP adapter를 기존 PatchCore/SimpleNet과 같은 방식으로 허용한다. 내부 알고리즘은 달라도 runner가 요청하는 함수 이름과 출력 형태가 같다는 것이 contract 충족의 의미다.

## 6. 입력 정규화 문제와 해결

공통 `BaseDataset`은 image를 ImageNet mean/std로 정규화한다. 반면 공식 AnomalyCLIP은 OpenAI CLIP mean/std를 사용한다.

```text
공통 DataLoader image
  = (pixel - ImageNet mean) / ImageNet std
       ↓ adapter가 역변환
pixel
       ↓ adapter가 재정규화
AnomalyCLIP input
  = (pixel - OpenAI CLIP mean) / OpenAI CLIP std
```

`Trainer_AnomalyCLIP._to_clip_normalized()`가 이 변환을 수행한다. 이 보정이 없으면 같은 이미지라도 공식 AnomalyCLIP이 학습·평가에 사용한 입력 분포와 달라져 score가 신뢰하기 어려워진다.

공간 크기도 별도로 맞췄다. 일반 공통 MVTec Dataset은 image를 바로 정사각형으로 resize하지만, 공식 AnomalyCLIP test transform은 `Resize(short edge) → CenterCrop`이다. 새 `anomalyclip_mvtec` variant가 image와 mask 모두에 같은 공간 변환을 적용한다. 따라서 adapter map과 ground-truth mask가 같은 `518×518` 좌표계에서 비교된다.

## 7. score와 anomaly map 생성

### 입력

공통 DataLoader가 batch dictionary를 제공한다.

```text
image       [B,3,518,518]  ImageNet-normalized image
mask        [B,1,518,518]  pixel ground truth
is_anomaly  [B]            image ground-truth label
```

### 처리

1. image를 CLIP normalization으로 변환한다.
2. 공식 `model.encode_image()`가 image feature와 지정 layer의 patch feature를 만든다.
3. prompt learner가 normal/abnormal text feature `[2,768]`를 만든다.
4. image feature와 text feature similarity의 softmax abnormal column을 image score로 쓴다.
5. patch feature와 text feature similarity를 layer별 spatial map으로 바꾼다.
6. layer map을 더하고 bilinear resize 및 Gaussian smoothing을 적용한다.

### 출력

```text
scores  [B]          image anomaly score
maps    [B,518,518]  pixel anomaly map
labels  [B]          is_anomaly 정답
masks   [B,518,518]  pixel 정답 mask
```

이 tuple은 공통 Trainer의 `predict()` 반환 형태와 같다. adapter의 `_compute_metrics()`는 labels가 정상/이상 둘 다 있을 때 image AUROC를, mask에 정상/결함 pixel 둘 다 있을 때 pixel AUROC를 계산한다.

## 8. 공식 checkpoint와 학습 범위

기본 config의 `anomalyclip_checkpoint_path`는 빈 문자열이다. 사용자는 공식 AnomalyCLIP source dataset에서 학습한 prompt checkpoint를 명시해야 한다.

```yaml
method: anomalyclip
anomalyclip_source_root: /home/test/anomalyclip
anomalyclip_checkpoint_path: /home/test/anomalyclip_checkpoints/visa_to_mvtec/epoch_15.pth
```

이 제약은 구현 누락이 아니다. AnomalyCLIP 공식 학습은 source의 normal/abnormal label과 pixel mask를 이용해 prompt learner를 최적화한다. 공통 MVTec train split은 `train/good`만 읽으므로, 그 data만으로 공식 supervised prompt loss를 재현하면 이상 class 학습 신호가 없다. 현재 adapter는 검증된 source checkpoint를 target dataset에 평가하는 cross-dataset/zero-shot 경로를 명시적으로 지원한다.

## 9. 직접 검증한 결과

검증 환경:

```text
official source: /home/test/anomalyclip
upstream revision: 3911738
prompt checkpoint: /home/test/anomalyclip_checkpoints/visa_to_mvtec/epoch_15.pth
runtime: Python 3.12, PyTorch 2.11.0+cu128, CUDA
```

| 검증 | 결과 |
|---|---|
| `get_method_spec("anomalyclip").load_class()` | `Trainer_AnomalyCLIP` 반환 |
| `main_net.net(args)` 조립 | `get_simplenet` hook이 `Trainer_AnomalyCLIP` 1개 생성 |
| 자체-backbone 분기 | placeholder `backbone.name="anomalyclip"`, `seed=None` 확인 |
| 공식 CLIP weight load | 성공 |
| checkpoint load | 성공, `checkpoint_loaded=True` |
| text feature | `[2,768]` 생성 |
| AnomalyCLIP 전용 MVTecDataset batch | 공통 key 9개, image/mask 각각 `[1,3,518,518]` / `[1,1,518,518]` |
| MVTec bottle 이미지 한 장 | score 1개 생성 |
| anomaly map | `[1,518,518]` 생성 |
| score finite 검사 | 통과 |
| 공통 lifecycle | 정상 1장 + 이상 1장으로 `run_trainer_lifecycle()` 성공 |
| lifecycle sanity metric | image AUROC 1.0, pixel AUROC 0.9677 |

마지막 두 수치는 두 장만 사용한 contract smoke test 결과이므로 성능 claim이 아니다. 이 값은 score와 map, label과 mask, metric dictionary가 끝까지 연결된다는 사실만 보여 준다.

## 10. 남은 문제와 다음 검증

### 아직 확인해야 할 항목

1. 공통 DataLoader 전체 test split에서 category별 CSV가 생성되는지.
2. adapter metric과 공식 `test.py` metric이 같은 checkpoint/target protocol에서 허용 오차 내 일치하는지.
3. `save_segmentation_images=True`가 base Trainer 전용 helper를 호출하지 않도록 runner 분기 또는 adapter 저장 API가 필요한지.
4. source-labelled Dataset을 공통 registry에 별도로 등록해 공식 prompt 학습까지 framework lifecycle으로 지원할지.

현재 포팅은 “공통 runner에서 공식 source-trained AnomalyCLIP checkpoint를 로드해 score/map을 생성하는 평가 adapter”까지 구현·검증됐으며, 전체 CSV run은 별도 실행 검증이 필요하다.

## 11. 공식 원본 AnomalyCLIP 파일과 함께 사용하는 방법

공식 원본을 framework 안으로 복사할 필요는 없다. 공식 원본은 모델과 학습된 prompt checkpoint를 제공하고, framework의 adapter가 그것을 불러 공통 Dataset·CSV 결과 형식에 연결한다.

### 실행 순서

1. 공식 원본 폴더와 `epoch_*.pth` checkpoint를 준비한다.
2. framework를 실행할 Python environment에 `requirements.txt`를 설치한다.
3. framework root에 `experiment_anomalyclip.yaml`을 만들고, 원본 폴더와 checkpoint 경로를 적는다.

```yaml
method: anomalyclip
dataset: mvtec
category: bottle
data_path: /home/test/data/mvtec
results_path: /home/test/anomalyclip_framework_results
seed: 111
batch_size: 1

# 공식 원본과 source-trained prompt checkpoint를 연결하는 두 경로
anomalyclip_source_root: /home/test/anomalyclip
anomalyclip_checkpoint_path: /home/test/anomalyclip_checkpoints/visa_to_mvtec/epoch_15.pth
```

4. 다음 명령으로 공통 runner를 실행한다.

```bash
cd /mnt/c/Users/test/Downloads/dinomaly_share_codebase/dinomaly_share_codebase
/home/test/miniforge3/envs/patchcore-gpu/bin/python main.py \
  --config experiment_anomalyclip.yaml
```

`anomalyclip_source_root`는 `AnomalyCLIP_lib/`와 `prompt_ensemble.py`가 들어 있는 공식 원본의 최상위 폴더여야 한다. checkpoint는 공식 `train.py`가 만든 `epoch_*.pth` 파일을 사용한다.

```text
experiment_anomalyclip.yaml
  → 공식 AnomalyCLIP 원본과 checkpoint를 adapter가 load
  → `anomalyclip_mvtec` Dataset이 image/mask 전처리
  → 공통 runner가 score, map, AUROC, CSV를 생성
```

원본 protocol 그대로 prompt를 새로 학습할 때만 공식 `train.py`를 사용한다. PatchCore·SimpleNet과 같은 방식으로 비교 평가할 때는 위 framework 실행 명령을 사용한다.

## Evidence Map

| 주장 | 근거 |
|---|---|
| registry, 자체-backbone, Dataset variant 선택 | `component_registry.py`, `main_net.py` |
| 공식 공간 전처리 | `datasets/anomalyclip_mvtec.py`, 공식 `AnomalyCLIP_lib/transform.py` |
| adapter contract와 score/map 구현 | `trainer/trainer_anomalyclip.py` |
| method config | `configs/anomalyclip.yaml` |
| 공식 source 경로와 checkpoint protocol | `method6/source/run_anomalyclip_visa_to_mvtec_wsl.sh` |
| upstream 및 기존 재현 결과 | `method6/markdown/ICLR24_AnomalyCLIP_reproduction_report.md` |


