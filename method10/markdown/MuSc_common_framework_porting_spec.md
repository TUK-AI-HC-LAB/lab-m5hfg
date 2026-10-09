# MuSc 공통 이상 탐지 framework 포팅: 파일별 코드 명세

작성·수정일: 2026-10-05. 공통 framework의 목적에 맞춰 전용 Dataset·공식 metric 중심의 기존 명세를 수정했다. 아래 코드 예시는 제안이며 아직 framework에 적용하거나 실행하지 않았다.

## 1. 결론과 대상 코드

추가 범위는 **`trainer/trainer_musc.py` + `configs/musc.yaml` + METHOD_REGISTRY 등록**이다. MuSc feature 추출과 LNAMD·MSM·RsCIN은 `trainer_musc.py` 안의 class/helper로 구현하고 별도 package나 backend 파일을 만들지 않는다. 데이터 목록·split·이미지/마스크 전처리·평가·집계·결과 CSV 저장은 공통 framework를 사용한다. 기존 runner가 전달하는 category별 전체 test DataLoader를 method 내부에서 처리한다.

기본 실행 흐름은 **공통 DataLoader → MuSc feature/LNAMD/MSM/RsCIN → 공통 평가 → 공통 결과 저장**이다. adapter는 방법 고유 계산과 입출력 연결만 담당한다. 공식 전처리·7개 metric은 별도 재현 진단이며 기본 공통 비교표를 대신하지 않는다.

실제 framework root: `C:/Users/test/Downloads/dinomaly_share_codebase/dinomaly_share_codebase` (WSL: `/mnt/c/Users/test/Downloads/dinomaly_share_codebase/dinomaly_share_codebase`). `method10`은 연구 기록·재현 artifact 위치이며 framework의 method 등록 위치가 아니다.

코드 근거: `component_registry.py`의 MethodSpec / DATASET_METHOD_VARIANTS / TRAINER_METHODS, `main_net.py`의 `net()`, `main_dataset.py`의 `dataset()`, `main_run.py`의 `train_and_collect_metrics()`, `trainer/trainer.py`의 `_predict_dataloader()`와 `_evaluate()`, `trainer/trainer_anomalyclip.py`의 독립 adapter contract.

## 2. 추가·수정할 파일

모든 경로는 위 framework root 기준이다.

| 파일 | 작업 | 필요한 코드 |
|---|---|---|
| `trainer/trainer_musc.py` | 신규 | `Trainer_MuSc`: model 준비, 전체 pool 예측, 기존 Trainer 평가·집계에 연결 |
| `configs/musc.yaml` | 신규 | 모델·입력 크기·stage·IA·RsCIN·pool 설정. 전처리 종류와 평가 구현은 공통 경로 유지 |
| `component_registry.py` | 수정 | METHOD_REGISTRY에 `musc` 등록. MuSc 전용 Dataset 등록 없음 |

`main.py`는 registry에서 CLI method 목록과 method YAML 이름을 읽는다. `main_net.py`에는 `uses_backbone=False` 분기가 있다. `main_dataset.py`의 기존 MVTec loader와 `main_run.py`의 CSV 저장을 사용한다. `datasets/base.py`, `datasets/mvtec.py`, DATASET_REGISTRY, DATASET_METHOD_VARIANTS와 공통 전처리·후처리·평가 코드는 수정하지 않는다. 공통 sample에 path가 없으면 test loader의 dataset metadata와 순서로 매핑한다. 순서를 확인할 수 없는 입력은 method에서 오류를 내고 기존 Dataset 계약을 임의로 바꾸지 않는다.

별도 MuSc library 폴더·Dataset·experiment·metric 모듈·전용 실행 script는 추가하지 않는다. 기존 실행 script나 runner에서 method 선택값만 `musc`로 지정한다. 구현에 사용한 공식 코드의 출처·commit과 필요한 license/attribution은 method 파일 주석과 기존 연구 문서에 기록한다.

## 3. Registry 코드

`component_registry.py`의 METHOD_REGISTRY에 아래 항목만 추가한다.

```python
# METHOD_REGISTRY
"musc": MethodSpec(
    "trainer.trainer_musc",
    "Trainer_MuSc",
    "musc",
    uses_backbone=False,
),

```

자체 model 생성은 `uses_backbone=False`로 연결한다. MuSc는 AnomalyCLIP 학습 prompt checkpoint가 아니라 OpenAI pretrained CLIP weight를 사용한다.

## 4. Adapter의 함수별 역할

기존 `Trainer`를 상속해 `_evaluate()`, `record_evaluation_epoch()`, `get_evaluation_metrics()`의 공통 경로를 재사용한다. 모델 준비와 전체 pool 예측은 override하고, 기존 batch별 `_predict_dataloader()`나 학습 loop로 들어가지 않도록 한다. 공통 계약은 다음 5개 함수다.

```python
from trainer.trainer import Trainer

class Trainer_MuSc(Trainer):
    def __init__(self, device):
        super().__init__(device)
        self.evaluation_results = []
    def load(self, backbone, layers_to_extract_from, device,
             input_shape, args, **kwargs): ...
    def set_model_dir(self, model_dir, dataset_name): ...
    def train(self, training_data, validation_data,
              test_data, dataset_name):
        self.evaluation_results = []
        self.record_evaluation_epoch(test_data)
    def predict(self, data): ...
    # get_evaluation_metrics()는 기존 Trainer 구현 사용
```

위 코드는 함수 signature 명세이며 완성된 구현은 아니다.

| 함수 | 처리 |
|---|---|
| `load()` | `args` 저장, MuSc CLIP 생성, stage·해상도·precision 검증. 공통 입력 tensor를 사용하고 별도 공식 preprocess로 재생성하지 않음 |
| `set_model_dir()` | 공통 runner가 전달한 저장 경로를 사용. 보조 raw output도 그 경로 아래에 저장 |
| `train()` | train/validation을 scoring에 사용하지 않고 공통 `record_evaluation_epoch(test_data)` 한 번 실행. optimizer/epoch 학습 없음 |
| `predict()` | 전체 DataLoader의 이미지·path를 모아 pool scoring. 정답은 별도 보관하고 metric 단계에서만 사용 |
| `get_evaluation_metrics()` | 기존 Trainer의 key와 집계 형식을 사용해 공통 runner에 반환 |

같은 `trainer_musc.py` 내부에 `_extract_pool_features()`, `_aggregate_neighborhood()`, `_mutual_scoring()`, `_rescore_images()` 같은 helper를 둔다. scoring helper는 image feature만 받고 label/mask는 받지 않는다. 설치된 CLIP 구현의 모델 로딩을 재사용하고, intermediate patch token 수집·해상도별 position embedding 처리가 필요하면 method 내부에서 구현한다. 기존 WinCLIP 코드나 공통 backbone loader를 수정하지 않는다.

`predict()` 반환 형식은 `(scores, maps, features, labels_gt, masks_gt)`다. scores는 `[N]`, maps는 공통 계약에 맞춘 `[N,H,W]` 등 고정 shape, features는 필요 없으면 `None`, label/mask는 이미지 순서에 맞춘다. 공식 raw map `[N,1,H,W]`와 공통 map 변환을 명시한다.

## 5. 전체 pool을 처리하는 코드 흐름

```python
# 설명용 의사 코드: 공식 함수에 실제 입력 shape를 맞추는 구현 필요
patch_features, class_features, paths = extract_all_batches(data)
patch_scores = mutual_score_pool(
    patch_features,
    stages=[6, 12, 18, 24],
    aggregation=[1, 3, 5],
    interval=(0.0, 0.3),
)
maps = upsample_musc(patch_scores, size=args.masksize)
scores_before = maps.reshape(len(paths), -1).max(-1)
scores = rescore_official(scores_before, class_features, k_list=[1, 2, 3])
```

- batch size는 feature 추출 단위이고 scoring reference 수가 아니다. batch 4장 안에서만 MSM을 수행하면 원래 MuSc와 다른 protocol이다.
- 자기 이미지 제외, LNAMD 후 feature normalization, stage/degree score 평균, RsCIN score 정규화·자기 이웃 포함 방식을 보존한다. MuSc patch map 복원의 내부 보간은 명시하고 출력 shape를 공통 masksize에 맞춘다. 평가 단계에서 공통 map 정규화를 한 번 사용하며 MuSc 전용 Gaussian smoothing이나 heatmap 정규화를 추가하지 않는다.
- category별 pool을 유지한다. framework의 `bottle+cable`처럼 category를 섞는 입력은 첫 포팅에서 명시적으로 거부한다.
- 단일 이미지 tensor `predict(image)`는 reference pool 정의 없이는 지원하지 않는다. 불완전한 pool·subtest도 거부하고 label-dependent truncation을 사용하지 않는다.
- 결과는 공통 loader의 sample 순서와 대응시킨다. path는 공통 sample 또는 dataset metadata에서 얻는다. 독립 loader와 공통 loader의 순서가 다를 수 있으므로 진단 시 배열 index만으로 비교하지 않는다.
- 풀 캐시는 ordered path 목록, config, model weight와 code version을 key로 삼는다. DataLoader 객체만 기준으로 캐시를 재사용하면 안 된다.

## 6. 공통 전처리와 모델 연결

기본 입력은 기존 공통 Dataset이 만든 tensor다. 현재 공통 ImageNet normalization, resize, mask 보간·이진화와 split을 유지하고 MuSc 전용 crop/resize/Dataset을 추가하지 않는다. 현재 WinCLIP·AnomalyCLIP 공통 입력 정책과 맞추기 위해 adapter에서 ImageNet→CLIP 재정규화도 기본으로 적용하지 않는다.

CLIP에 필요한 feature L2 normalization은 알고리즘 내부 연산으로 유지한다. 이는 입력 이미지의 mean/std normalization과 다르다. 공식 CLIP 입력과 공통 입력의 차이는 실험 조건으로 기록하며, 독립 실행과 정확히 같은 점수가 나온다고 전제하지 않는다. 정규화 정책을 개선하려면 향후 모든 관련 CLIP method에 공통으로 적용할 별도 protocol 변경으로 검증한다.

method는 이미지 경로로 공식 Dataset을 다시 만들지 않고 전달받은 공통 pool만 scoring한다. mask·label은 metric용으로 따로 보관하고 feature·MSM·RsCIN으로 전달하지 않는다. 같은 masksize에서 score/map을 반환한 뒤 기존 공통 평가·저장 코드에 연결한다.

공식 MuSc의 실행 class 전체를 `sys.path.insert()`로 import하지 않는다. 공식 repository와 framework에 `datasets`, `utils`, `models` 이름이 겹치기 때문이다. 필요한 알고리즘 연산을 새 method 파일에 구현하고 기존 설치 라이브러리의 명확한 package 경로를 사용한다. 일반 CLIP 로더의 기본 출력만으로 MuSc의 stage별 patch token을 얻었다고 가정하지 않는다. 공식 feature와 비교해 token 선택, normalization, position embedding 처리 및 precision을 검증한다. 이 명세는 기존 설치 라이브러리로의 연결이 이미 검증됐다는 뜻은 아니다.

## 7. Config 제안

`configs/musc.yaml`에 추가할 설정 예시다. `musc_` option은 새 adapter가 읽어 구현할 항목이며 아직 parser/API에 존재하는 옵션이라고 해석하면 안 된다.

```yaml
resize: 518
imagesize: 518
masksize: 518
batch_size: 4
meta_epochs: 1
train_backbone: false
subtest: false
backbone_names: [musc]
layers_to_extract_from: [layer6, layer12, layer18, layer24]
musc_backbone_name: ViT-L-14-336
musc_pretrained: openai
musc_feature_layers: [5, 11, 17, 23]
musc_r_list: [1, 3, 5]
musc_ia_min: 0.0
musc_ia_max: 0.3
musc_rscin_k_list: [1, 2, 3]
musc_divide_num: 1
musc_tf32: false
musc_precision: official_amp
musc_input_policy: common
musc_metric_policy: common
```

`backbone_names`와 `layers_to_extract_from`은 factory 계약을 위한 값이고, 실제 feature 추출은 MuSc 전용 설정을 사용한다. `meta_epochs: 1`은 기존 runner의 설정 호환용이며 MuSc를 1 epoch 학습한다는 뜻이 아니다.

518은 이번 MuSc method config의 입력 크기다. 같은 해상도를 모든 method에 강제하는 설정은 아니다. 엄격한 동일 해상도 비교를 요청하면 experiment 설정으로 각 method에 공통 적용하고 별도 실행으로 남긴다. CLIP의 position embedding과 patch grid가 지원하는지 load 단계에서 확인한다.

현재 config 우선순위는 defaults → method YAML → experiment YAML → CLI다. 기존 `experiment.yaml`에서 `method: musc`를 선택하거나 기존 CLI의 `--method musc`를 사용한다. 데이터 경로·category·seed·저장 경로도 기존 설정과 CLI를 사용하고, MuSc 내부 설정만 `configs/musc.yaml`에서 읽는다. method YAML의 입력 크기 등이 experiment 또는 CLI에서 덮어써질 수 있으므로 최종 적용 설정을 실행 결과에 기록한다. 별도의 MuSc 전용 experiment 파일이나 CLI option은 추가하지 않는다.

## 8. 평가와 저장

기존 `Trainer.record_evaluation_epoch()` → `_evaluate()` → `get_evaluation_metrics()`를 사용한다. 표준 key는 `auroc_*`, `pixel_auroc_*`, `sal_f1_*`, `pro_*`의 min/max/mean/std다. 현재 공통 PRO는 미구현 sentinel -1이며 그대로 미측정으로 표시한다. MuSc만 공식 AUPRO를 계산해 `pro_mean`에 채우지 않는다.

공통 `_evaluate()`의 map 정규화·mask ceil·saliency F1 정의를 재사용한다. **saliency F1은 논문의 Pixel F1-max와 같은 지표가 아니다.** 평가 수식을 adapter에 복제하지 않는다. AP/F1-max/AUPRO가 공통 비교에 필요해지면 모든 method가 사용할 공통 평가 확장으로 추가하고 재평가한다.

공식 7개 지표는 기존 `method10/source`의 독립 재현·진단 경로에 남긴다. 이번 method 추가에서는 새 공식 metric 모듈이나 별도 평가 CSV 경로를 만들지 않는다. 기존 독립 결과를 참고하되 기본 framework 결과 CSV·평균에는 섞지 않는다.

기본 결과 저장은 기존 framework의 경로와 CSV 형식을 사용한다. 연결 검증에 필요한 scores_before/after_RsCIN, raw map, sample 순서, 적용 config, code/model 출처는 기존 저장 위치에 보조 근거로 남기며 별도 저장 모듈을 만들지 않는다. weight·dataset·대용량 NPZ는 repo 외부의 원본을 참조한다.

## 9. 연결 검증 순서와 기존 evidence

1. registry와 기존 method YAML 로더로 adapter를 생성하고 load 인자 계약을 확인한다.
2. 공통 bottle loader의 83장과 label/mask/split을 확인하고 train/val이 scoring에 들어가지 않는지 검사한다.
3. **동일한 공통 입력 tensor**에 대해 `trainer_musc.py` 내부의 계산 helper와 `predict()` 연결 결과를 path별로 비교한다. feature/stage/MSM/RsCIN 연산은 기존 독립 재현 구현과 고정 feature 입력으로 대조하고, 원시 이미지 전처리 차이와 분리해 검증한다.
4. 공통 `_evaluate()`로 평가해 기존 runner의 CSV/key/집계와 연결되는지 확인한다. 공통 평가의 정상 작동이 포팅 완료 조건이다.
5. 기존 공식 입력의 독립 baseline과 비교는 입력·mask·metric 정책 차이를 설명하는 보조 진단으로 수행한다. 공식 숫자와의 일치를 포팅 성공 조건으로 두지 않는다.
6. MVTec 15 category를 공통 입력·평가로 실행하고 category별 및 macro 평균을 저장한다. 기존 독립 결과와 다른 run 이름·경로로 보존한다.
7. 기존 method의 registry/lazy import/입력·평가 정책이 유지되는지 대표 실행으로 확인한다.

기존 독립 재현 근거: [독립15 category 결과](MuSc_paper_vs_local_mvtec_all.md), [category_metrics.csv](../source/results/mvtec_all_paper_20261002/category_metrics.csv), [bottle 이미지별 score](../source/results/bottle_paper_20261002/image_scores.csv). 이 결과는 공식 입력·공식 평가 baseline이며 새 공통 평가 결과와 한 표의 동일 조건 결과로 합치지 않는다. 차이를 분석할 때 입력 정책과 metric 정책을 각각 고정해 원인을 분리한다.

MuSc는 test pool 자체를 사용하는 transductive 방법이다. 같은 공통 평가를 사용해도 다른 method와 정보 사용 조건이 완전히 같아지는 것은 아니므로 최종 표에 `unlabeled category test pool 사용`을 명시한다.

이번 명세의 범위는 MVTec 연결이다. VisA는 현재 registry가 MVTec-style loader를 사용하므로 공식 VisA split/layout과 일치하는지 별도로 확인한 뒤 대응한다.
