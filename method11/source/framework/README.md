# DINOMALY shared codebase

11개 image anomaly detection method를 동일한 dataset runner에서 학습하고 평가하는 공유용 코드입니다. 체크포인트와 dataset은 포함하지 않으며, 필요한 pretrained weight는 첫 실행 때 내려받습니다.

## 실행 설정

`experiment.yaml`의 네 값만 바꾸면 됩니다.

```yaml
method: simple
dataset: mvtec
category: screw
data_path: /datasets/mvtec
```

여러 category를 합칠 때는 다음처럼 적습니다.

```yaml
category:
  - bottle
  - cable
```

지원 method:

```text
coad, dinomaly, glass, padim, patchcore, promptad,
rd, rd_orig, simple, uniad, winclip
```

지원 dataset key:

```text
custom, lg, medical, multi, multi_glass,
mvtec, mvtec_glass, realiad, visa
```

`mvtec`, `medical`, `custom`, `visa`, `multi`는 MVTec 형식의 디렉터리를 사용합니다. `lg`의 `data_path`는 dataset 경로와 class mapping을 담은 JSON 파일입니다. `realiad`는 Real-IAD 이미지 root이며 metadata 위치가 다르면 `realiad_jsons_path`를 `experiment.yaml`에 추가합니다. GLASS texture 경로가 기본값과 다르면 `anomaly_source_path`를 추가합니다.

## Docker 실행

```bash
cp .env.example .env
# .env의 DINOMALY_DATA_ROOT를 실제 dataset 상위 경로로 수정
docker compose up --build -d
docker compose exec app python main.py
```

YAML을 수정하지 않고 선택값을 덮어쓸 수도 있습니다.

```bash
docker compose exec app python main.py \
  --method patchcore \
  --dataset mvtec \
  --category bottle \
  --data-path /datasets/mvtec
```

method별 기본 hyperparameter는 `configs/<method>.yaml`에 있습니다. 공통 override도 `experiment.yaml`에 추가할 수 있습니다.

## Dataset layout

MVTec 형식의 최소 구조는 다음과 같습니다.

```text
<data_path>/<category>/
├── train/good/*.png
├── test/good/*.png
├── test/<defect>/*.png
└── ground_truth/<defect>/*_mask.png
```

통합 코드는 `LICENSE`를 따릅니다. vendored method, pretrained checkpoint, dataset은 각 원 저작물의 라이선스와 배포 조건을 별도로 확인해야 합니다.
