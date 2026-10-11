# QueCo 공통 framework 연결

현재 snapshot의 registry와 Trainer lifecycle을 통해 QueCo를 실행한다.
[main.py](main.py)는 기존 파일을 그대로 사용한다.

## 연결한 파일

- [trainer/trainer_ourzad.py](trainer/trainer_ourzad.py): 공통 DataLoader를 받아 전체 test bank를 만들고 추론한다.
- [QueCo_lib/](QueCo_lib/): 제공된 QueCo 원본의 모델·prompt·patch matching·fusion·metric 구현과 tokenizer 사전.
- [component_registry.py](component_registry.py): `queco → Trainer_OurZAD` 등록. 자체 CLIP/DINO를 사용하므로 `uses_backbone=False`.
- [configs/queco.yaml](configs/queco.yaml): 모델과 추론 설정.
- [experiment.yaml](experiment.yaml): 기존 공통 실험 설정 사용. 별도 QueCo experiment 파일은 삭제했다.
- [requirements_queco.txt](requirements_queco.txt): QueCo 추가 패키지 목록.

QueCo는 target train/validation 이미지로 파라미터를 학습하지 않는다.
공통 runner가 만든 train/validation loader는 전달만 받고, 같은 category의 test 이미지 전체를 비교한다.
자기 자신은 참조 후보에서 제외한다. 정답 label/mask는 최종 metric 계산에 사용한다.

## 준비와 실행

기존 GPU framework 환경에서 WSL로 아래 스크립트를 실행한다.

```bash
bash prepare_queco_wsl.sh
bash run_queco_wsl.sh --category bottle --data-path /home/test/data/mvtec
```

소수 이미지 동작 검사는 다음과 같다.

```bash
bash run_queco_wsl.sh --smoke
```

category와 데이터 경로는 공통 experiment.yaml에서 수정하거나 CLI로 지정한다.
run_queco_wsl.sh는 공통 YAML을 읽고 method만 queco로 선택한다.
기존 experiment.yaml의 다른 모델·category·데이터 경로는 덮어쓰지 않았다.
여러 category를 지정하면 category별로 독립적인 bank를 만들고 평가한다.

```bash
bash run_queco_wsl.sh --category bottle cable --data-path /datasets/mvtec
```

`QUECO_PYTHON`으로 Python 환경을 지정할 수 있다.
`FRAMEWORK_BASE`는 기존 공통 framework의 경로이며,
snapshot에 없는 공통 `utils` 등의 모듈을 이 경로에서 참조한다.
두 스크립트는 현재 준비된 실행 환경을 기본값으로 사용한다.

## 가중치

CLIP은 `ViT-L/14@336px`, DINO는 `dinov2_vitb14_reg`를 사용한다.
준비 스크립트가 공식 다운로드 경로로 가중치를 확보하고 일반 CLIP/torch cache에 저장한다.
데이터셋과 가중치는 Git source 폴더에 복사하지 않는다.
확보된 가중치의 크기·SHA256과 패키지 버전은
[environment.json](results/queco_setup/environment.json)에 남긴다.

## 입력과 출력

- 입력: batch 1의 `image [1,3,518,518]`, 이진 mask, image label, 이미지 경로.
- 공통 Dataset의 입력을 그대로 사용한다. 추가했던 CLIP 정규화 변환은 제거했다.
- 출력: image score와 `anomaly map [N,518,518]`.
- `train()`의 tuple: Image AUROC/AP/F1 → Pixel AUROC/AP/F1/AUPRO.
- QueCo는 다른 모델처럼 `evaluation_results`에 원시 지표를 기록한다.
  공통 `Trainer.get_evaluation_metrics()`를 상속하여 각 지표의 min/max/mean/std를 집계하고,
  runner는 이름 붙은 dictionary를 CSV로 저장한다.
  `auroc_mean`은 기존 framework용 Image AUROC 별칭이다.
  `pro_mean`은 QueCo의 AUPRO와 같고, 미계산한 `sal_f1`은 -1로 표시한다.
- F1은 원본 구현의 precision-recall curve에서 선택한 최대 F1이다.

기본값은 입력 518, 특징 layer 6/12/18/24, 참조 10장,
OT + mutual matching, entropy fusion, pose aux 비활성화이다.
batch 1과 category당 최소 test 이미지 2장이 필요하다.
진단 이미지 저장과 base segmentation helper는 기본적으로 꺼 두었다.

## 검증 근거

아래 결과는 CLIP 정규화 변환과 추가 입력 검사를 제거하기 전 측정값이다.
제거 후 전체 추론 재실행은 아직 하지 않았다.

정상 3장·결함 3장 동작 검사:
[smoke.json](results/queco_smoke/smoke.json),
[metrics.csv](results/queco_smoke/metrics.csv).
이 소수 이미지 점수는 전체 성능 결과로 해석하지 않는다.

전체 test 실행 결과는 아래 CSV에 저장된다.
[results_queco.csv](results/queco/queco__layer24_/results_queco.csv).
MVTec AD bottle 전체 test 83장(정상 20장, 결함 63장), seed 0의 결과:

| 구분 | AUROC | AP | 최대 F1 | AUPRO |
|---|---:|---:|---:|---:|
| Image | 0.990476 | 0.996882 | 0.984375 | — |
| Pixel | 0.978997 | 0.813001 | 0.750804 | 0.953700 |

표는 위 CSV의 값을 소수점 여섯 자리로 표시한 것이며, 값의 범위는 0–1이다.
이 전체 test CSV는 공통 getter 전환 전에 생성했다. 전환 후에는 6장 동작 검사에서
공통 getter 상속, 7개 지표 보존 및 CSV 저장을 확인했다.
전체 test를 다시 실행하면 같은 지표 이름에 min/max/std 열도 추가된다.
전체 category 평가와 제공된 원본 runner와의 수치 일치는 별도 확인 범위이다.
