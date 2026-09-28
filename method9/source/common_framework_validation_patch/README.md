# 공통 framework 정상 validation 분할 코드

원본 framework는 `C:\Users\test\Downloads\dinomaly_share_codebase\dinomaly_share_codebase`에 있으며 Git 저장소가 아니다. 따라서 GitHub에서 W40 근거를 바로 열 수 있도록, 이 폴더에 **실제로 사용한 코드·설정·실행 script·작은 raw 결과**를 원래의 상대 경로 구조로 보관한다.

- `validation_split.patch`: 원본 framework에 적용한 Python 코드 변경이다.
- `anomalyclip_common_input.patch`: AnomalyCLIP도 WinCLIP과 같이 공통 ImageNet 정규화 입력을 그대로 받게 한 변경이다.
- `run_all_methods_bottle_validation_wsl.sh`: `bottle`, seed 0에서 12개 등록 method를 순차 실행한 스크립트다.
- `framework_snapshot/`: W40 문서에서 설명하거나 실행에 사용한 framework 코드·YAML·shell script의 Git 추적용 snapshot이다. `main.py`, registry, Dataset, 각 method Trainer, GLASS Dataset, AnomalyCLIP 설정을 원래 폴더 구조대로 포함한다. patch를 읽지 않아도 수정 후 파일을 바로 확인할 수 있다.
- `results/`: W40에서 인용한 작은 raw CSV/TSV 결과 사본이다. `all_methods_initial_20260926/`, `all_methods_validation_20260928/`, `all_methods_nearest_mask_20260928/` 및 AnomalyCLIP·PatchCore smoke test 결과를 포함한다. 로그·dataset·checkpoint는 포함하지 않는다.

데이터셋, checkpoint, DTD texture, raw log는 용량 및 재배포 문제로 포함하지 않는다. 이들은 GitHub에서 열 수 있는 파일이 아니며, 실행 script의 환경 변수/경로로 별도 준비해야 한다. 실행 결과와 자세한 해석은 [`meetings/2026-W40_brief.md`](../../../meetings/2026-W40_brief.md) 및 [`W40 전체 실행 결과`](../../../related_work/markdown/W40_공통_프레임워크_전체_method_실행_결과.md)에 있다.

적용 대상 파일:

```text
datasets/base.py
datasets/mvtec.py
main_dataset.py
main.py
trainer/trainer_anomalyclip.py
configs/anomalyclip.yaml
run_all_methods_bottle_validation_wsl.sh
```

## GitHub에서 따라가는 순서

1. [`framework_snapshot/component_registry.py`](framework_snapshot/component_registry.py)에서 method 이름과 Trainer class 연결을 확인한다.
2. [`framework_snapshot/main.py`](framework_snapshot/main.py) → [`framework_snapshot/main_dataset.py`](framework_snapshot/main_dataset.py) → [`framework_snapshot/datasets/mvtec.py`](framework_snapshot/datasets/mvtec.py) 순서로 데이터 분할과 DataLoader 생성을 확인한다.
3. [`framework_snapshot/run_all_methods_bottle_validation_wsl.sh`](framework_snapshot/run_all_methods_bottle_validation_wsl.sh)에서 동일 조건의 12-method 실행 명령을 확인한다.
4. [`results/all_methods_validation_20260928/status.tsv`](results/all_methods_validation_20260928/status.tsv)과 각 method 하위 `results_*.csv`에서 raw 결과를 확인한다.

핵심 변경은 MVTec `train/good` 이미지를 seed 고정으로 TRAIN/VAL로 분할하고, TEST를 그대로 유지하는 것이다. `train_val_split=0.9`는 90% TRAIN·10% VAL이고, `1.0`은 이전처럼 모든 정상 train 이미지를 TRAIN에 사용한다.
