# 공통 framework 정상 validation 분할 코드

W40에서 사용한 코드·설정·실행 script·raw 결과를 원래의 상대 경로 구조로 보관한다.

- `validation_split.patch`: 원본 framework에 적용한 Python 코드 변경이다.
- `anomalyclip_common_input.patch`: AnomalyCLIP도 WinCLIP과 같이 공통 ImageNet 정규화 입력을 그대로 받게 한 변경이다.
- `run_all_methods_bottle_validation_wsl.sh`: `bottle`, seed 0에서 12개 등록 method를 순차 실행한 스크립트다.
- `framework_snapshot/`: W40 문서에서 설명하거나 실행에 사용한 framework 코드·YAML·shell script의 Git 추적용 snapshot이다. `main.py`, registry, Dataset, 각 method Trainer, GLASS Dataset, AnomalyCLIP 설정을 원래 폴더 구조대로 포함한다. patch를 읽지 않아도 수정 후 파일을 바로 확인할 수 있다.
- `results/`: W40에서 인용한 작은 raw CSV/TSV 결과 사본이다. `all_methods_initial_20260926/`, `all_methods_validation_20260928/`, `all_methods_nearest_mask_20260928/` 및 AnomalyCLIP·PatchCore smoke test 결과를 포함한다. 로그·dataset·checkpoint는 포함하지 않는다.

데이터셋, checkpoint, DTD texture, raw log는 포함하지 않는다. 실행 결과와 해석은 [`2026-W40 brief`](../../../meetings/2026-W40_brief.md) 및 [`W40 전체 실행 결과`](../../../related_work/markdown/W40_공통_프레임워크_전체_method_실행_결과.md)에 있다.

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

핵심 변경은 MVTec `train/good` 이미지를 seed 고정으로 TRAIN/VAL로 분할하고, TEST를 그대로 유지하는 것이다. `train_val_split=0.9`는 90% TRAIN·10% VAL이고, `1.0`은 이전처럼 모든 정상 train 이미지를 TRAIN에 사용한다.
