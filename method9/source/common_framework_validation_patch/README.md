# 공통 framework 정상 validation 분할 코드

원본 framework는 `C:\Users\test\Downloads\dinomaly_share_codebase\dinomaly_share_codebase`에 있으며 Git 저장소가 아니다. 이 폴더에는 W40에서 실제로 적용·실행한 validation 관련 코드 변경만 보관한다.

- `validation_split.patch`: 원본 framework에 적용한 Python 코드 변경이다.
- `anomalyclip_common_input.patch`: AnomalyCLIP도 WinCLIP과 같이 공통 ImageNet 정규화 입력을 그대로 받게 한 변경이다.
- `run_all_methods_bottle_validation_wsl.sh`: `bottle`, seed 0에서 12개 등록 method를 순차 실행한 스크립트다.
- `framework_snapshot/`: 현재 원본 framework에서 실제로 바뀐 Python/config/script 파일을 원래 폴더 구조대로 복사한 Git 추적용 snapshot이다. patch를 읽지 않아도 수정 후 파일을 바로 확인할 수 있다.

데이터셋, checkpoint, DTD texture, raw log는 용량 및 재배포 문제로 포함하지 않는다. 실행 결과와 자세한 해석은 `meetings/2026-W40_brief.md` 및 `related_work/markdown/W40_공통_프레임워크_전체_method_실행_결과.md`에 있다.

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
