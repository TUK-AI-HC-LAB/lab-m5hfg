# Method26 — QueCo 공통 프레임워크

제공된 QueCo 추가 ZIP을 이 method의 기준 소스로 보관한다. 기존 method9의 QueCo 결과·실행 자료는 이곳으로 이동했다. 다른 모델의 method9 자료는 유지한다.

## 현재 소스와 실제 결과

- [framework](source/framework/): 제공된 ZIP에서 추출한 공통 프레임워크. Python 코드와 YAML은 수정하지 않았다. __pycache__·pyc는 제외했다.
- [QueCo Trainer](source/framework/trainer/trainer_ourzad.py), [모델 라이브러리](source/framework/trainer/QueCo_lib/), [설정](source/framework/configs/queco.yaml), [공통 experiment](source/framework/experiment.yaml).
- [최근 screw CSV](source/results/queco_screw_20261011/results_queco.csv), [실행 기록](source/results/queco_screw_20261011/execution.json).
- [ZIP에 포함된 같은 실행 CSV](source/framework/results/queco_check_20261011_01/queco__layer24_/results_queco.csv), [실행 로그](source/framework/results/queco_check_20261011_01/run.log).
- [재실행 스크립트](source/run_queco_screw_w42.sh), [W42 실행·포팅 기록](../meetings/2026-W42_brief.md).

최근 결과는 2026-10-11 MVTec AD screw 전체 test 160장(정상 41장·결함 119장), batch=1, runner seed=0, Trainer 내부 seed=42의 단일 실행이다. 추가 학습은 없고 fusion diagnostics는 켜져 있다. 이미지·mask 크기는 518×518이다.

| 평가 단위 | AUROC (%) | AP (%) | F1 (%) | AUPRO (%) |
|---|---:|---:|---:|---:|
| Image | 89.32 | 95.33 | 93.78 | — |
| Pixel | 98.54 | 37.63 | 40.19 | 94.05 |

원시 CSV는 0–1 단위이고 표는 %로 표시했다. 15-category 평균이나 여러 seed 평균이 아니다. auroc_std=0은 이 Trainer에 기록된 평가값이 한 개이기 때문이다.

## 재실행

기존 일반 experiment.yaml을 사용하며 별도 QueCo experiment 파일은 없다. 데이터셋과 CLIP·DINOv2 가중치는 repository에 포함하지 않는다.

```bash
bash source/run_queco_screw_w42.sh source/framework <MVTec경로> <Python실행파일>
```

이 스크립트는 실행 후 정리한 재실행용이며, 자료 이동 작업 중 추론을 새로 실행하지 않았다. Python 환경은 framework/requirements.txt와 모델 import 의존성을 충족해야 한다.

## 이전 실행과 구분

- [legacy framework snapshot](source/legacy_framework_snapshot/): 이전 공통 프레임워크 작업본과 과거 QueCo 결과. 현재 ZIP 소스와 동일하지 않다.
- [과거 공통 15-category CSV](source/legacy_framework_snapshot/results/queco_all_categories_20261010_glf3FQ/queco__layer24_/results_queco.csv): 추가 CLIP 정규화 변환이 있던 2026-10-10 실행 결과. 현재 코드 결과로 해석하지 않는다.
- [원본 runner 기록](source/original_runner/): 당시 제공된 원본 runner의 실행 자료. 누락된 공통 의존성을 기존 경로에서 연결한 실행이며, 공개 공식 commit 동일성은 확인되지 않았다.

## ZIP 출처 확인

입력 파일 식별명: dinomaly_share_codebase(add QueCo).zip.

SHA-256: 9CAEDA80444C5D9A37EFFF4837AD87E3E53A3214DD16B0EFA4ED55A7447C9036.

ZIP 내부의 screw CSV와 별도 보관 CSV의 SHA-256을 대조했다. 현재 외부 작업본의 main.py·registry·기본 Trainer·QueCo Trainer·queco.yaml도 ZIP 추출본과 동일하다. 단, 실행 당시 전체 소스 해시가 기록되지 않아 소급한 완전 재현성 검증을 주장하지 않는다.
