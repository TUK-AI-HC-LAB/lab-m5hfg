# 공통 프레임워크 코드 snapshot

2026-10-17에 로컬 실행 원본과 대조하여 누락된 코드·설정·토크나이저 자원을 추가함. 기존 snapshot 파일은 덮어쓰지 않았으므로, W40에 저장한 코드와 이후 원본의 보조 코드가 함께 있는 구성임. 과거 실행 환경 전체의 동일 복제나 모든 method의 실행 성공을 뜻하지 않음.

## 포함한 것

- `utils.py`, `metrics.py`, 원본 `requirements.txt`, `LICENSE`.
- Dataset package 초기화 파일과 COAD·Dinomaly·GLASS·WinCLIP 등의 누락된 내부 Python 코드.
- 모델 구조 JSON·YAML과 CLIP 토크나이저 BPE 어휘. BPE `.txt.gz`는 tokenizer가 읽는 어휘 자원이며 모델 가중치나 데이터셋 압축본이 아님.
- `external/AnomalyCLIP/`: 공식 AnomalyCLIP adapter가 참조하는 Python 소스와 tokenizer 자원. 복사 원본 commit: `3911738c0867544f545a076ad78f3f11d9ecbfdf`. 자체 `LICENSE`와 `requirements.txt`를 포함함.

## 제외한 것

데이터셋, DTD 이미지, 모델 가중치·checkpoint, 가상환경, cache, 실행 로그, 원본 results 폴더는 이번 복사에 포함하지 않음. 기존 snapshot에 있던 파일은 삭제하지 않음. 기본 설정에 남아 있는 데이터·checkpoint 경로는 현재 환경에 맞게 지정해야 함.

## 실행 경로

이 폴더를 작업 디렉터리로 사용함. Python, PyTorch/torchvision 및 선택한 method의 외부 패키지를 먼저 별도 환경에 설치해야 함. 원본 requirements는 버전 고정 파일이 아니며 CUDA 환경 복원이나 설치 성공을 보장하지 않음. `requirements.snapshot.txt`는 원본 목록 외의 명시적 의존성을 추가한 보조 목록임. 이번 작업에서는 환경 설치나 실험 재실행을 하지 않음.

```text
python main.py --method patchcore --dataset mvtec --category bottle --data-path /actual/path/to/mvtec --results-path /actual/path/to/results
```

위 명령은 실행 형식 예시임. 기존 WSL 실행 shell script는 과거 원본 경로를 가리키므로 이 폴더 실행용으로 자동 변경하지 않았음.

AnomalyCLIP은 기존 `configs/anomalyclip.yaml`의 `/home/test/anomalyclip` 경로를 보존함. snapshot 안에 포함된 공식 소스를 사용하려면 이 폴더에서 별도 설정을 선택함:

```text
python main.py --config experiment_anomalyclip_snapshot.yaml --method anomalyclip --dataset mvtec --category bottle --data-path /actual/path/to/mvtec --results-path /actual/path/to/results
```

학습된 prompt checkpoint와 CLIP pretrained weights는 별도임. 로컬 checkpoint 경로를 쓰려면 별도 설정의 `anomalyclip_checkpoint_path`를 지정해야 함. 소스 포함만으로 이전 학습 결과를 재현하거나 다운로드 없이 실행할 수 있는 것은 아님.

## 확인한 범위

상위 폴더의 `verify_snapshot_dependencies.py`로 Python AST 문법 검사와 원본에 존재하는 로컬 정적 import 대상의 누락을 점검함. 패키지를 실제 import하거나 모델을 실행하는 검사는 아님. 동적 import, 외부 라이브러리 호환성, 데이터·checkpoint는 별도 검증 대상임.

이번 점검 결과: Python 180개 문법 검사 통과, 원본 대비 정적 로컬 import 누락 0개, registry에 등록된 module 경로 22개 존재 확인, BPE 어휘 파일 4개 읽기 확인. 현재 Windows Python에서 `main.py --help`를 실제 시도했지만 외부 패키지 `loguru` 미설치로 중단됨. 같은 환경에 PyTorch/torchvision도 없어 실행 준비 완료라고 판정하지 않음. 별도 환경 설치는 이번 요청에 포함하지 않음.

```text
python ../verify_snapshot_dependencies.py
```

복사한 파일별 SHA256과 검증 결과는 상위 폴더의 `snapshot_completion_20261017.json`에 기록함. 로컬 보완 후 별도 사용자 요청에 따라 GitHub `master`에도 업로드함.

GitHub 업로드 보완: 로컬에는 이미 있었으나 `datasets/` 제외 규칙으로 빠져 있던 `datasets/anomalyclip_mvtec.py`, `datasets/explicit_realiad.py`, `datasets/lgdata.py`도 코드 파일로 명시적으로 포함함. 이미지 데이터 폴더는 업로드하지 않음.
