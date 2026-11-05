# Simple3D 공식 연산 유지 — 실행 오류·환경 호환 처리

## 원본과 실행 버전

- 공식 저장소: https://github.com/hustCYQ/MiniShift-Simple3D
- 공식 commit: `5951e097ed5f6c52c1e058490f8c64f9b6323501`
- 보존 원본: `C:/Users/test/Desktop/Codex/Project/Simple3D-official-clean-20261104`
- 최소 수정 실행본: `C:/Users/test/Desktop/Codex/Project/Simple3D-official-compatible-20261104`
- GPU KD-tree 수정판: `method12/source/simple3d_gpu_kdtree.py`, `simple3d_gpu_kdtree.cu`. 이번 실행본에서 사용하지 않음.
- sh: [run_official_compatible.sh](../source/run_official_compatible.sh)
- verification: [verify_official_compatible.py](../source/verify_official_compatible.py)
- result: [검증 상태·실측 결과·소스 해시](../source/results/official_compatible_20261104/verification.json)
- 수정 증거: [compatibility_changes.patch](../source/results/official_compatible_20261104/compatibility_changes.patch)

## 5개 파일에 적용한 변경

| 파일 | 오류 원인 | 실행 호환 수정 |
|---|---|---|
| main.py | 없는 data.quan import, logs 폴더 자동 생성 없음 | 미사용 import 제거, 로그 쓰기 전에 폴더 생성 |
| patchcore_runner.py | 없는 data.quan import | 미사용 import 제거 |
| data/MiniShiftAD.py | 저자 PC에 고정된 데이터 경로, 문자열 전체 test→gt 치환 | 환경 변수로 경로 지정, test 기준 상대 경로로 gt 연결 |
| data/anomalyshape.py | 저자 PC 경로, 문자열 전체 test→GT 치환 | 환경 변수로 경로 지정, 해당 category의 GT 폴더 조회 |
| feature_extractors/features.py | CPU 점수맵을 CUDA 인덱스로 조회 | 조회할 때 인덱스를 점수맵과 같은 장치로 이동 |

CPU 점수맵을 GPU로 옮기는 이전 우회와 달리, 이번에는 원본 CPU 점수맵을 유지하고 인덱스를 CPU로 옮김. 따라서 이전 우회판과 전송 비용·CPU/GPU 평균 계산 위치가 같다고 주장하지 않음. 알고리즘 변경은 아니지만 부동소수점 bitwise 동일성은 보장하지 않음.

## 유지한 연산과 설정

- 법선과 FPFH: 원본 Open3D legacy CPU 경로 유지. GPU FPFH 대체 없음.
- FPS/gather: PointNet2 native CUDA 연산 유지. NVRTC 대체 없음.
- 좌표 이웃: native KNN_CUDA 유지. CPU cKDTree·GPU KD-tree·FAISS 대체 없음.
- 보간: 공식 full-sort 보간 유지. topk 최적화 없음.
- 메모리뱅크 NN: 공식 torch.cdist/min 유지.
- MSND 40/80/80/120의 원본 중복 연결 유지.
- num_group=4096, group_size=128, max_nn=40, num_MSND=2, LFSA/MSND 활성화, coreset 5% 유지.
- 데이터 원본 해상도 유지, 입력·정답·특징·점수 재사용 없음.

## 실행 환경 범위

RTX 5080에서 작동하는 기존 native 환경(Python 3.11, Torch 2.11+CUDA12.8, Open3D 0.20) 사용함. 저자의 Python3.8/Torch1.12/CUDA11.3 환경과 동일하지 않음. KNN_CUDA는 이전에 확보한 소스 보존 포크, PointNet2는 sm_120 빌드 사용함. GPU 이름과 실제 모듈 위치는 검증 JSON에 저장함.

## 실제 검증 범위

ShapeNet ashtray0 및 MiniShift capsule 각각 정상 학습 1개로 새 뱅크를 만들고 정상 1개·결함 1개를 공식 predict 경로로 처리함. 공식 DataLoader의 batch_size=1, num_workers=1 및 원본 shape를 유지함. 입력 점 수와 출력 점 수 일치·최종 점수 유한값·전체 GT 경로 존재를 확인함. 최종 성공 여부는 위 verification.json의 status가 기준임.

이 검증은 실행 오류 해결 확인용이며 52개 전체 재현 결과나 평균 추론 성능이 아님. 워밍업 없는 소규모 실행 시간이므로 기존 전체 벤치마크와 직접 비교하지 않음.

검증 중 나타나는 `could not project vectors` 메시지는 원본 coreset 코드가 차원 투영 실패를 잡고 원래 특징으로 계속 실행하는 fallback임. 예외 종료와 구분함. 읽기 전용 NumPy 배열 변환 경고도 확인했으며, 이번 수정에서 해당 원본 FPFH 수치 경로를 바꾸지 않음.
