# PatchCore 재현 실행 준비

## 구현체

- upstream: https://github.com/amazon-science/patchcore-inspection
- revision: `fcaa92f124fb1ad74a7acf56726decd4b27cbcad`
- local source: `method1/source/patchcore-inspection/`

공식 구현체를 새로 받은 뒤 Windows에서 같은 결과를 재현하려면 `method1/source/patchcore_windows_compat.patch`를 적용함. 이 patch는 FAISS CPU 검색의 peak memory를 낮추고, Windows에서 히트맵 파일을 저장할 수 있게 함.

## 확인 완료

- Python 3.10.11
- GPU: NVIDIA GeForce RTX 5080 (16GB)
- PyTorch 2.11.0+cu128에서 CUDA 인식 확인
- FAISS CPU 1.14.3 설치 및 PatchCore CLI 도움말 실행 확인

## 재현 대상

- MVTec AD 15개 category
- 공식 baseline detection 설정: WideResNet-50, `layer2`/`layer3`, 입력 224, coreset 10%, patch size 3, nearest neighbor 1, seed 0
- 공식 예시의 기대 평균: image AUROC 0.992, pixel AUROC 0.981, PRO 0.944

## 실행 방법

MVTec AD 원본의 `mvtec/` 폴더를 `method1/source/patchcore-inspection/data/mvtec/`에 둔 뒤 실행함.

```powershell
cd method1/source
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_patchcore_mvtec.ps1 -Category bottle
```

전체 category 실행은 아래와 같음.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_patchcore_mvtec.ps1 -Category bottle,cable,capsule,carpet,grid,hazelnut,leather,metal_nut,pill,screw,tile,toothbrush,transistor,wood,zipper
```

## 현재 제한

- Windows용 공식 `faiss-gpu` 배포본이 없어, 특징 추출은 GPU에서 수행하고 FAISS 최근접 탐색만 CPU에서 수행함. metric 결과에는 영향을 주지 않지만 실행 시간은 더 길어질 수 있음.
