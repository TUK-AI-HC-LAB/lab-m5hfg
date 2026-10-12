# WinCLIP MVTec·VisA 속도·메모리 자체 측정

질문: 기존 0/4-shot 설정에서 query 처리 시간과 GPU 사용량은 얼마인가. 측정 전 가설: 4-shot은 support 비교와 메모리 때문에 0-shot보다 비용이 증가한다.

- commit: `eef4f8cce6a80eddfa07415b503c22c6c3427351` (공개 비공식 재현 구현)
- sh: `method7/source/run_winclip_benchmark.sh`
- result: `method7/source/results/benchmark_20261011/summary.csv`

| Dataset | Shot | Batch | ms/image | FPS | Peak allocated MiB | Peak reserved MiB |
|---|---:|---:|---:|---:|---:|---:|
| mvtec | 0 | 1 | 7669.4076 | 0.13 | 1140.03 | 1360.00 |
| mvtec | 0 | 32 | 270.6806 | 3.69 | 3280.01 | 4356.00 |
| mvtec | 4 | 1 | 15313.6718 | 0.07 | 1139.97 | 1484.00 |
| mvtec | 4 | 32 | 511.2956 | 1.96 | 3374.92 | 4452.00 |
| visa | 0 | 1 | 7692.2555 | 0.13 | 1139.96 | 1360.00 |
| visa | 0 | 32 | 271.5677 | 3.68 | 3281.09 | 4356.00 |
| visa | 4 | 1 | 15343.6908 | 0.07 | 1139.96 | 1460.00 |
| visa | 4 | 32 | 512.2499 | 1.95 | 3356.18 | 4430.00 |

## 측정 조건과 해석 범위

MVTec 15개·VisA 12개 모든 category 포함. 각 category의 순서상 첫 3장을 batch1로 각각 warm-up 1회 후 3회 측정한다. 별도로 첫32장의 batch32를 warm-up 1회 후 3회 측정한다. 고정 입력 반복 benchmark이며 전체 test 정확도 재평가가 아니다. 배치별 시간은 GPU 동기화한 wall time을 총 이미지 수로 나눈다. 입력 로딩은 사전 수행하고 H2D·특징 추출·native 점수 계산·map interpolation을 포함한다. setup/CPU output copy/metric/파일 저장은 제외한다. 모델·text·support 초기화 시간은 initialization.csv에 별도 저장했다.

실제 모델 ViT-B-16-plus-240 / laion400m_e31 / 240px / seed10 / native FP32. upstream query AST를 그대로 추출해 호출하며 AMP·torch.compile·알고리즘 가속을 새로 적용하지 않았다. 동일 입력 반복 출력의 allclose와 유한값, map 크기 및 raw output SHA를 검증했다. 기존 결과의 historical support manifest가 없어 당시 선택 이미지의 동일성은 인증할 수 없다. 기존 정확도 표는 재측정 없이 유지한다.

GPU 메모리는 bytes/2^20의 MiB이며, setup 및 inference 중 max PyTorch allocated/reserved이다. 단위가 불명확한 논문 MB 및 nvidia-smi 전체 VRAM과 동일하지 않다. 전체 dataset의 map을 GPU에 누적 보관하지 않고 query마다 해제해 inference working memory를 측정했다. CUDA context/외부 allocator/Windows graphics 사용량은 이 값에 포함되지 않는다. 별도의 competing compute process는 측정 전후 확인했으며 graphics 부하는 독립 검증하지 않았다.

MuSc Table13에는 WinCLIP 0-shot/MVTec 행이 있으나 batch/warm-up/timing 경계/memory API를 명시하지 않는다. 따라서 자체 정의된 batch1 값을 해당 행에 연결하고 batch32 및 추가 dataset/shot은 이 보고서에 구분한다. 다른 방법 benchmark와 구현·측정 범위가 달라 속도 순위를 단정할 수 없다. 정확도나 논문 runtime 수치를 이 측정으로 검증한 것은 아니다.

검증: verification.json / category_verification.json. Raw 반복 시간: timing_samples.csv. 입력·support SHA: input_manifest.csv. 환경 및 package: environment.json / runtime_packages.txt / nvidia_smi.txt. 원본 코드 SHA 및 추출 AST: source_hashes.json / native_query_body.py.txt. 원시 map은 `/home/test/winclip_benchmark_20261011`에 보존한다.
