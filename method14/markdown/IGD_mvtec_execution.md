# IGD MVTec AD 자체 학습·평가

## 현재 실행: local batch8

현재 상태: 사용자가 나중에 진행하기로 요청해 2026-10-07 학습 프로세스를 중단했다. 마지막 저장 recovery checkpoint와 optimizer 상태를 보존하며 자동 재실행은 하지 않는다. 재개 명령은 `bash /mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method14/source/run_igd_accelerated.sh`다. 저장 시점 이후의 미저장 단계는 다시 계산하고, 기존 wrapper의 shuffle/RNG 재시작 한계는 유지한다.

사용자 요청으로 local batch를 8장(1,800 patches/step)으로 늘려 15개 category를 처음부터 다시 학습한다. global batch16, MAX_EPOCH256, sample_rate1.0 및 가속 설정은 유지한다. 기존 batch4 가중치와 기록은 그대로 보존한다. batch8 대표 단계의 loss·gradient 유효성은 앞선 배치 비교에서 통과했다. 최대 tensor 할당량은 약 13.54GB, 처리량은 8.58 images/sec였다. 이 측정은 전체 실행 시간이나 최종 정확도를 입증하지 않는다. 공식 local batch2와의 차이는 최종 비교표에 명시한다.

현재 원시 결과는 `/home/test/igd_results/mvtec_batch8_seed42_20261007`, log는 `/home/test/igd_results/batch8_20261007.log`다. [현재 설정](../source/results/mvtec_batch8_seed42_20261007/environment.json), [배치 검증](../source/results/mvtec_batch8_seed42_20261007/batch_preflight.json), [이전 batch4 실행](IGD_batch4_previous_execution.md)을 보존한다. 완료 뒤 자동 평가·검증하고 현재 결과만 비교표에 반영한다.

## 이전 실행: local batch4

2026-10-07 사용자의 요청으로 local batch를 2에서 4로 늘렸다. global batch16, MAX_EPOCH256, 전체 학습 이미지 및 평가 대상은 유지한다. batch4는 900 patches/step이다. optimizer 업데이트 수는 local 전체 약 464,512회에서 232,256회로 줄지만 step당 처리량이 늘므로 실행 시간이 절반이 되지는 않는다. batch 변경은 학습 조건 변경이며 최종 비교표에도 공식 batch2와의 차이를 명시한다.

대표 local 학습 단계에서 batch2/4/8의 처리량은 각각 9.03/8.74/8.58 images/sec였고 최대 tensor 할당량은 3.84/7.58/13.54GB였다. 모두 loss·gradient 유효성을 통과했다. 이 측정에서는 큰 batch가 더 빠르지 않았다. 사용자가 요청한 batch 확대 중 가장 빠르고 메모리 여유가 있는 batch4를 선택한다. 기존 batch2 결과와 섞지 않고 새 run tag에서 처음부터 실행한다. 기존 가속 실행 기록은 [이전 batch2 기록](IGD_batch2_accelerated_previous_execution.md)으로 보존한다.

현재 원시 결과 경로는 `/home/test/igd_results/mvtec_largebatch_seed42_20261007`, log는 `/home/test/igd_results/large_batch_20261007.log`다. [배치 비교 측정](../source/results/mvtec_largebatch_seed42_20261007/batch_preflight.json)과 [현재 환경](../source/results/mvtec_largebatch_seed42_20261007/environment.json)에 설정을 보존한다. 아래 이전 실행 관련 기록은 역사적 기록이며 현재 batch 변경 판단은 이 절을 기준으로 한다.

| Item | Content |
|---|---|
| Title | Deep One-Class Classification via Interpolated Gaussian Descriptor |
| Authors | Yuanhong Chen, Yu Tian, Guansong Pang, Gustavo Carneiro |
| Conference / Journal | AAAI |
| Year | 2022 |
| Paper link | [원문](https://arxiv.org/abs/2101.10043) |
| GitHub / Official code | [tianyu0207/IGD](https://github.com/tianyu0207/IGD) |
| Reason for investigation | MuSc Table 2 IGD의 Image AUROC·Pixel AUROC를 이 PC에서 측정 |

질문: 공식 IGD local/global 모델을 모두 학습하고 공식 다중 스케일 점수 계산으로 MVTec 15개 category의 검증 가능한 결과를 확보할 수 있는가? 실행 전 가설은 학습 알고리즘을 유지하며 경로·API 호환성 및 깨진 평가 연결을 복구하면 모든 test 이미지의 점수·맵을 얻을 수 있다는 것이다. 학습 완료와 원시 지표 재계산으로 가설의 동작 부분을 검증한다.

## 실행 조건

| 항목 | 값 |
|---|---|
| commit | `1ce995214ef8adf09f6c5d3dc01ce6042b472a34` |
| GPU | RTX 5080 |
| 학습 대상 | MVTec AD 정상 학습 이미지 전체, sample_rate=1.0 |
| 모델 | 공식 p32.ssim_module.twoin1Generator와 p256.mvtec_module.twoin1Generator256 |
| 초기 encoder | local: 저장소 제공 Encoder_KD_ckpt; global: 공식 코드의 ImageNet ResNet18 |
| 학습량 | 양쪽 MAX_EPOCH=256; 실제 iteration은 공식 `int(train_size/BATCH_SIZE*MAX_EPOCH)` |
| batch | local 8장×225 patches; global 16장 (사용자 요청으로 local 공식 batch2에서 변경) |
| seed | 42, 이 실행의 재현성 선택; 공식 job에는 seed 지정 없음 |
| optimizer | 공식 Adam 및 polynomial LR scheduling, 원래 loss/validation 주기 유지 |
| 정밀도 | BF16 model autocast, FP32 loss·Gaussian 통계, TF32 활성 |
| 입력 | Resize256, CenterCrop256, RGB ImageNet mean/std |
| 최종 모델 | 마지막 iteration; test AUROC로 checkpoint 선택하지 않음 |
| 이미지 점수 | 공식 inference_det의 local/global 0.5 혼합, 32 patch stride16 |
| 분할 맵 | 공식 residual MS-SSIM + 절댓값 L1, local fold·RGB grayscale, global grayscale, 0.5 혼합 |

이번 수정은 원본 저장소 밖 `/home/test/igd_results/mvtec_full_seed42_20261007/runtime`에 기록한다. 데이터 경로를 현재 원본으로 연결하고, `train_data.next()`를 `next(train_data)`로 바꾼다. 라이브러리의 MS-SSIM 입력 크기 검사만 실제 scale 수(32 patch는 4)에 맞게 보완하며 loss 계산식은 변경하지 않는다. 현대 Adam에 필요한 beta의 int 0은 float 0.0으로 전달한다. 공식 validation은 그대로 수행하되 ETA·tqdm 표시와 TensorBoard 기록은 생략한다. validation 후 eval mode를 유지하는 원래 코드 동작도 임의로 바꾸지 않는다.

공식 inference_loc는 외부 pretrain 모듈, CUDA1, 존재하지 않는 checkpoint 경로와 맞지 않는 DualDataLoader 호출을 포함해 원형 그대로 실행할 수 없다. 이 실행은 공식 job이 학습한 모델을 연결하고, 공식 map 조립 함수와 residual 모듈의 실제 연산식을 사용한 별도 평가 wrapper로 복구한다. 따라서 공식 평가 entrypoint를 무수정 실행했다고 표현하지 않는다. 평가에서는 정상 이미지를 포함한 1,725장 모두 저장한다.

MuSc Table 2에 필요한 값은 Image AUROC와 Pixel AUROC다. Pixel AUROC는 15개 category 각각의 전체 test 픽셀을 합쳐 계산한다. 원래 IGD localization entrypoint는 이상 이미지만 평가해 이미지별 AUROC를 평균하므로 그 집계도 `official_style_anomaly_image_pixel_auroc`로 함께 보존한다. 두 값은 혼용하지 않는다. 이미지·픽셀 AP와 F1-max도 저장한다. IGD는 MuSc Table 11의 RsCIN 대상이 아니며 Table 2에 AUPRO·시간·메모리 수치가 요구되지 않아 추가 실행하지 않는다.

- commit: `1ce995214ef8adf09f6c5d3dc01ce6042b472a34`
- sh: [run_igd_accelerated.sh](../source/run_igd_accelerated.sh), [환경 준비](../source/prepare_igd.sh)
- result: [가속 실행 상태·환경](../source/results/mvtec_accelerated_seed42_20261007/environment.json)
- modifications: [경로·iterator diff](../source/results/mvtec_full_seed42_20261007/compatibility.patch), [공식·실행 source SHA-256](../source/results/mvtec_full_seed42_20261007/runtime_sources.json)

```bash
bash /mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method14/source/run_igd_accelerated.sh
```

가중치·원시 결과는 `/home/test/igd_results/mvtec_full_seed42_20261007`, log는 `/home/test/igd_results/run_20261007.log`에 저장한다. 전체 학습 뒤 [evaluate_igd.py](../source/evaluate_igd.py), [finish_igd.py](../source/finish_igd.py)가 평가·검증·이 보고서와 MuSc 비교표 갱신을 자동 수행한다. 완료 전 수치는 비교표에 넣지 않는다.

2026-10-07 상태 확인 중 WSL의 uptime이 약 1분이고 학습 프로세스가 없는 것을 확인했다. 원인은 확정하지 않았다. 이전 상태는 `environment_interrupted_20261007.json`으로 보존했다. local bottle 완료 checkpoint는 재사용한다. hazelnut의 초기 recovery는 판별기 상태를 포함하지 않아 정확한 학습 재개가 불가능하므로 `recovery_interrupted_5865.pt`로 보존하고 해당 category를 처음부터 재실행했다. 새 recovery는 generator·discriminator·양쪽 optimizer·c·sigma·iteration을 모두 포함한다. 이후 중간 재개는 데이터 shuffle/RNG를 다시 시작하므로 uninterrupted 실행과 bitwise 동일하지 않음을 기록한다. 재실행 log는 `/home/test/igd_results/resumed_20261007.log`다.

bottle 첫 검증 파일과 완료 파일의 시각 차이 약 3시간 20분은 경과시간이며 순수 GPU 연산 시간으로 측정한 값이 아니다. 이 값을 이미지 수에 비례시킨 약 58시간은 local 전체에 대한 거친 외삽이다. 시스템 정지·검증·저장 등의 영향과 global의 연산량은 분리 측정하지 않았으므로 정확한 총 소요시간으로 해석하지 않는다.

## 가속 재실행

사용자 요청으로 FP32 실행을 중단하고 BF16·TF32 가속 조건으로 15개 category를 모두 처음부터 다시 실행한다. 기존 가중치와 기록은 원래 경로에 보존한다. 새 경로는 /home/test/igd_results/mvtec_accelerated_seed42_20261007이다. channels-last, cuDNN benchmark, fused Adam, persistent workers, prefetch factor4, pinned memory와 non-blocking GPU 전송을 적용한다. 모델 연산은 BF16 autocast를 사용하고 반환값을 FP32로 변환해 MS-SSIM·Gaussian·loss 계산과 NumPy 평가를 유지한다. 학습량·batch·sample_rate는 줄이지 않는다.

450개 patch의 동일한 대표 batch로 warm-up을 제외한 학습 step을 확인했다. FP32는 0.4731초, 가속은 0.2301초로 약 2.06배 빨랐다. 4개 연속 step에서 generator·discriminator loss와 gradient가 유효했다. 이 테스트는 전체 데이터의 학습 수치 동등성이나 총 소요시간을 입증하는 benchmark가 아니다. torch.compile은 별도 사전 검증에서 실제로 빨라지고 유효한 경우에만 loss에 적용한다.


MS-SSIM 컴파일 사전 검증도 통과해 실제 학습에 적용했다. 대표 loss forward/backward는 eager 0.00414초, compiled 0.00173초였고 scalar 값·gradient 유효성을 확인했다. 가속 설정과 성능 근거는 [가속 검증](../source/results/mvtec_accelerated_seed42_20261007/acceleration_preflight.json), [컴파일 검증](../source/results/mvtec_accelerated_seed42_20261007/compile_preflight.json), [global 검증](../source/results/mvtec_accelerated_seed42_20261007/global_acceleration_preflight.json)에 보존한다. 가속 환경 준비는 [prepare_igd_accelerated.sh](../source/prepare_igd_accelerated.sh)로 재현한다.


