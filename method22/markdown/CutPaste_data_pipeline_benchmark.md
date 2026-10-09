# CutPaste 데이터 공급 개선 실측 (2026-10-08)

실제 cable/image 학습 checkpoint와 기존 CutPaste3Way augmentation, base batch32/effective96, compiled FP16+TF32를 사용했다. 실제 학습 process group을 측정 동안만 정지하고 watchdog/finally로 재개했다. 각 설정8회 warmup 후32회 update 시간을 GPU synchronize로 측정했다. 실제 checkpoint/optimizer를 변경하지 않은 별도 benchmark 모델이며 평가와 checkpoint 저장 비용은 제외한다.

작업 수4/8/12 비교에서는4개가 가장 빨랐다. data wait는 거의 없고 CPU concat으로 pinned tensor를 새 unpinned tensor에 복사한 후 GPU 전송/배치 layout 변환하는 구간이 병목이었다. 작업 수 증가가 해결하지 못했다.

|Workers|배치 합치기|Main CPU threads|평균 update(s)|평균 data wait(s)|평균 합치기/전송(s)|평균 GPU 학습(s)|
|---|---|---|---|---|---|---|
|4|cpu_concat|8|0.084768|0.000217|0.044001|0.040551|
|4|gpu_concat|8|0.051945|0.000672|0.004383|0.046890|
|4|gpu_concat|1|0.065431|0.012405|0.003863|0.049163|
|8|gpu_concat|1|0.070582|0.000279|0.004617|0.065686|

선택: workers4/main threads8 유지, 각 pinned tensor를 별도로 비동기 GPU 전송한 뒤 GPU에서 concat한다. normal→CutPaste→Scar 순서·batch96·channels_last·augmentation·normalization·optimizer·총 update 수를 유지한다. 비교한 각 GPU concat 설정의 첫 배치에서 기존 CPU concat 입력과 `torch.equal`로 모든 값/순서가 동일함을 확인했다. 모델 precision은 이전 compiled FP16+TF32 설정을 유지한다.

평균 update 0.0848s → 0.0519s로 약1.63배였다. 합치기/전송 구간은 44.0ms → 4.4ms였다. 단기 실측이며 category/patch 크기·CPU 부하에 따라 달라진다. 전체 종료시간이나 GPU 사용률 보장은 아니다.

저장 checkpoint부터 재개하며 worker/sampler RNG stream은 restart 전과 bitwise 동일하지 않다. 기존 완료 category/raw 결과는 보존한다. 반복 진행률 polling이나 자동 ETA 계산은 추가하지 않았다.

- 실행 source: [run_cutpaste.py](../source/run_cutpaste.py), [run_cutpaste.sh](../source/run_cutpaste.sh)
- 재현 benchmark: [benchmark_data_pipeline.py](../source/benchmark_data_pipeline.py), 기본 transfer 비교 / `--workers-only` worker 비교
- [작업 수 비교 raw](../source/results/mvtec_scratch3way_seed42_20261008/data_pipeline_benchmark.json)
- [전송 방식 비교 raw](../source/results/mvtec_scratch3way_seed42_20261008/data_transfer_benchmark.json)
- [재개 기록](../source/results/mvtec_scratch3way_seed42_20261008/data_pipeline_restart.json)
- 실제 학습 log: `/home/test/cutpaste_results/run_20261008.log`
- Raw/checkpoint: `/home/test/cutpaste_results/mvtec_scratch3way_seed42_20261008`
- commit: `10d8bf71df76d3a97f0106efee1d76f81d983149` (기존 공개 구현 unchanged)


적용 중 일시적 PID1848 재개에서 다른 random batch의 loss 상대차이가1.18~1.49%로 기존1% gate를 넘어 FP32를 선택했다. 해당 검증 기록은 `cable/image/acceleration_from_step_64256.json`에 보존했다. 입력 값이 동일한 데이터 전송 변경 때문에 기존 학습 정밀도가 재개마다 바뀌는 것은 적절하지 않으므로, 이미 검증된 해당 model의 compiled FP16+TF32 모드를 checkpoint 재개 시 보존하도록 수정했다. 새 model은 여전히 초기 비교/선택을 하고, 재개한 학습에서도 finite loss/GradScaler 검사는 유지한다. PID2074로 다시 checkpoint부터 재개했다. `acceleration_preserved_at_step_*.json`에 모드 유지 근거를 기록하며, 임의 새 batch의 FP32 gradient/loss 동등성을 주장하지 않는다.
