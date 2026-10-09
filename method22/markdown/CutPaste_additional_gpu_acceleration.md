# CutPaste 추가 GPU 가속 비교 (2026-10-08)

base batch32/effective96·64patch·동일 augmentation·모델 checkpoint를 유지하고 실제 학습을 잠시 정지하여 비교했다. 각 설정8회 warmup 뒤32updates 구간 전체를 시작/끝 CUDA synchronize로 측정했다. GPU에 작업을 비동기 제출하므로 개별 단계의 host 시간은 GPU 실행 시간이 아니다. 아래는 전체 측정 구간의 update당 벽시계 시간이다. checkpoint 저장·평가·전체 AUROC 동등성은 이 benchmark의 검증 대상이 아니다.

|비교|설정|compiled model|평균 update(s)|
|---|---|---|---|
|전송 겹치기|gpu_concat|True|0.013623|
|전송 겹치기|prefetch|True|0.019002|
|전송 겹치기|prefetch_fused|True|0.015302|
|compiled 동기화/optimizer 비교|gpu_concat_sync|True|0.014301|
|compiled 동기화/optimizer 비교|gpu_concat|True|0.013828|
|compiled 동기화/optimizer 비교|gpu_concat_fused|True|0.014339|
|실제 eager 모델|gpu_concat_sync|False|0.013825|
|실제 eager 모델|gpu_concat|False|0.013586|

별도 CUDA stream/백그라운드 prefetch와 fusedSGD는 이 설정에서 빨라지지 않아 실제 실행에 적용하지 않았다. 기존 foreachSGD·GPU concat·worker4를 유지한다. 순차 동기화 제거는 실제 eager 모델에서 13.825ms →13.586ms(약1.8% 처리량 증가)로 차이가 작아 측정 변동의 영향을 받을 수 있다. 큰 가속이나 GPU 사용률100%를 주장하지 않는다.

실제 코드에서는 finite loss 확인을 CUDA 비동기 assertion으로 수행하고 detached scalar loss를256updates 동안 보관한다. 보고 시 한 번에 CPU로 보내 FP64 NumPy 평균을 구한다. 기존 loss 값은 원래 FP32 tensor에서 Python float64로 변환한 값이므로 평균 정밀도를 유지한다. 모델 backward·GradScaler·SGD·학습률·batch·총update를 바꾸지 않는다. 실행 중 detached scalar256개를 보관하는 비용은 작다. checkpoint는 기존256update 간격을 유지한다. 재개는 기존 검증된 이 model의 FP16/compile mode를 유지한다.

- 추가 비교 script: [benchmark_data_pipeline.py](../source/benchmark_data_pipeline.py)
- 전송 겹치기 재현: `--overlap --prefetch`
- compiled 동기화/optimizer 비교: `--overlap`
- 실제 model compile 조건 비교: `--overlap --actual`
- [전송 겹치기 raw](../source/results/mvtec_scratch3way_seed42_20261008/data_overlap_benchmark.json)
- [compiled 비교 raw](../source/results/mvtec_scratch3way_seed42_20261008/data_sync_benchmark.json)
- [실제 모델 비교 raw](../source/results/mvtec_scratch3way_seed42_20261008/data_async_actual_benchmark.json)
- [실제 학습 코드](../source/run_cutpaste.py), [실행 script](../source/run_cutpaste.sh)
- 변경 전 source: [run_cutpaste_before_async_loss.py.txt](../source/results/mvtec_scratch3way_seed42_20261008/run_cutpaste_before_async_loss.py.txt)
- Raw/model: `/home/test/cutpaste_results/mvtec_scratch3way_seed42_20261008`
- 학습 log: `/home/test/cutpaste_results/run_20261008.log`
- pinned commit: `10d8bf71df76d3a97f0106efee1d76f81d983149`
