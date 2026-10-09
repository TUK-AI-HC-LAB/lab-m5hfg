# CutPaste 가속 속도 대조

cable image 모델의 저장된 checkpoint에서 실제 MVTec 정상32장과 CutPaste/Scar를 합친96장 batch로 측정했다. 각 방식마다 같은 모델 상태에서 시작하고 warmup3회 후20회 GPU 업데이트의 median을 기록했다. 데이터 로딩·checkpoint 저장·전체 평가 시간은 제외한다.

- Checkpoint step: 55040
- Checkpoint SHA: `2544522224da4c35676d295106defaf0707ff979250d1d8d96073da7f87364bc`
- 기존 학습은 측정 중만 일시 중지하고 원래 정밀도로 재개했다.

| 방식 | Update ms | FP32 대비 속도 | Loss 상대차이 | Gradient 상대L2차이 |
|---|---:|---:|---:|---:|
| fp32 | 123.68 | 1.00× | 0.000% | 0.000% |
| tf32 | 79.20 | 1.56× | 0.060% | 1.086% |
| bf16 | 42.84 | 2.89× | 1.883% | 4.115% |
| fp16 | 37.50 | 3.30× | 0.236% | 1.277% |
| compiled_bf16 | 36.04 | 3.43× | 1.537% | 3.970% |

FP16과 TF32는 이 checkpoint의 기존 기준(loss≤1%, gradient≤5%)을 만족한다. BF16과 compiled BF16은 finite지만 loss 차이가1%를 넘는다.

초기 학습 batch 검사와 이미 학습된 checkpoint의 대조는 서로 다른 질문이다. 초기 검사에서 가속이 꺼졌다고 이후에도 수치 기준을 통과하지 못한다고 단정할 수 없다. 반대로 한 checkpoint의 손실·기울기 검사는 최종 AUROC 동등성 증명이 아니다.

torch.compile(BF16)의 최초 forward/backward는 약25.67초였다. 컴파일 비용은 위 반복 시간에 포함하지 않는다. 반복 CUDA 연산의 결과로 전체 실행 ETA를 확정하지 않으며, 평가/GDE/CPU augmentation 및 아직 측정하지 않은 patch 학습의 비중을 고려해야 한다.

[원시 측정 JSON](../source/results/mvtec_scratch3way_seed42_20261008/acceleration_benchmark.json)
