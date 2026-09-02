# AnomalyCLIP VisA→MVTec AD 실행 보고

## Paper Metadata

| Item | Content |
|---|---|
| Title | AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection |
| Authors | Qihang Zhou, Guansong Pang, Yu Tian, Shibo He, Jiming Chen |
| Conference / Journal | ICLR |
| Year | 2024 |
| Paper link | https://openreview.net/pdf?id=buC4E91xZE |
| GitHub / Official code | https://github.com/zqhang/AnomalyCLIP |
| Reason for investigation | source domain으로 학습한 prompt를 target에 zero-shot으로 적용하는 CLIP 기반 anomaly detection 방법을 실행·검토하기 위함 |

## 실행 요약

- upstream revision: `3911738c0867544f545a076ad78f3f11d9ecbfdf`
- protocol: VisA source → MVTec AD target.
- target MVTec AD는 prompt 학습과 normal reference에 사용하지 않고 평가에만 사용함.
- 학습: `ViT-L/14@336px`, 518px, batch 8, 15 epoch, layer 24, depth 9, `n_ctx=12`, `t_n_ctx=4`, seed 111.
- 평가: MVTec AD 15개 범주, `sigma=4`, Image AUROC / Image AP / Pixel AUROC / Pixel AUPRO.

## 결과

| Metric | Mean (%) | Raw result |
|---|---:|---|
| Pixel AUROC | 91.0 | [`results.csv`](../source/results/anomalyclip_visa_to_mvtec/results.csv) |
| Pixel AUPRO | 81.6 | [`results.csv`](../source/results/anomalyclip_visa_to_mvtec/results.csv) |
| Image AUROC | 91.6 | [`results.csv`](../source/results/anomalyclip_visa_to_mvtec/results.csv) |
| Image AP | 96.2 | [`results.csv`](../source/results/anomalyclip_visa_to_mvtec/results.csv) |

## 재현 경로

- 실행 script: [`run_anomalyclip_visa_to_mvtec_wsl.sh`](../source/run_anomalyclip_visa_to_mvtec_wsl.sh)
- VisA metadata helper: [`generate_anomalyclip_metadata.py`](../source/generate_anomalyclip_metadata.py)
- official log → CSV helper: [`collect_anomalyclip_results.py`](../source/collect_anomalyclip_results.py)
- WSL cache-path compatibility patch: [`anomalyclip_wsl_compat.patch`](../source/anomalyclip_wsl_compat.patch)
- 예측지도 생성: [`visualize_anomalyclip_examples.py`](../source/visualize_anomalyclip_examples.py)

모델 checkpoint, CLIP weight, VisA·MVTec AD dataset, upstream clone은 용량과 repository 규칙 때문에 포함하지 않음.

## 예측지도

각 사례는 input, anomaly heatmap, overlay 파일로 저장함. 빨강·노랑은 상대적으로 높은 anomaly score 위치임.

- [`bottle / broken_large`](../source/results/anomalyclip_visa_to_mvtec/visualizations/bottle_broken_large_overlay.png)
- [`pill / color`](../source/results/anomalyclip_visa_to_mvtec/visualizations/pill_color_overlay.png)
- [`transistor / bent_lead`](../source/results/anomalyclip_visa_to_mvtec/visualizations/transistor_bent_lead_overlay.png)

## 범위와 한계

- 논문 및 공식 script의 model/data/hyperparameter protocol을 따름.
- 논문은 PyTorch 2와 RTX 3090을 사용했고, 이번 실행은 PyTorch 2.11 / CUDA 12.8 / RTX 5080임. 따라서 bitwise 동일 재현은 주장하지 않음.
- 대표 anomaly map은 점수의 공간 분포를 보여주는 보조 자료이며, 위치 성능 평가는 raw CSV의 Pixel AUROC/AUPRO로 확인해야 함.
