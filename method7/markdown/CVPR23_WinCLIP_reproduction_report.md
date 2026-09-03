# WinCLIP MVTec AD 재현 보고

## Paper Metadata

| Item | Content |
|---|---|
| Title | WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation |
| Authors | Jongheon Jeong, Yang Zou, Taewan Kim, Dongqing Zhang, Avinash Ravichandran, Onkar Dabeer |
| Conference / Journal | CVPR |
| Year | 2023 |
| Paper link | https://openaccess.thecvf.com/content/CVPR2023/papers/Jeong_WinCLIP_Zero-Few-Shot_Anomaly_Classification_and_Segmentation_CVPR_2023_paper.pdf |
| GitHub / Official code | No original-author release confirmed; reproduction: https://github.com/zqhang/Accurate-WinCLIP-pytorch |
| Reason for investigation | target normal reference를 쓰지 않는 WinCLIP과 쓰는 WinCLIP+의 정보 조건 및 성능 변화를 확인하기 위함 |

## 실행 조건

- Dataset: MVTec AD 15개 범주.
- Model: LAION-400M CLIP `ViT-B/16+`.
- Input: 240px, window stride 1 on ViT patch embeddings.
- Prompts: 논문의 normal/abnormal state word와 template compositional ensemble.
- Conditions: WinCLIP 0-shot, WinCLIP+ 1/2/4 normal-shot.
- Seed: 10.
- reproduction revision: `eef4f8cce6a80eddfa07415b503c22c6c3427351`.

## 결과

| Condition | Target normal reference | Pixel AUROC (%) | AUPRO (%) | Image AUROC (%) | Raw CSV |
|---|---:|---:|---:|---:|---|
| WinCLIP 0-shot | 0장 | 82.3 | 61.9 | 90.4 | [`results.csv`](../source/results/winclip_mvtec/zero_shot/results.csv) |
| WinCLIP+ 1-shot | 범주별 1장 | 93.6 | 84.1 | 93.7 | [`results.csv`](../source/results/winclip_mvtec/one_shot/results.csv) |
| WinCLIP+ 2-shot | 범주별 2장 | 93.8 | 84.8 | 93.7 | [`results.csv`](../source/results/winclip_mvtec/two_shot/results.csv) |
| WinCLIP+ 4-shot | 범주별 4장 | 94.2 | 85.4 | 95.3 | [`results.csv`](../source/results/winclip_mvtec/four_shot/results.csv) |

## 재현 경로와 범위

- 실행 script: [`run_winclip_mvtec_wsl.sh`](../source/run_winclip_mvtec_wsl.sh)
- official log → CSV helper: [`collect_winclip_results.py`](../source/collect_winclip_results.py)
- 논문 원 저자의 공식 code release는 확인하지 못했음. 따라서 결과는 논문 조건을 구현한 공개 재현 코드의 결과이며, 원 저자 runtime과 bitwise 동일 재현을 주장하지 않음.
- dataset, model weight, upstream clone, raw log는 repository에 포함하지 않음.
