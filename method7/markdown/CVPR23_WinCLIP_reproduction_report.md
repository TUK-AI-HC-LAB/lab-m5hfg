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

### MVTec AD와 VisA target 비교

같은 모델·입력 크기·prompt ensemble·seed에서 target dataset만 변경함. WinCLIP에는 source-domain 학습이 없고, few-normal-shot에서는 각 target의 정상 reference만 사용함.

| Condition | MVTec Pixel AUROC (%) | VisA Pixel AUROC (%) | MVTec AUPRO (%) | VisA AUPRO (%) | MVTec Image AUROC (%) | VisA Image AUROC (%) |
|---|---:|---:|---:|---:|---:|---:|
| 0-shot | 82.3 | 73.2 | 61.9 | 51.0 | 90.4 | 75.5 |
| 1-shot | 93.6 | 94.7 | 84.1 | 80.5 | 93.7 | 83.8 |
| 2-shot | 93.8 | 95.1 | 84.8 | 81.2 | 93.7 | 83.4 |
| 4-shot | 94.2 | 95.2 | 85.4 | 81.4 | 95.3 | 84.1 |

VisA raw CSV: [`0-shot`](../source/results/winclip_visa/zero_shot/results.csv), [`1-shot`](../source/results/winclip_visa/one_shot/results.csv), [`2-shot`](../source/results/winclip_visa/two_shot/results.csv), [`4-shot`](../source/results/winclip_visa/four_shot/results.csv).

한 파일에서 보는 metric별 비교: [`winclip_target_comparison.csv`](../source/results/winclip_target_comparison.csv).

VisA와 MVTec AD는 범주·결함·test split이 다르므로, 이 표는 dataset 난이도를 확정하는 비교가 아님. 같은 구현에서 target dataset과 target normal reference가 결과에 주는 변화를 확인하는 표임.

## 재현 경로와 범위

- 실행 script: [`run_winclip_mvtec_wsl.sh`](../source/run_winclip_mvtec_wsl.sh)
- official log → CSV helper: [`collect_winclip_results.py`](../source/collect_winclip_results.py)
- 논문 원 저자의 공식 code release는 확인하지 못했음. 따라서 결과는 논문 조건을 구현한 공개 재현 코드의 결과이며, 원 저자 runtime과 bitwise 동일 재현을 주장하지 않음.
- dataset, model weight, upstream clone, raw log는 repository에 포함하지 않음.
