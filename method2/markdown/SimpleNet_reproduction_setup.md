# SimpleNet MVTec AD 재현 설정

## Paper Metadata

| Item | Content |
|---|---|
| Title | SimpleNet: A Simple Network for Image Anomaly Detection and Localization |
| Authors | Zhikang Liu, Yiming Zhou, Yuansheng Xu, Zilei Wang |
| Conference / Journal | CVPR 2023 |
| Year | 2023 |
| Paper link | https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_SimpleNet_A_Simple_Network_for_Image_Anomaly_Detection_and_Localization_CVPR_2023_paper.pdf |
| GitHub / Official code | https://github.com/DonaldRR/SimpleNet |
| Reason for investigation | 논문 본문 조건으로 MVTec AD 15개 범주를 재현하고, PatchCore와 다른 판별기 기반 점수 방식을 확인하기 위함. |

## 구현체와 환경

- 공식 구현체: `DonaldRR/SimpleNet`, revision `351a2b8d4e8cfc944dbccbf9bc6ceda930c6f26b`.
- 환경: WSL2 Ubuntu, Python 3.12, PyTorch 2.11.0+cu128, CUDA GPU.
- 데이터: MVTec AD. WSL 내부 경로 `/home/test/data/mvtec`에 둠. GitHub에는 올리지 않음.
- backbone: ImageNet 사전학습 WideResNet-50-2의 `layer2`, `layer3`.

## 논문 본문에 맞춘 최종 실행 설정

| 항목 | 값 |
|---|---:|
| 범주 / seed | MVTec AD 15개 / 0 |
| batch size | 4 |
| resize / center crop imagesize | 256 / 224 |
| feature dimension | 1,536 / 1,536 |
| patch size | 3 |
| meta epochs / discriminator epochs | 40 / 4 |
| Gaussian noise standard deviation | 0.015 |
| discriminator | hidden 1,024, 2 layers, margin 0.5, learning rate 0.0002 |
| adapter | bias 없는 FC, Adam learning rate 0.0001, weight decay 0.00001 |
| data loader workers | 2 |

- 공식 코드 기본값 `329 / 288`, batch 8로 수행했던 초기 기준선은 보존함.
- 최종 표와 W32의 주 결과는 위 `256 / 224`, batch 4의 논문 본문 조건 실행임.

## 로컬 clone 수정 사항

| 파일 | 수정 | 이유 |
|---|---|---|
| `main.py` | workers 0일 때 `prefetch_factor=None` | 최신 PyTorch 호환 |
| `metrics.py` | `np.bool` → `np.bool_`, `append()` → `loc` | 최신 NumPy·pandas 호환 |
| `simplenet.py` | adapter FC의 `bias=False` | 논문에 적힌 bias 없는 adapter 반영 |
| `simplenet.py` | adapter optimizer를 Adam, weight decay `1e-5`로 설정 | 논문 학습 설정 반영 |

호환 수정과 라이브러리 버전 차이 때문에 bitwise 동일 실행은 아님. 하지만 논문에 명시된 입력 전처리, batch size, backbone, noise, adapter, optimizer, 판별기 설정은 공개 코드에서 지원하는 범위로 맞춤.

## 재현 실행 경로

- 최종 순차 실행기: [`run_simplenet_mvtec_paper_protocol_wsl.sh`](../source/run_simplenet_mvtec_paper_protocol_wsl.sh)
- 결과 취합기: [`collect_simplenet_paper_protocol_results.py`](../source/collect_simplenet_paper_protocol_results.py)
- 최종 raw table: [`SimpleNet_MVTecAD_WR50_paper_protocol_results.csv`](../source/results/SimpleNet_MVTecAD_WR50_paper_protocol_results.csv)

공식 clone, 데이터셋, 가중치, TensorBoard event와 개별 실행 파일은 대용량 산출물이므로 GitHub에 올리지 않음.
