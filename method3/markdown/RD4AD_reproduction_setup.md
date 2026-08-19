# RD4AD MVTec AD 재현 설정

## Paper Metadata

| Item | Content |
|---|---|
| Title | Anomaly Detection via Reverse Distillation From One-Class Embedding |
| Authors | Hanqiu Deng, Xingyu Li |
| Conference / Journal | CVPR 2022 |
| Year | 2022 |
| Paper link | https://openaccess.thecvf.com/content/CVPR2022/papers/Deng_Anomaly_Detection_via_Reverse_Distillation_From_One-Class_Embedding_CVPR_2022_paper.pdf |
| GitHub / Official code | https://github.com/hq-deng/RD4AD |
| Reason for investigation | PatchCore와 SimpleNet 다음으로, teacher 특징을 student decoder가 역방향으로 복원하는 이상 탐지 방식을 비교하기 위함. |

## 구현체와 환경

- 공식 구현체: `hq-deng/RD4AD`, revision `6554076872c65f8784f6ece8cfb39ce77e1aee12`.
- 환경: WSL2 Ubuntu, Python 3.12, PyTorch 2.11.0+cu128, CUDA GPU.
- 데이터: MVTec AD. WSL 내부 `/home/test/data/mvtec`에 두고 GitHub에는 올리지 않음.
- backbone: ImageNet 사전학습 WideResNet-50-2.

## 실행 설정

| 항목 | 값 |
|---|---:|
| 범주 / seed | MVTec AD 15개 / 111 |
| 입력 크기 / batch size | 256 / 16 |
| 학습 epoch / 평가 간격 | 200 / 10 epoch |
| optimizer | Adam, learning rate 0.005, betas (0.5, 0.999) |
| score map smoothing | Gaussian filter sigma 4 |
| teacher | 고정된 ImageNet 사전학습 WideResNet-50-2 |
| 학습 대상 | one-class bottleneck embedding과 student decoder |

## 로컬 수정 사항

| 파일 | 수정 | 이유 |
|---|---|---|
| `main.py` | data root, category, epoch, 결과 CSV를 command option으로 추가 | WSL 내부 데이터 사용과 범주별 재현성 확보 |
| `test.py` | `np.bool` → `np.bool_`, `DataFrame.append()` → `df.loc[...]` | 최신 NumPy·pandas 호환 |

teacher·bottleneck·decoder 구조, loss, optimizer 값, 200 epoch 기본값은 바꾸지 않음.

## 재현 실행 경로

- 실행기: [`run_rd4ad_mvtec_wsl.sh`](../source/run_rd4ad_mvtec_wsl.sh)
- 논문 조건 재실행기: [`run_rd4ad_mvtec_paper_protocol_wsl.sh`](../source/run_rd4ad_mvtec_paper_protocol_wsl.sh)
- 결과 취합기: [`collect_rd4ad_results.py`](../source/collect_rd4ad_results.py)
- 최종 raw table: [`RD4AD_MVTecAD_WR50_results.csv`](../source/results/RD4AD_MVTecAD_WR50_results.csv)
- 논문 조건 재실행 raw table: [`RD4AD_MVTecAD_WR50_paper_protocol_results.csv`](../source/results/RD4AD_MVTecAD_WR50_paper_protocol_results.csv)

공식 clone, dataset, ImageNet 가중치, checkpoint는 대용량 실행 산출물이므로 GitHub에 올리지 않음.
