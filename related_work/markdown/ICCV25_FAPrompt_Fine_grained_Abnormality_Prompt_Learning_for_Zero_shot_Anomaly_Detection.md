# FAPrompt: Fine-grained Abnormality Prompt Learning for Zero-shot Anomaly Detection

## Paper Metadata

| Item | Content |
|---|---|
| Title | Fine-grained Abnormality Prompt Learning for Zero-shot Anomaly Detection |
| Authors | Jiawen Zhu, Yew-Soon Ong, Chunhua Shen, Guansong Pang |
| Conference / Journal | International Conference on Computer Vision (ICCV) |
| Year | 2025 |
| Paper link | https://www.openaccess.thecvf.com/content/ICCV2025/papers/Zhu_Fine-grained_Abnormality_Prompt_Learning_for_Zero-shot_Anomaly_Detection_ICCV_2025_paper.pdf |
| GitHub / Official code | https://github.com/mala-lab/FAPrompt |
| Reason for investigation | W39의 H1: target 고유의 정상·결함 경계를 직접 보지 못할 때 기존 기준 상태가 오탐·누락을 낼 수 있는지 검토하기 위함. |

## 한 문장 요약

FAPrompt는 target 학습 이미지 없이, coarse한 abnormal prompt 하나 대신 여러 fine-grained abnormal prompt를 학습하고 test image에서 얻은 abnormality prior로 prompt를 동적으로 조정하여 cross-dataset zero-shot AD를 개선하는 방법임 [1].

## 이 논문이 지적한 문제

- 기존 zero-shot AD의 handcrafted 또는 learnable prompt는 `damaged`, `defective`처럼 coarse한 이상 의미를 주로 표현함.
- 따라서 dataset마다 다른 세부 결함 양상을 포착하기 어렵고, 보지 못한 target dataset에서 generalization이 약해질 수 있음 [1].
- 이 문제는 W39의 H1과 완전히 같은 문장은 아니지만, **정적 이상 기준이 target의 세부 결함 의미를 충분히 대표하지 못할 수 있다**는 원인 가설을 구체화함.

## 방법

| 구성 | 하는 일 | W39 H1과의 연결 |
|---|---|---|
| CAP (Compound Abnormality Prompt learning) | 하나의 coarse abnormal prompt 대신 상호 보완적인 여러 abnormal prompt를 학습함 | 이상 기준 `c`가 다양한 결함 의미를 담도록 만듦 |
| DAP (Data-dependent Abnormality Prior learning) | 각 test image의 abnormal feature에서 sample-wise prior를 얻어 abnormal prompt를 동적으로 조정함 | target 학습 data나 normal reference 없이 query 자체에 맞춰 기준 `c`를 조정함 |
| multi-scale visual feature와 prompt similarity | 이미지·위치 특징과 조정된 prompt의 유사도로 image/pixel score를 계산함 | W39의 visual-text similarity 계열과 같은 출력 단위를 유지함 |

## 논문 결과와 해석 범위

- 논문은 MVTec AD, VisA를 포함한 산업·의료 19개 데이터셋을 평가했고, 산업 데이터에서 기존 최고 비교 방법 대비 최대 Image AUROC +3.9%p, Image AP +3.0%p를 보고함 [1].
- 논문은 기본적으로 MVTec AD source로 학습해 다른 dataset을 평가하고, MVTec AD 평가는 VisA source로 바꾸는 cross-dataset protocol을 사용함 [1].
- 이 결과는 정적 coarse prompt보다 fine-grained·query-dependent prompt가 여러 target에서 유리할 수 있다는 prior-work evidence임.
- 그러나 현재 저장소의 WinCLIP 0/1-shot 결과와 같은 model, category, split, seed로 비교한 결과가 아님. 따라서 FAPrompt의 수치를 W39 H1의 직접 검증 결과로 사용하지 않음.

## W39에 적용할 수 있는 이유와 아직 바꾸지 않을 부분

- 적용 가능한 이유: H1의 `target 정렬 부족`을 target training data 없이 완화할 수 있는 방향을 보여줌. query에서 얻은 prior를 써서 기준 상태를 sample별로 조정함.
- 아직 바꾸지 않을 부분: CAP·DAP를 현재 방법에 바로 추가하지 않음. 먼저 동일 category·split에서 normal reference 유무와 anomaly map을 비교해 H1의 오탐·누락 현상이 실제로 관찰되는지 확인해야 함.
- 다음 검증: WinCLIP 0-shot/1-shot 조건에서 같은 query image의 input, ground-truth mask, anomaly map을 모아 reference가 줄이는 오탐·누락 유형을 기록함. 이후 FAPrompt의 query-dependent prior가 그 유형을 target-free로 줄일 후보인지 검토함.

## 근거 경로

- 원문 PDF: [`ICCV25_FAPrompt_Fine_grained_Abnormality_Prompt_Learning_for_Zero_shot_Anomaly_Detection.pdf`](../paper/ICCV25_FAPrompt_Fine_grained_Abnormality_Prompt_Learning_for_Zero_shot_Anomaly_Detection.pdf)
- 현재 H1: [`meetings/2026-W39_brief.md`](../../meetings/2026-W39_brief.md)
- 기존 WinCLIP raw comparison: `method7/source/results/winclip_target_comparison.csv`

## 참고문헌

[1] Zhu, Jiawen, Yew-Soon Ong, Chunhua Shen, and Guansong Pang. "Fine-grained Abnormality Prompt Learning for Zero-shot Anomaly Detection." *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 2025, pp. 22241-22251.
