# CutPaste MVTec 15개 카테고리 실행

## Paper Metadata

| Item | Content |
|---|---|
| Title | CutPaste: Self-Supervised Learning for Anomaly Detection and Localization |
| Authors | Chun-Liang Li, Kihyuk Sohn, Jinsung Yoon, Tomas Pfister |
| Conference / Journal | CVPR |
| Year | 2021 |
| Paper link | https://openaccess.thecvf.com/content/CVPR2021/html/Li_CutPaste_Self-Supervised_Learning_for_Anomaly_Detection_and_Localization_CVPR_2021_paper.html |
| GitHub / Official code | 저자 공식 구현 확인되지 않음. 공개 비공식 PyTorch 구현 https://github.com/Runinho/pytorch-cutpaste |
| Reason for investigation | MuSc Table2 full-shot CutPaste의 Image/Pixel AUROC를 이 PC에서 측정 |

## 문제·실험 논리

정상 이미지와 CutPaste/Scar로 만든 합성 이상 이미지를 3-way 분류하도록 representation을 학습한다. 가설은 이 proxy task가 처음 보는 실제 결함의 특징도 구분하게 만든다는 것이다. 정상 train의 learned representation으로 Gaussian density를 구성해 이상을 판단한다. MuSc의 CutPaste 행은 scratch 3-way 모델의 이미지 검출과 별도 patch 모델의 localization에 대응한다. ImageNet pretrained transfer learning 버전이나 ensemble과 구분한다.

## 고정 조건

| 항목 | 설정 |
|---|---|
| Public commit | 10d8bf71df76d3a97f0106efee1d76f81d983149 |
| Data | MVTec AD 15개 category, 정상 train 전체, test 총1725. 이미지·mask SHA와 라벨 보존 |
| Models | category별 scratch ResNet18 이미지 모델256 및 random patch64 모델, 총30개 |
| Proxy classifier | 공개 ProjectionNet의 MLP [512,128] + 3 classes. pooled512 backbone features로 GDE 구성 |
| Update count | 논문의 256 epochs × 256 parameter updates = 모델당65,536 steps. 공개 run_training.py의 기본256 steps만 실행하는 동작을 수정 |
| Batch | 정상32장 × [normal, CutPaste, Scar] = effective96, 논문 3-way batch96 |
| Optimizer | SGD lr0.03, momentum0.9, weight_decay3e-5, one-cycle cosine decay, foreach GPU |
| Augmentation | translation max10%, 전체 ColorJitter0.1 후 공개 CutPasteNormal/Scar. normal area[.02,.15], aspect[.3,1/.3], scar width[2,16]/height[10,25]/rotation[-45,45], patch jitter.1 |
| Input | fixed square256 bilinear, ImageNet mean/std normalization. patch 학습 전 RandomCrop64 |
| Density | raw pooled512의 LedoitWolf covariance 및 squared Mahalanobis, 논문 Eq2. 공개 eval의 L2-normalization과 sqrt score를 적용하지 않음 |
| Patch inference | appendix A.5대로32×32 patch stride4 →57×57 위치, aligned category는 위치별 GDE, texture/hazelnut/screw는 모든 train patch를 합친 단일 GDE |
| Aligned categories | bottle,cable,capsule,metal_nut,pill,toothbrush,transistor,zipper |
| Pixel map | Gaussian32 receptive-field kernel, sigma8, stride4 conv_transpose →256 map. coverage normalization이나 추가 smoothing 없음 |
| Image score | 이미지 전용 모델의 GDE, patch map 최대값과 구분 |
| Mask | fixed256 NEAREST, >0 |
| Repeat | seed42 1회, 논문의5회 평균/standard error와 다름. ± 미표기 |
| GPU | RTX5080, channels_last, cuDNNbenchmark, worker4/cache/pin/prefetch, foreach SGD. 사용자 요청으로 compiled FP16 + GradScaler + TF32 정책으로 재개. 실제 배치 loss/gradient 검증 후 eager FP16 → TF32 → FP32 순 fallback |

## 논문·공개 구현 차이와 정확성 한계

저자 코드 실행이 아니다. 논문의 TensorFlow 모델 대신 pinned 공개 PyTorch backbone/projection/augmentation 코드를 사용한다. scratch 모델로 초기화하고 backbone freeze를 하지 않는다. 공개 기본값 pretrained=True 및 256번만 업데이트하는 training loop를 그대로 사용하지 않는다. 공개 checkout의 tracked source는 수정하지 않는다.

공개 MLP head1 기본에 해당하는 [512,128]을 선택했다. 논문은 head의 정확한 구조를 이 설정으로 특정하지 않는다. random translation의 정확한 범위와 receptive-field Gaussian sigma 역시 본문에서 확인되지 않아 각각10%,8을 고정했으며 test로 튜닝하지 않는다. 입력 normalization, ResNet initialization 및 augmentation의 PIL/PyTorch 구현이 저자 TensorFlow 환경과 다르므로 bitwise 동등성 및 논문 수치 재현을 주장하지 않는다.

논문 §4.2는64 patch 학습을 설명하고 appendix A.5는32 patch dense inference를 설명한다. 이 실행은 둘을 그대로 따라 train64/test32로 설정하며 설명 간 차이를 감추지 않는다. Gaussian kernel은 합1로 정규화하며 overlap coverage로 나누지 않는다. 이 후처리의 불명확한 원문 세부 선택은 pixel 비교의 한계다.

논문은5개 random seeds의 평균·standard error를 보고한다. 이번 실행은 이 PC의 단일 seed42 결과로15개 category 비가중 평균만 보고한다. pretrained/ensemble/반복실험 결과를 대신 표시하지 않는다.

학습 이미지 sampling은 replacement random sampling이며 공개 Repeat3000 modulo/shuffle과 다르다. 마지막 모델을 평가하고 중간 test로 model selection하지 않는다. 256steps마다 optimizer·model·main Python/NumPy/Torch/CUDA RNG checkpoint를 저장한다. 재개시 worker RNG와 sampler stream은 완전 복원하지 못하므로 중단 없는 학습과 bitwise 동등성을 보장하지 않는다.

## 검증 및 산출물

첫 실제 학습 배치에서 FP32와 BF16 loss 및 gradient를 대조한다(loss 상대차이≤0.01, gradient 상대L2차이≤0.05). BF16 미통과시 TF32-only를 대조하고, 그것도 미통과면 FP32를 쓴다. 초기 batch의 수치 검사이며 최종 수렴이나 AUROC 일치 증명은 아니다.

GPU FP64의 LedoitWolf 계산은 실제 train features의 선택 위치 또는 실제 subset을 sklearn과 대조한다. reported density에는 train 전체 및 모든 patch를 쓰며 coreset·patch subsampling을 하지 않는다. 검증은 전체 input/raw/checkpoint SHA, 모든 Image GDE refit·Image/Pixel AUROC, 모든 receptive-field map 재처리 및 첫 map의 독립 NumPy Gaussian 누적을 포함한다. patch score 계산은 선택 위치를 독립 대조하며 모든 model forward를 이중 실행하지 않는다.

- [준비](../source/prepare_cutpaste.sh), [실행](../source/run_cutpaste.sh)
- [학습](../source/run_cutpaste.py), [평가·density](../source/evaluate_cutpaste.py), [검증](../source/finish_cutpaste.py)
- [결과](../source/results/mvtec_scratch3way_seed42_20261008/)
- Raw/model/density: `/home/test/cutpaste_results/mvtec_scratch3way_seed42_20261008`
- Log: `/home/test/cutpaste_results/run_20261008.log`

검증 passed 이후에만 MuSc Table2에 자체 측정치를 반영한다. PDF는 paper폴더에 보존한다. 연구 원본을 Obsidian에서 함께 참조하며 dataset·가상환경·대형weight/feature/precision을 복사하거나 색인하지 않는다. 결과 CSV·environment·검증 JSON을 보존하고 논문 보고 수치를 복사하지 않는다.


## 사용자 요청에 따른 가속 재실행 (2026-10-08)

기존 PID469의 CutPaste 학습/worker process group만 중단했다. 완료한 bottle 결과와 마지막 cable/image checkpoint를 보존하고 동일한 총65,536 update 목표로 재개한다. 중단 직전 checkpoint 이후의 미저장 update는 재실행된다. 완료 category는 raw SHA 검증 후 재학습·재평가하지 않는다.

compiled FP16 + TF32를 실제 각 model batch에서 eager FP16과 검증한다(loss 상대차이≤1%, gradient 상대L2≤5%, finite). FP16 대 FP32의 loss≤1%·finite를 검사하고 gradient 차이는 기록한다. 실패하면 eager FP16, TF32, FP32 순으로 검사해서 선택한다. BF16과 FP16은 동시에 쓰는 옵션이 아니며, 앞선 BF16 benchmark는 loss 기준을 넘어서 FP16을 우선한다. channels_last, cuDNN benchmark, foreach SGD, 이미지 cache, worker4/pinned memory/prefetch/비동기 GPU copy는 유지한다. 컴파일 초기 비용이 발생한다.

FP16은 GradScaler를 적용하고 checkpoint에 scaler 상태를 추가 보존한다. 기존 main RNG를 검증 전후 복원하고, worker/sampler 재개 비동등성은 남는다. category별 precision 구간이 혼합되므로 원래 FP32 실행과 최종 AUROC 동등성을 주장하지 않는다. 추론·LedoitWolf·score 검증은 기존 FP32/FP64 조건을 유지한다.

- 재실행: [run_cutpaste.sh](../source/run_cutpaste.sh)
- 가속 선택/검증: [cutpaste_acceleration.py](../source/cutpaste_acceleration.py)
- 변경 전 학습 source: [run_cutpaste_before_acceleration.py.txt](../source/results/mvtec_scratch3way_seed42_20261008/run_cutpaste_before_acceleration.py.txt)
- 각 model의 `acceleration_from_step_*.json`: 적용 시작 update, loss/gradient 오차, 실패/fallback, 선택 precision/compile 기록
- [가속 실측](CutPaste_acceleration_benchmark.md)


재시작 첫 cable 배치에서 compiled/eager FP16의 FP32 대비 loss 차이는 각각0.142%/0.175%이고 gradient 상대L2는12.31%/12.29%였다. TF32는2.20%였다. 앞선 고정 batch benchmark의1.28%와 달라서 FP16의 FP32 gradient 동등성을 주장할 수 없다. 사용자가 가속 전체 적용을 요청하여 FP16은 finite와 loss 상대차이≤1%를 gate로 사용하고 FP32 gradient는 진단값으로 보존한다. 컴파일은 eager FP16 대비 loss≤1%·gradient≤5%를 추가 검사하며 실패시 eager FP16을 쓴다. 이 정책 변경을 숨기지 않고 최종 결과 provenance에도 보존한다.


최종 재개 PID860은 cable/image 58,368 update부터 compiled FP16 + TF32를 적용했고 58,624 update 저장까지 확인했다. 이번 재개 batch에서는 FP32 대비 compiled FP16 loss 차이0.117%, gradient2.33%로 이전의 엄격 기준도 통과했다. compiled 대 eager FP16 차이는 loss0.121%, gradient1.25%로 통과했다. 이전 검증 batch의12.3% gradient 차이 기록은 그대로 남긴다. 초기 epoch 시간에는 컴파일/검증 비용이 포함되므로 전체 가속률로 해석하지 않는다.


## 데이터 공급 개선 (2026-10-08)

worker4/8/12 실측에서 worker 증가는 느려졌다. 실제 병목은 CPU batch concat과 GPU 전송 구간이었다. normal/CutPaste/Scar pinned tensor를 각각 비동기 GPU로 전송한 뒤 GPU concat으로 변경했다. 이미지 값과 순서는 기존 방식과 exact `torch.equal` 검증을 통과했고 batch32×3, augmentation, normalization, 총학습량을 유지한다. 작업 수4·main threads8 유지. 단기 GPU 학습 포함 update 평균0.0848s →0.0519s(약1.63배). 실제 학습을 checkpoint부터 재개했고 변경·검증·성능은 [데이터 공급 실측 보고](CutPaste_data_pipeline_benchmark.md)에 기록했다. 기존 데이터 준비 병목 추정은 이번 실측으로 CPU concat/전송 병목으로 구체화했다.


적용 중 일시적 PID1848 재개에서 다른 random batch의 loss 상대차이가1.18~1.49%로 기존1% gate를 넘어 FP32를 선택했다. 해당 검증 기록은 `cable/image/acceleration_from_step_64256.json`에 보존했다. 입력 값이 동일한 데이터 전송 변경 때문에 기존 학습 정밀도가 재개마다 바뀌는 것은 적절하지 않으므로, 이미 검증된 해당 model의 compiled FP16+TF32 모드를 checkpoint 재개 시 보존하도록 수정했다. 새 model은 여전히 초기 비교/선택을 하고, 재개한 학습에서도 finite loss/GradScaler 검사는 유지한다. PID2074로 다시 checkpoint부터 재개했다. `acceleration_preserved_at_step_*.json`에 모드 유지 근거를 기록하며, 임의 새 batch의 FP32 gradient/loss 동등성을 주장하지 않는다.


추가 GPU 가속 요청에 따라 별도 전송 stream·fusedSGD·CPU/GPU 동기화 축소를 비교했다. 전송 stream·fusedSGD는 느려서 제외했고, 매update의 loss CPU 조회를256update마다 모아서 수행하도록 변경했다. 실제64patch/eagerFP16 모델 단기 비교의 처리량 개선은 약1.8%로 작았다. 조건·검증·실패한 가속도 [추가 가속 실측](CutPaste_additional_gpu_acceleration.md)에 보존했다. 배치·증강·학습률·총학습량·SGD는 유지하고 PID3347로 checkpoint부터 재개했다.


## 중단 후 재개 (2026-10-09)

사용자 요청으로 PID440에서 grid/image checkpoint3,840 update부터 재개했고4,352 update 저장을 확인했다. bottle/cable/capsule/carpet의 완료 결과4개는 raw SHA 검증 후 보존했다. 이전 실행 log는 Windows 연결 경로의 training.csv 쓰기와 environment.json 접근에서 `OSError: [Errno 5] Input/output error`를 기록한다. Windows sleep/shutdown/장치 오류 중 실제 원인은 특정하지 않는다. 현재 연결 폴더 쓰기·읽기와 GPU를 확인한 뒤 기존 FP16·optimizer·GradScaler·main RNG를 복원했다. worker/sampler 재개 한계는 유지한다.

C: 여유 약36GB(33.6GiB), Ubuntu VHD도 C:에 있다. 미완료11 category의 feature와 density 배열만 최소56.1GiB(약60GB)가 필요하고 checkpoint·prediction은 별도다. WSL 내부 `df`의588GB 여유는 VHD 내부 용량이며 실제 C: 여유를 대신하지 않는다. 완주를 위해 C: 추가30GB 이상 또는 다른 저장 경로가 필요함을 사용자에게 요청했다. 기존 raw 결과나 dataset은 삭제하지 않았다. 현재 grid 학습은 진행한다.

- 재개 기록: [resume_20261009.json](../source/results/mvtec_scratch3way_seed42_20261008/resume_20261009.json)
- 실행: [run_cutpaste.sh](../source/run_cutpaste.sh)
- 실제 log: `/home/test/cutpaste_results/run_20261008.log`
