# P-SVDD(Patch SVDD) BTAD 공식 구현 실행

## Paper Metadata

| Item | Content |
|---|---|
| Title | Patch SVDD: Patch-level SVDD for Anomaly Detection and Segmentation |
| Authors | Jihun Yi, Sungroh Yoon |
| Conference / Journal | ACCV |
| Year | 2020 |
| Paper link | https://openaccess.thecvf.com/content/ACCV2020/html/Yi_Patch_SVDD_Patch-level_SVDD_for_Anomaly_Detection_and_Segmentation_ACCV_2020_paper.html |
| GitHub / Official code | https://github.com/nuclearboy95/Anomaly-Detection-PatchSVDD-PyTorch |
| Reason for investigation | MuSc Table14 BTAD full-shot 비교군 Image/Pixel AUROC를 이PC에서 직접 측정 |

## 문제·가설

제품01·02·03의모든정상train으로제품별encoder를학습하고test741장을평가한다. 가설은인접patch의SVDD거리손실과8방향상대위치self-supervision이정상patch표현을만들며, 정상trainpatch와의최근접L2거리로이상영역을구분한다는것이다. 실제Image AUROC와Pixel AUROC로평가하고논문보고수치를대신사용하지않는다.

## 고정 실행 조건

| 항목 | 설정·근거 |
|---|---|
| 구현 | 공식codes/networks.py EncoderHier(K64,D64), 내부EncoderDeep(K32), PositionClassifier두개. pretrained model미사용, scratch학습 |
| Dataset | BTAD원본,train[400,399,1000],test[70,230,441]. method11meta.json과경로·라벨·mask대조 |
| Epoch | 공식main_train.py의300개loop slot: slot0학습없음,slot1..299학습. 따라서299개gradient학습epoch. slot0을추가학습으로바꾸지않음 |
| Sampling | 공식repeat100, batch64, shuffle. 각slot은ceil(train_images×100/64)updates |
| Optimizer | Adam lr1e-4, weight_decay0, fused CUDA |
| λ | 공식CLI기본1고정. README의bottle예시는1e-3이고논문은category에따라λ가달라짐을논의한다. MuSc BTAD의정확λ는공개본문만으로확정하지못해기본1사용, BTADtest로튜닝하지않음 |
| Seed | local42. 원본seed미지정이며실험seed동등성주장하지않음 |
| Input | native현재Pillow RGB resize256×256(defaultBICUBIC), 정상train의위치별RGB평균을빼고255로나눔. ImageNet정규화없음 |
| Mask | native현재Pillow resize256×256→>128. BTADground_truth를image stem으로연결 |
| Training | 원본pos64+pos32+λ×(SVDD64+SVDD32), RGBshift/독립pixelGaussiannoise sd0.02, 좌표sampling원본그대로 |
| Inference | 원본infer(batch64) K64/S16 KDTree, K32/S4 NGT(approximate) NN1. GPUencoder추출후CPU최근접검색 |
| Fusion | native patch score를patch영역에분배·overlap평균후두scale pixel map곱. image는곱map의pixel최대. sum/각scale결과도별도보존 |
| Precision | FP32+TF32 matmul/cuDNN, cuDNNbenchmark, AMP없음 |

## 원본과의 차이·정확성 한계

원본MVTec loader는png/good와mask목록순서에의존하므로wrapper에서BTAD ok/ko및bmp/png를처리하고동일stem mask를매칭한다. 공식학습dataset·crop·loss·encoder·classifier코드를수정하지않는다. 원본NormalizedLinear가no_grad에서정규화weight를만드는동작도보존한다.

원본은매slot전체test평가후encoder를덮어쓰지만checkpoint를test성능으로선택하지않는다. 중간test평가를생략하고마지막모델만평가한다. 이는모델선택규칙을유지하는시간최적화이나RNG소모·worker수변경등으로원본학습과bitwise동일하다고주장하지않는다. worker4/persistent/prefetch/pin_memory·fusedAdam을사용하며epoch별encoder·classifier·optimizer·main RNG전체를원본WSL에저장한다. 재개시worker RNG가완전복구되지않으므로중단없는실행과bitwise학습동일성은보장하지않는다.

NGT는원본처럼근사NN이다. 정확NN이나coreset으로바꾸지않는다. 최신ngt2.8.0.post1과현재torch/Pillow를사용하므로작가의당시환경과일치하지않는다. API·거리계산은작은실제BTAD입력으로검증하고그출력을보존한다. MuSc논문조건에가까운실행이지만λ/seed/환경이확정되지않아논문수치동등성의증거로해석하지않는다.

## 검증·산출물

첫실제GPU학습batch의finite손실·gradient를보존한다. 작은실제이미지에서nativeencoder·KDTree·NGT API및검색된vector거리계산을검증하며KDTree는brute-force최근접과대조한다. 최종verifier는전체train/test/maskSHA,raw/checkpointSHA,300slots·299updateepochs,제품별원본sum/product fusion,전체Image/Pixel AUROC를독립재계산한다. 검증passed이후에만MuSc비교표에반영한다.

- commit: `934d6238e5e0ad511e2a0e7fc4f4899010e7d892`
- 준비: [prepare_psvdd.sh](../source/prepare_psvdd.sh)
- sh: [run_psvdd.sh](../source/run_psvdd.sh)
- result: [결과 폴더](../source/results/btad_seed42_20261008/)
- raw: `/home/test/psvdd_results/btad_seed42_20261008`
- log: `/home/test/psvdd_results/run_20261008.log`

Primary PDF는CVF가공개하는ACCV2020version을paper폴더에보존한다. 연구원본을Obsidian에서함께참조하고dataset·checkpoint·가상환경을복사하거나색인하지않는다.

## 자체 측정 결과

수치는 %, 제품3개 비가중 평균이다. seed42단일 학습이며 ±를 붙이지 않는다.

| Product | Test images | Image AUROC | Pixel AUROC |
|---|---|---|---|
| 01 | 70 | 80.3693 | 87.3481 |
| 02 | 230 | 78.9000 | 96.8814 |
| 03 | 441 | 50.0000 | 50.0000 |
| Mean | 741 | 69.7564 | 78.0765 |

[검증](../source/results/btad_seed42_20261008/verification.json) · [CSV](../source/results/btad_seed42_20261008/category_metrics.csv)
