# VT-ADL BTAD 공식 구현 자체 실행

## Paper Metadata

| Item | Content |
|---|---|
| Title | VT-ADL: A Vision Transformer Network for Image Anomaly Detection and Localization |
| Authors | Pankaj Mishra, Ricardo Verk, Daniele Fornasier, Claudio Piciarelli, Gian Luca Foresti |
| Conference / Journal | IEEE/IES International Symposium on Industrial Electronics (ISIE) |
| Year | 2021 |
| Paper link | https://doi.org/10.1109/ISIE45552.2021.9576231 |
| GitHub / Official code | https://github.com/pankajmishra000/VT-ADL |
| Reason for investigation | MuSc Table14의 BTAD full-shot 비교군 Image AUROC·Pixel AUROC를 직접 측정 |

## 문제와 사전 가설

BTAD의 제품01·02·03 각각 정상 train 전체로 별도 모델을 학습하고 공식 test split 전체를 평가한다. MuSc 표가 요구하는 값은Image AUROC와Pixel AUROC다. 원논문 보고 수치는 결과 칸에 복사하지 않는다. 가설은 정상 이미지 재구성과 transformer patch 밀도에 기반한 점수가 제품별 이상 이미지·픽셀을 구분한다는 것이다. 정상 학습 손실과 test AUROC가 이 가설을 지지하는지 평가하며, test를학습·선택에사용하지 않는다.

## 실행 설정

| Item | 실행 및 논문 대조 |
|---|---|
| Dataset | 원본 `/home/test/data/btad_original/BTech_Dataset_transformed`, 제품01·02·03의train/ok전체와test/ok·ko전체. 다운로드·archive provenance는method11/source/results/btad_dataset_provenance.json에 보존 |
| Split counts | 정상train은[400,399,1000], test는[70,230,441], 총741장. 이전APRIL-GAN에사용한meta.json과파일순서·라벨·mask경로를최종대조 |
| Model | 공개VT_AE, scratch ViT dim512/depth6/heads8/MLP1024, patch64, DigitCaps 및 공개decoder, 공개MDN |
| Epoch / batch | 논문TableI·공개train.py의400epoch / batch8 |
| Optimizer | Adam, lr0.0001, weight_decay0.0001, fused CUDA 구현 |
| Gaussian components | 논문TableI의150개. 공개mdn1.py 기본10개와 다르므로 명시적으로150 적용 |
| Decoder | 논문은5개 transposed conv라고 기술하지만 공개model_res18.py는6개. 공개구조유지; 완전수치동등성미확인 |
| Train loss | 공개5*MSE -0.5*SSIM + native mdn_loss_function; noisy encoded feature Gaussian sd0.2, 공개VT_AE의원본noise함수 |
| Seed |123. 공개mvtech loader의torch.manual_seed123에맞춤; Python·NumPy·CUDA추가고정. 작가실험seed와완전동일하다고주장하지않음 |
| Image transform | 공개mvtech.py normal transform에맞춰bilinear Resize550×550→CenterCrop512→ToTensor RGB. ImageNet정규화없음 |
| Mask transform | 동일Resize·crop,ToTensor L→>0. 원본Process_mask와같음; nearest로바꾸지않음 |
| Precision |FP32 + matmul/cuDNN TF32, cuDNNbenchmark. AMP미사용 |
| Checkpoint selection | 최저mean train loss epoch. 400epoch전부실행, test로선택하지않음 |

## 공개 코드에서 보완한 것

공개train.py는MVTec하드코딩로더만사용하고BT_dataset.py는CSV·특정크기이미지용이며testmask를로드하지않는다. 실행wrapper에BTAD로더를추가하되공개Mvtec변환을적용한다. 정상ok/비정상ko 및 ground_truth/ko는동일stem으로매칭하며학습·test별SHA와라벨을보존한다. testmask와image를파일목록순서만으로zip하지않는다. 기존MVTec로더가선택하는test validation샘플은checkpoint선택에필요하지않아사용하지않는다.

원본train.py는AE의model.zero_grad만호출하여MDNgradient가batch사이에누적된다. 두모듈을함께최적화하려는목적에맞게optimizer.zero_grad(set_to_none=True)로두모듈gradient를매step초기화한다. 이것은공개train.py그대로의학습과다른명시적bug수정이다. nativeAE·MDN·SSIM·밀도손실·initialization코드는그대로사용하고외부checkout을수정하지않는다. 로더worker4/persistent/pinned transfer와fusedAdam은실행최적화다.

## 평가 정의

원본test.py의Patch_Overlap_Score(upsample1)에맞춰각이미지별MDNpatchloss를8×8에서512×512로UpsamplingBilinear2d(align_corners=True)한뒤Gaussian sigma4를적용한다. 원본의`PRO_score=roc_auc_score`는AUPRO가아니라Pixel AUROC이므로MuSc Table14의AS(Pixel AUROC)에연결한다. threshold선택은AUROC에필요하지않아생략한다. plot·SSIM중복추론을생략한다.

Image AUROC는원본함수의total_loss `MSE - loss2 + max(patch_density)`를사용한다. loss2=-SSIM이므로실제점수는MSE+SSIM+patch밀도최대다. 이상도가SSIM과같은방향으로더해지는정의지만원본공식그대로보존하며max anomaly map으로바꾸지않는다. 구성요소별값도raw에저장한다. Category결과는3개제품의비가중평균이며단일seed에±를붙이지않는다.

## 검증과 산출물

첫실제GPUbatch의finite loss/gradient·실제batch8·Gaussian150·VRAM증거를보존한다. 최종검증은3개제품400epoch, checkpoint의train손실최소선택, 모든train/test/maskSHA, raw·checkpointSHA, 이미지점수구성요소, 첫이미지의pixel map변환및전체Image/Pixel AUROC를독립재계산한다. 검증passed후에만MuSc표를갱신한다. checkpoint/원시맵은WSL에보존하고연구원본을Obsidian에서참조한다.

- commit: `20b58e2dd810e1d747b9a4132899cc474691377e`
- 준비: [prepare_vtadl.sh](../source/prepare_vtadl.sh)
- sh: [run_vtadl.sh](../source/run_vtadl.sh)
- result: [결과 폴더](../source/results/btad_seed123_20261008/)
- raw: `/home/test/vtadl_results/btad_seed123_20261008`
- log: `/home/test/vtadl_results/run_20261008.log`
- model pretraining: 없음. 제품별scratch학습을실제로수행한다.

출판DOI·저자기관원문페이지를확인했지만다운로드가능한proceedings본문을확보하지못해arXiv PDF(https://arxiv.org/pdf/2104.10036)를보존한다. 논문·공개코드차이와단일seed및localprecision차이를고려해야하며MuSc논문값과수치동등성을주장하지않는다.

첫실제GPU학습batch는batch8·Gaussian150으로유한손실/gradient검사를통과했다. 측정peak allocated는약3,368MiB이며다른batch나전체실행의최대값을대표하지않는다. [첫batch검증JSON](../source/results/btad_seed123_20261008/01/first_gpu_step.json).

## 자체 측정 결과

단위는 %. Mean은 제품3개 비가중 평균, seed123 단일 재학습이다.

| Product | Test images | Image AUROC | Pixel AUROC | Selected epoch |
|---|---|---|---|---|
| 01 | 70 | 94.4606 | 93.4933 | 400 |
| 02 | 230 | 76.3333 | 93.1403 | 400 |
| 03 | 441 | 4.0488 | 52.5862 | 382 |
| Mean | 741 | 58.2809 | 79.7399 | — |

[검증](../source/results/btad_seed123_20261008/verification.json) · [카테고리 CSV](../source/results/btad_seed123_20261008/category_metrics.csv)
