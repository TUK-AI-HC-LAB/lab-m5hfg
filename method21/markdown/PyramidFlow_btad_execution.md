# PyramidFlow BTAD 저자 구현 실행

## Paper Metadata

| Item | Content |
|---|---|
| Title | PyramidFlow: High-Resolution Defect Contrastive Localization Using Pyramid Normalizing Flow |
| Authors | Jiarui Lei, Xiaobo Hu, Yue Wang, Dong Liu |
| Conference / Journal | CVPR |
| Year | 2023 |
| Paper link | https://openaccess.thecvf.com/content/CVPR2023/html/Lei_PyramidFlow_High-Resolution_Defect_Contrastive_Localization_Using_Pyramid_Normalizing_Flow_CVPR_2023_paper.html |
| GitHub / Official code | https://github.com/FourthM/PyramidFlow (README의 과거 gasharper 주소는 현재 unavailable) |
| Reason for investigation | MuSc Table14 BTAD full-shot의 Image/Pixel AUROC를 이 PC에서 직접 측정 |

## 문제·가설

정상 BTAD 이미지 쌍의 latent pyramid 차이를 줄이도록 flow를 학습한다. 가설은 정상 train 전체의 latent 평균을 template로 쓰고 그 template에서 벗어난 영역을 이상으로 볼 수 있다는 것이다. MuSc의 BTAD 비교값은 PyramidFlow 원문 Table4의 Ours(Res18)에 대응하므로 pretrained ResNet18 버전을 실행한다. 무사전학습 FNF 버전과 구분한다.

## 고정 조건

| 항목 | 설정 |
|---|---|
| Official commit | c463b1cb0c2b084cdbae290234e3f8657298e981 |
| autoFlow dependency | 보존 fork https://github.com/twimclee/pyramidflow-moai, fa322f9175966ebed2759b0422460b401c318b05 의 autoFlow.py만 import |
| Model | 공식 PyramidFlow, resnetX18, 4 pyramid layers, 4 stacks, 64 channels, kernel7 |
| Dataset | BTAD train [400,399,1000], test [70,230,441], 제품별 full-shot, augmentation 없음 |
| Epoch / Batch | 공식 train.py의 15 epochs, batch2, shuffle, drop_last=True |
| Optimizer | Adam lr2e-4, eps1e-4, wd1e-5, betas(0.5,0.9), grad clip1.0, fused CUDA |
| Seed | 공식 seed0 |
| Encoder | ImageNet ResNet18 V1의 conv1/bn1/relu/maxpool/layer1. native no_grad로 고정, training 중 BN running stats 업데이트는 그대로 보존 |
| Input / Mask | 1024×1024 bilinear RGB + ImageNet normalization / 256×256 bilinear mask + native round int |
| Volume Norm | CVN dims(0,1). 공개 auto 분기에서 BTAD 이름은 SVN 목록에 없으므로 기본 fallback CVN 사용 |
| Loss | 공식 BatchDiffLoss p2 → compose_pyramid → channel mean → FFT2 abs mean |
| Template | 마지막 모델에서 augmentation 없이 정상 train 전체 batch1 latent 평균 |
| Evaluation | test batch4, scale별 abs(latent-template) → native compose → channel mean. 별도 Gaussian smoothing 없음. image score는 map 최대 |
| GPU optimization | native SequentialNet(savemem=False), cuDNNbenchmark, worker4/pin/persistent, fusedAdam. FP32, TF32는 실제 loss/gradient 대조 통과시만 허용. FFT 때문에 AMP 미사용 |
| Selection | 마지막 epoch15. test 결과로 checkpoint 선택하지 않음 |

## 코드 수정 및 논문 조건 한계

공식 tracked source는 수정하지 않고 BTAD loader·실행 wrapper로 호출한다. Python import가 만드는 untracked __pycache__는 source 변경으로 취급하지 않는다. 공식 NumPy LU mask buffer가 float64가 될 수 있어 전체 모델을 .float()로 명시적으로 FP32 변환한다. 제품03의 원본은 800×600이라 공개 Resize(int)의 종횡비 유지 결과가 pyramid divisibility 조건에 맞지 않는다. 논문 supplement의 pretrained 1024×1024 입력에 맞춰 세 제품 모두 fixed square resize한다. mask도 fixed square256으로 대응한다.

원문 Table4는 Res18을 사용한다. 그러나 BTAD 제품별 volumeNorm/layer 설정과 checkpoint 선택 조건은 공개되지 않았다. 이번 실행의 CVN·4layers는 공개 auto fallback 및 default이며 원문의 정확한 BTAD 설정이 확인됐다고 주장하지 않는다. test 기반으로 이를 바꾸지 않는다.

공식 VolumeNorm은 running_mean에 sample_mean의 autograd graph를 계속 연결한다. training 출력에는 running_mean을 사용하지 않아, 통계 갱신만 no_grad·detach로 수행한다. 수정 전후 출력·입력 gradient·통계값의 exact equality를 독립 검증하고 보존한다. 학습 출력과 손실을 바꾸지 않는 메모리 수정이다.

공식 demo는 매 epoch train→normal template→test를 수행하고 best-test Pixel AUROC/PRO 모델을 저장한다. 이번 실행은 학습15epoch 후 최종 template·test를 1회 수행한다. 따라서 test 기반 모델 선택을 제거했고, RNG 소비·worker·cuDNN·optimizer 구현도 달라져 원본과 bitwise 학습 동등성은 주장하지 않는다. 이것은 정확한 demo 전체 재생이 아닌 변경점을 명시한 로컬 실행이다. 최종 checkpoint와 optimizer, seed, 모든 epoch loss/step 수를 보존한다. 15epoch 완료 모델·15행 history·GPU 검증이 모두 남은 경우 평가만 재개한다. 학습 중간 epoch 재개는 구현하지 않는다.

최초 02 평가에서 이미지 라벨과 mask 최대값이 다른 사례를 발견했다. 원본을 추가 검사한 결과 02/test/ko/0145.png의 제공된 ground-truth가 원래부터 모두 0이었다(original_missing_mask_diagnosis.json). 축소로 결함이 소실됐다고 처음 추정했으나 원본 검사로 이를 정정한다. image label을 축소 mask 최대값에서 추론하는 공식 demo 방식 대신 원본 이미지 라벨을 유지한다. pixel mask는 제공된 원본과 변환을 그대로 보존하며 임의로 영역을 채우지 않는다. image-label/mask 불일치 목록을 mask_resize_audit.json에 기록한다. 01·02의 완료 checkpoint를 재사용해 불필요한 재학습 없이 평가를 다시 수행하며, 이전 환경·오류 log·resume의 checkpoint SHA도 보존한다. 빈 원본 mask는 pixel 평가의 데이터 한계다.

원문과 supplement의 pyramid downsample/upsample 설명이 일부 다르지만 이 실행은 공식 model.py의 max-pool/down 및 nearest/up·Gaussian kernel 구현을 그대로 따른다. 최신 PyTorch/torchvision과 보존된 autoFlow 의존성을 사용하므로 저자 당시 환경 일치도 주장하지 않는다.

## 검증·산출물

첫 실제 BTAD 학습 batch의 FP32/TF32 loss 및 gradient를 비교한다. loss 상대차이≤0.005, gradient 상대L2차이≤0.02일 때만 TF32를 켜고, 아니면 FP32로 진행한다. 모든 train/test/mask·raw·checkpoint SHA, 15epoch 및 batch 수, 최종 image max 및 Image/Pixel AUROC를 독립 대조한다. 첫 test 이미지의 latent/template를 별도로 저장하고 SciPy separable convolution으로 pyramid map을 독립 재구성한다. 전체 test forward를 이중 실행하거나 논문 수치 동등성을 검증하는 절차는 아니다.

- [준비](../source/prepare_pyramidflow.sh), [실행](../source/run_pyramidflow.sh)
- [학습·평가](../source/run_pyramidflow.py), [검증](../source/finish_pyramidflow.py)
- [결과 폴더](../source/results/btad_res18_seed0_20261008/)
- Raw·checkpoint·template: `/home/test/pyramidflow_results/btad_res18_seed0_20261008`
- Log: `/home/test/pyramidflow_results/run_20261008.log`

검증 passed 이후에만 MuSc 비교표에 자체 측정치를 반영한다. 원문 PDF는 paper 폴더에 보존하고 supplement는 참고자료로 related_work에 둔다. 연구 원본을 Obsidian에서 함께 참조하고 dataset·weight·환경을 복사하거나 색인하지 않는다.

## 자체 측정 결과

제품 3개 비가중 평균, 단위 %. 마지막 epoch15의 모델이며 best-test 선택은 하지 않았다.

| Product | Test images | Image AUROC | Pixel AUROC |
|---|---|---|---|
| 01 | 70 | 100.0000 | 95.2966 |
| 02 | 230 | 82.6167 | 97.2557 |
| 03 | 441 | 91.0488 | 94.4270 |
| Mean | 741 | 91.2218 | 95.6597 |

[검증](../source/results/btad_res18_seed0_20261008/verification.json) · [CSV](../source/results/btad_res18_seed0_20261008/category_metrics.csv)
