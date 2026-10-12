# DSR MVTec 공개 가중치 자체 평가와 RsCIN 비교

## Paper Metadata

| Item | Content |
|---|---|
| Title | DSR – A dual subspace re-projection network for surface anomaly detection |
| Authors | Vitjan Zavrtanik, Matej Kristan, Danijel Skočaj |
| Conference / Journal | ECCV |
| Year | 2022 |
| Paper link | https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136910526.pdf |
| GitHub / Official code | https://github.com/VitjanZ/DSR_anomaly_detection |
| Reason for investigation | MuSc Table11 DSR MVTec full-shot RsCIN 적용 전후 Image AUROC/F1-max/AP를 이 PC에서 측정 |

## 질문·가설

질문: 같은 DSR 기본 이미지 점수에서 RsCIN 적용이 image AUROC/F1-max/AP를 어떻게 바꾸는가. 가설: CLS 특징이 유사한 이미지 간 smoothing이 일부 불안정 점수를 완화할 수 있으나 유사한 정상·이상이 섞인 category에서는 악화될 수 있다. 평균 및 category별 결과로 판단한다.

## 실행·설정

- commit: `672bfb81434fd2a6c5ef00db858cef8834c54f28`
- sh: `method25/source/run_dsr_mvtec_rscin.sh`
- result: `method25/source/results/mvtec_public_20261011/`
- raw/log: `/home/test/dsr_results/`
- 공개 가중치: 저자 README 링크의 archive `15plhikrUjYCcx23JVxxBKb-HBwKAb8UK`, `/home/test/dsr_assets/checkpoints`. archive/각 checkpoint/PDF SHA는 `source/asset_provenance.json`에 기록한다.

MVTec15개/1,725장 test를 평가한다. 공개 discrete latent model과 category별 두 subspace restriction·image reconstruction·anomaly detection·upsampling 모델을 로드한다. 공개 학습 가중치를 자체 평가하며 추가 학습은 수행하지 않는다. 원래 학습 로그·checkpoint 선정 과정은 독립 검증할 수 없다. 논문 성능 숫자를 결과로 복사하지 않는다.

공식 loader와 동일한 cv2 BGR/INTER_LINEAR 256×256 resize, /255, CHW FP32를 사용한다. ImageNet 정규화는 적용하지 않는다. Batch1/FP32/no AMP/no TF32, cuDNN benchmark와 loader worker4를 사용한다. 최신 NumPy와 불필요한 imgaug training import의 호환 문제를 피하려고 runtime loader copy에서 사용하지 않는 imgaug/perlin import 두 줄만 제거한다. upstream 원본은 수정하지 않고 SHA를 기록한다.

공식 evaluate_model의 image scoring AST를 그대로 추출한다. anomaly detector softmax의 anomaly 확률에21×21 avg_pool2d(stride1,padding10)를 적용한 최대값이 image score다. 원본에서는 pixel-only upsampling/crop 결과로 image score를 계산하지 않는다. 대량 image 평가에서는 해당 pixel-only branch만 생략하며 category 첫 실제 이미지에서는 완전한 공식 evaluator를 실행해 image score를 대조한다. 최종 upsampled pixel map 지표는 이번 요청에 포함하지 않는다.

RsCIN은 기존 ViT-L14-336/518 projected L2-normalized CLS를 이미지 경로·라벨·SHA 대조 후 재사용한다. 공식 MVTec k=[2,3], MMO의1 추가로 실제 평균은[2,3,1]이다. 저장된 기본 score에서 exact RsCIN 재계산과 별도 NumPy smoothing을 검증한다. saved probability map에서21pool/max를 재계산하고 독립 CPU FP64 uniform filter로 모든 image score를 확인한다. 원시 확률map·label·score 전후 및 category별 metrics를 보존한다. 15category 비가중 평균으로 표를 채운다.

공개 image reconstruction checkpoint에는 현재 모델에서 사용하지 않는 `vq._ema_w`, `vq._ema_cluster_size`, `vq._embedding.weight`가 포함된다. 공식 evaluator는 이 모델만 strict=False로 로드한다. 첫 준비 검증에서 이 키를 발견했고, 동일한 공식 로드 동작을 유지하면서 허용 키를 위 세 개로 제한하고 missing key는 금지했다. category 검증 JSON에 무시된 키를 보존한다. 나머지 모델은 strict load를 유지한다.

cuDNN benchmark를 사용하는 FP32 반복 forward는 bitwise 동일성을 요구하지 않는다. 첫 capsule native 대조에서1.583×10^-7의 절대 score 차이를 관찰했고, 선택 이미지 완전한 evaluator 대조에는 rtol1e-5/atol1e-6을 적용했다. 공식 점수 알고리즘·가중치는 변경하지 않았다. 저장 map 재계산 및 독립 CPU pooling 검증도 모든 이미지에 수행한다.

## 자체 측정 결과

15 category 비가중 평균, 단위 %.

| RsCIN | Image AUROC | F1-max | AP |
|---|---:|---:|---:|
| w/o | 98.2495 | 96.6198 | 99.1055 |
| w | 98.0835 | 96.8560 | 99.2139 |

15개 category/1,725장 평가 및 검증 완료. RsCIN 후 평균 AUROC는 -0.1660%p, F1-max는 +0.2363%p, AP는 +0.1083%p로 지표마다 변화 방향이 다르다. capsule·metal_nut·pill은 AUROC가 감소했다. 따라서 이 실행만으로 RsCIN이 모든 지표·category를 개선했다고 주장할 수 없다. `rscin_category_metrics.csv`와 raw prediction에 전후 결과를 모두 보존했고 MuSc Table11에 검증된 자체 값만 반영했다. 공개 가중치 자체 평가이며 추가 학습 결과가 아니다.
