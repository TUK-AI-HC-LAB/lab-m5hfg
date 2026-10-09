# RegAD 구현체 실행

## Paper Metadata

| Item | Content |
|---|---|
| Title | Registration Based Few-Shot Anomaly Detection |
| Authors | Chaoqin Huang, Haoyan Guan, Aofan Jiang, Ya Zhang, Michael Spratling, Yan-Feng Wang |
| Conference / Journal | ECCV |
| Year | 2022 |
| Paper link | [공식 출판 PDF](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840300.pdf) |
| GitHub / Official code | [MediaBrain-SJTU/RegAD](https://github.com/MediaBrain-SJTU/RegAD) |
| Reason for investigation | MuSc 비교표의 RegAD 결과를 이 PC에서 직접 측정 |

질문: 공식 공개 checkpoint·고정 support set과 공식 scoring으로 MVTec15개 category의 검증 가능한 few-shot 결과를 얻을 수 있는가? 실행 전 가설은 동일한 covariance·Mahalanobis 계산을 batch화하면 계산식을 유지하며 GPU 동기화 비용을 줄일 수 있다는 것이다. 제한된 실제 feature 대조와 보존한 raw 출력의 지표 재계산으로 검증한다.

## 실행 범위와 조건

사전 검사에서 공식 archive의 capsule/grid 2·4·8-shot support6개가0bytes임을 확인했다. 나머지39개 support 및45개 공개 checkpoint는 정상이다. 이2개 category에만 정상 train 이미지에서 seed668로 round당 shot개를 비복원 추출하고 공식 Resize224 LANCZOS·ToTensor를 적용해10round 고정 support를 생성했다. 원래 빈 파일·archive는 변경하지 않고 별도 local_support_set을 사용한다. [원본 자료 검사](../source/results/mvtec_public_20261007/public_assets_verification.json), [로컬 support 이미지·SHA·선택 기록](../source/results/mvtec_public_20261007/support_overrides.json)에 남긴다. 저자의 고정 support와 동일하다고 주장하지 않는다. 중단 후 완료 round를 재사용할 때 RNG가 다시 시작하므로 support augmentation의 permutation과 수치가 연속 실행과 bitwise 같다는 보장은 하지 않는다.

우선 공개된2/4/8-shot 각각 MVTec15개 category, 고정 support10round를 평가한다. 4-shot부터 수행한다. 학습을 다시 하지 않고 공식 checkpoint를 사용한다. 공개 자료에는32-shot model/support가 없어32-shot은 별도 학습 조건이 필요하며, BTAD4-shot은 transfer/training protocol 확인이 남아 있다. 이 조건들은 완료로 표시하지 않는다.

ResNet18 기반 STN, rotation_scale, 원래 Encoder·Predictor를 사용한다. 공개 고정 support에서 소회전8종·평행이동8종·horizontal flip·gray·90도 회전3종을 추가한다. query/support는 Resize224 LANCZOS와 ToTensor만 사용한다. ImageNet normalization은 공식 코드에서 주석 처리되어 있다. mask는NEAREST resize 후>0.5로 이진화한다. 위치별 covariance는원래torch.cov+0.01I, 세층 특징concat, Mahalanobis distance,224 bilinear upsample·Gaussian sigma4·category min-max를 사용한다. image score는맵최대값이다.

FP32, convolution TF32·cuDNN benchmark 활성화, matmul TF32 비활성화, torch.no_grad, query batch1, worker4를 사용한다. seed668은공식default를따르며 CPU/random/numpy도명시적으로고정한다. 10개의 support round변동과 모델 재학습seed변동을구분한다.

## 호환 변경 및 검증

원본 checkout은수정하지않는다. Pillow ANTIALIAS를같은LANCZOSenum으로교체한다. mask경로의원래전체문자열test치환은/home/test사용자명까지변경해실패하므로 category/test부분만치환한다. 모델을임의ImageNet가중치로다시받지않고동일architecture를생성해공개전체STN/ENC/PREDstate를strict load한다. 공식queryforward까지torch.no_grad를적용한다.

공식test의거리계산만16개spatial position씩batchinverse·matmul로묶는다. covariance·거리공식은유지하며 첫실제covariance/query일부를원래scalar루프와rtol/atol1e-3으로대조한다. 전체출력의bitwise동등성을의미하지않는다. 변경diff와SHA를보존한다.

Image/Pixel AUROC·AP·F1-max와200threshold AUPRO를각round에서계산한다. RsCIN은기존MuSc ViT-L-14-336공유class token을이미지SHA·라벨·경로로정렬해공식Mobile_RsCIN에입력한다. 원시NPZ와모든round점수·support/checkpointSHA를남긴다. 완료후Image지표전체round와Pixel지표각category의round0을독립재계산한다.

## 재현·근거

- commit: `5e2c1f8c18d302b0354471567846fee3ed2ff063`
- sh: [준비](../source/prepare_regad.sh), [실행](../source/run_regad.sh)
- result: [환경·상태](../source/results/mvtec_public_20261007/environment.json), [호환diff](../source/results/mvtec_public_20261007/compatibility.patch), [원본·runtimeSHA](../source/results/mvtec_public_20261007/runtime_sources.json)

실행명령은 `bash /mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method16/source/run_regad.sh`다. 원시맵·log는 `/home/test/regad_results`, 공개자료는 `/home/test/regad_assets`에둔다. 완료round는재사용하고미완료round만다시계산한다. 전체검증뒤MuSc Table1과Table11을자동갱신한다. Table13속도는별도단독benchmark가필요하므로현재평가전체walltime을추론시간으로대입하지않는다.

## 자체 측정 결과

공식 public checkpoint를 이 PC에서 평가했다. 학습은 다시 수행하지 않았다. 수치는 %, 평균은 category 비가중 평균이다. ±는10개 고정 support round별 macro metric의 표준편차(ddof0)이며 모델 재학습 seed의 편차가 아니다.

| Shot | image_auroc | image_f1_max | image_ap | pixel_auroc | pixel_f1_max | pixel_ap | aupro |
|---|---|---|---|---|---|---|---|
| 4 | 88.8213 ± 1.0163 | 92.2460 ± 0.3467 | 94.7066 ± 0.7681 | 96.1330 ± 0.2160 | 51.3182 ± 0.8179 | 47.8845 ± 0.9246 | 87.9181 ± 0.4449 |
| 2 | 85.3483 ± 0.9444 | 91.5156 ± 0.4033 | 92.6173 ± 0.5601 | 95.7334 ± 0.2142 | 49.9700 ± 0.7673 | 46.8633 ± 0.8481 | 86.3844 ± 0.5385 |
| 8 | 91.3723 ± 0.4621 | 93.3837 ± 0.2852 | 95.7999 ± 0.2973 | 97.1553 ± 0.1231 | 54.9382 ± 0.2908 | 51.2973 ± 0.3915 | 90.2199 ± 0.2385 |

[카테고리 CSV](../source/results/mvtec_public_20261007/category_metrics.csv) · [검증](../source/results/mvtec_public_20261007/verification.json)

450회 원시 맵/이미지 SHA·라벨·점수와 공개 checkpoint/support SHA를 확인했다. Image3개 지표는 전체 round, Pixel3개 지표는 각 category·shot의 round0에서 독립 재계산했다. AUPRO는200 threshold 근사 구현이고 별도 재계산하지 않았다. 제한된 실제 특징에서 batch Mahalanobis 대조를 통과했으나 전체 출력의 bitwise 일치를 주장하지 않는다. 32-shot과 BTAD는 이 완료 범위에 포함되지 않는다.
