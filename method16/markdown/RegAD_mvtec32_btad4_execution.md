# RegAD MVTec32-shot·BTAD4-shot 실행

MuSc Table2와Table14의 두 조건을 이 PC에서 평가한다. Primary paper와공식코드는 기존 [RegAD 실행 보고서](RegAD_mvtec_execution.md)를 공유하며 method16에 이어서 보존한다.

## 학습과평가 구분

[RegAD 부록Large k](https://arxiv.org/html/2207.07361v1#S0.SS2.SSS1)는 support수를 늘릴 때 재학습/finetuning하지 않는 확장을 설명한다. 앞선 보고서의 '32-shot은 반드시 별도32-shot 학습모델 필요'는 과도한 제한이었다. 이번 실행은 **공식학습된 공개checkpoint를 사용한 자체평가**, 추가로이PC에서gradient학습한결과가 아니다.

- MVTec32: 공식4-shot의category별checkpoint(해당target제외학습)를그대로사용, target정상train에서32개reference 선택. 15category·1725장 test·10support round.
- BTAD4: MVTec bottle4 checkpoint를test결과를보기전에고정선택, BTAD01/02/03 정상train에서제품별4개reference. BTAD학습/모델선택없음. 3제품·741장·10round 전이평가.
- MuSc는해당32-shotcheckpoint 및BTAD의학습source/선택조건을공개하지않았다. MVTec4→32확장과BTAD고정source선택은명시적local조건이며저자원실험과동일한학습이력/수치를보장하지않는다.

## 실행조건·가속

공식 commit `5e2c1f8c18d302b0354471567846fee3ed2ff063`, ResNet18/STN rotation_scale. 원본checkout은수정하지않으며별도runtime에서Pillow LANCZOS호환만변경한다. 이미지224×224LANCZOS·ToTensor, ImageNet정규화없음,masknearest>128. 종횡비가다른BTAD는정사각형resize로shape를고정하며이차이를기록한다.

referenceaugmentation은공식22변환(원본1+회전8+이동8+flip1+gray1+90도회전3). 각round seed668+round,비복원선택한원본support수는32/4이다. augmentation으로원본shot수를부풀리지않는다.

support eval forward를16장chunk로처리하여32×22=704장전체를한번에넣을때의activation메모리를줄인다. 모든support를유지하며eval BatchNorm을사용한다. query는공식batch1. 3층원래featureconcat은고정native함수와exact대조한다. 사용되지않는encoder/predictor cosine진단loss계산은 생략하며native최종Mahalanobis scoring에영향이없다.

FP32, convolution TF32·cuDNNbenchmark, matmulTF32미사용. covariance(mean/subtract,ddof1,+.01I)와inverse/Mahalanobis를공간16개씩GPUbatch로계산한다. 3개공간위치의실제covariance와9개query거리값을native torch.cov/scalar함수와대조한다. supportforward chunk와native거대batch가bitwise동일하다는주장은하지않는다.

56→224bilinear·Gaussian sigma4·category minmax·image mapmax. 원시npz에서전체후처리,mask,Image/PixelAUROC를재계산하고이미지/마스크/SHA·checkpoint/support선택manifest를보존한다. 10round category비가중평균의평균±표준편차(ddof0)를저장한다.

## 산출물

- sh: [run_regad_remaining.sh](../source/run_regad_remaining.sh)
- source: [run_regad_remaining.py](../source/run_regad_remaining.py)
- result: [결과 폴더](../source/results/remaining_mvtec32_btad4_20261010/)
- raw: `/home/test/regad_results/remaining_mvtec32_btad4_20261010`
- log: `/home/test/regad_results/run_remaining_20261010.log`

추론용support/queryfeature는임시memmap이며round검증후삭제한다. 원시map/거리/labels/score와support원본경로·SHA 및modelcheckpoint는유지한다. 기존2/4/8-shot결과는변경하지않는다. 180category-round 최종검증을통과한뒤비교표에자동반영한다.

## 자체 측정 결과

단위 %,10 support round macro평균±표준편차(ddof0).

| Dataset | Shot | Image AUROC | Pixel AUROC |
|---|---|---:|---:|
| mvtec | 32 | 94.5213 ± 0.2996 | 97.4228 ± 0.0460 |
| btad | 4 | 87.9947 ± 1.1049 | 92.3018 ± 1.5105 |
