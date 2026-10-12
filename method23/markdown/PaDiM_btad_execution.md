# PaDiM BTAD full-shot 실행

## Paper Metadata

| Item | Content |
|---|---|
| Title | PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization |
| Authors | Thomas Defard, Aleksandr Setkov, Angelique Loesch, Romaric Audigier |
| Conference / Year | ICPR Workshops 2020, proceedings 2021 |
| Paper | [출판 정보](https://link.springer.com/chapter/10.1007/978-3-030-68799-1_35); 출판PDF 다운로드가 HTML을 반환하여 [arXiv 원문](https://arxiv.org/abs/2011.08785)을 보존 |
| Code | 저자 공식 구현 확인되지 않음. [공개 비공식 재현](https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master), commit `616004b9b7fa6a507d196dd3eeeba9e0a93f160f` |
| Purpose | MuSc Table14에 필요한 BTAD full-shot Image AUROC 및 Pixel AUROC 자체 측정 |

정상train 전체에서 patch별 정규분포를 추정한다. CNN은 ImageNet 고정 모델이며 optimizer 학습은 없다. 기존 method9 공통 프레임워크의 MVTec 결과와 별도로, PaDiM 논문 재현을 위한 자기완결적인 실험과 primary paper를 method23에 보존한다.

## 고정 조건과 차이

- BTAD 제품01/02/03, 정상train400/399/1000, test70/230/441(총741). 원본 metadata label과empty GT mask를 그대로 보존.
- WideResNet50×2 ImageNetV1, layer1/2/3 concatenate1792에서 PythonRandom1024로550채널 선택. 같은채널을 제품 전체에 사용. 공개 코드 기본seed1024 고정.
- 이미지256×256 bicubic·center crop224·ImageNet 정규화. mask는nearest·같은crop·>128 binary. cropped224 좌표에서 평가한다. 공통 framework의 resize/후처리로 대체하지 않는다.
- 위치별 평균·표본공분산(ddof1)+0.01I. FP64 covariance 계산 후 공개 코드처럼 FP32 저장; GPU FP32 inverse와 Mahalanobis sqrt. 56→224 bicubic, Gaussian sigma4, image score=최종map 최대값.
- RTX5080, features/distance FP32, AMP/TF32 미사용, batch16/workers4, cuDNN benchmark. GPU에서 공간32개씩 covariance/inverse/추론을 처리하되 전체train 유지.
- 공개 loader의 LANCZOS 및 bilinear map을 논문 본문의 bicubic으로 맞췄다. MuSc는 PaDiM의 정확한backbone/seed/resize를 공개하지 않으므로 설정과수치 동등성을 주장하지 않는다.

## 검증·보존

실제feature concatenate는 고정 public 함수의unfold/fold 출력과 exact 대조한다. 3개공간위치의 공분산/precision을 CPU NumPy로 독립계산하고 실제test9개patch의 Mahalanobis를 CPU FP64로 검증한다. 전체map 후처리,mask,image score,AUROC를 저장npz에서재계산한다. 입력/마스크·가중치·코드·Gaussian모델·원시예측 SHA를 보존한다. 검증 범위는 표본 covariance/거리와 전체후처리/지표이며 모든CNN forward의 이중실행이 아니다.

최종model은 mean.npy·precision.npy·channel_indices.npy로 저장한다(제품당precision 약3.8GB). feature는fit/검증에 사용하는 임시파일이며 완료후 삭제한다. 저장model과 dataset으로 예측 재생성이 가능하다. 원시prediction과검증sample은 유지한다.

- sh: [run_padim_btad.sh](../source/run_padim_btad.sh)
- source: [run_padim_btad.py](../source/run_padim_btad.py)
- result: [결과 폴더](../source/results/btad_wr50_d550_seed1024_20261010/)
- raw: `/home/test/padim_results/btad_wr50_d550_seed1024_20261010`
- log: `/home/test/padim_results/run_btad_20261010.log`

제품별결과와3개비가중평균을 저장한다. 최종검증통과후 MuSc Table14에 자동반영한다. 논문성능수치를 결과값으로 복사하지 않는다.

## 자체 측정 결과

제품3개 비가중 평균, 단위 %.

| Product | Image AUROC | Pixel AUROC |
|---|---:|---:|
| 01 | 99.1254 | 97.4993 |
| 02 | 85.8167 | 95.7182 |
| 03 | 99.4390 | 99.5838 |
| Mean | 94.7937 | 97.6004 |
