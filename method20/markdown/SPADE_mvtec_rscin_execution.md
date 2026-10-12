# SPADE MVTec full-shot와 RsCIN 전후 평가

MuSc Table11에 필요한 MVTec AD 15개 category·1,725장 test의 Image AUROC, F1-max, AP를 이 PC에서 계산한다. 정상 train 전체로 특징 갤러리를 구축하며 gradient 학습은 없다. 이 표가 요구하지 않는 pixel map 생성은 이번 실행에 포함하지 않는다.

## 논문·구현 근거와 조건

- Primary paper: [Sub-Image Anomaly Detection with Deep Pyramid Correspondences](https://arxiv.org/abs/2005.02357), Niv Cohen/Yedid Hoshen. method20의 기존 primary PDF 재사용.
- 공개 비공식 구현: [byungjae89/SPADE-pytorch](https://github.com/byungjae89/SPADE-pytorch), 고정 commit `077c67be21d68a38b4442db7311c87e708728286`. 저자 공식 구현 실행으로 표기하지 않는다.
- 기존 [BTAD 실행](SPADE_btad_execution.md)의 WideResNet50×2 ImageNetV1, shorter-side256 LANCZOS·center-crop224·ImageNet 정규화, avgpool2048를 유지한다.
- Image score는 논문의 K=50 squared L2 평균이다. 공개 비공식 구현의 기본 K=5/unsquared와 다르며 원문 방식으로 구현한다. coreset·근사검색·test tuning 없음.
- RTX5080, FP32, AMP/TF32 미사용, cuDNN benchmark, batch32/workers4. 고정 모델에서 학습정밀도 변경 없이 특징을 추출한다.
- RsCIN은 공식 MuSc `Mobile_RsCIN`, ViT-L-14-336 518 입력 projected/normalized class feature. PatchCore에서 보존한 feature를 경로·라벨·이미지 SHA 일치 검증 후 재사용한다.
- RsCIN 창 [1,2,3]은 앞선 비교 실행과 동일한 공식 코드 기본값이다. 원문 본문 MVTec [2,3]과 차이가 있으므로 원문 수치 동등성을 주장하지 않는다.

## 검증·저장

전체 test의 K50 이웃과 image score를 CPU FP64 직접차이로 독립 계산해 GPU FP32 결과와 비교한다(rtol1e-5/atol1e-4). 입력 SHA·학습/test 분리·1,725개 라벨을 확인한다. 저장된 npz에서 지표를 재계산하고 RsCIN을 다시 적용해 결과의 정확한 일치를 확인한다. 15개 category 비가중 평균을 사용한다.

- sh: [run_spade_mvtec_rscin.sh](../source/run_spade_mvtec_rscin.sh)
- source: [run_spade_mvtec_rscin.py](../source/run_spade_mvtec_rscin.py)
- result: [결과 폴더](../source/results/mvtec_k50_seed42_20261010/)
- raw: `/home/test/spade_results/mvtec_k50_seed42_20261010`의 train/test avgpool.npy 및 predictions.npz
- log: `/home/test/spade_results/run_mvtec_rscin_20261010.log`

최종 검증 통과 후에만 MuSc 비교표에 수치를 반영한다. category별 이미지 manifest·전후점수 CSV·metric CSV·검증 JSON 및 원시 결과를 보존한다. 기존 BTAD 자료는 유지한다.

## 자체 측정 결과

15 category 비가중 평균, 단위 %.

| RsCIN | Image AUROC | F1-max | AP |
|---|---:|---:|---:|
| w/o | 80.4887 | 88.9839 | 90.6749 |
| w | 81.8956 | 90.6157 | 91.0754 |
