# DRAEM 공식 구현체 MVTec AD 실행

| Item | Content |
|---|---|
| Title | DRAEM - A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection |
| Authors | Vitjan Zavrtanik, Matej Kristan, Danijel Skocaj |
| Conference / Journal | ICCV |
| Year | 2021 |
| Paper link | [공식 PDF](https://openaccess.thecvf.com/content/ICCV2021/papers/Zavrtanik_DRAEM_-_A_Discriminatively_Trained_Reconstruction_Embedding_for_Surface_Anomaly_ICCV_2021_paper.pdf) |
| GitHub / Official code | [VitjanZ/DRAEM](https://github.com/VitjanZ/DRAEM) |
| Reason for investigation | MuSc Table 11의 DRAEM 및 DRAEM+RsCIN 행을 이 PC의 측정값으로 작성 |

확인 질문: 공식 공개 DRAEM 체크포인트를 현재 RTX 5080 환경에서 MVTec AD 15개 카테고리 모두 평가하고, 같은 이미지의 점수에 공식 MuSc RsCIN을 적용할 수 있는가?

실행 전 가설: 공식 네트워크·평가 전처리·점수 정의를 보존하면 1,725장 전체의 유효한 점수와 카테고리별 지표가 생성된다. RsCIN의 이미지 간 일관성 보정은 이미지 AUROC를 개선할 수 있지만, 카테고리별 개선이나 전체 평균의 개선은 보장하지 않는다. 개선 여부는 자체 실행 전후 수치로 판단한다.

이번 작업은 공개 가중치의 로컬 평가다. 정상 학습 이미지를 이용한 카테고리별 재학습은 수행하지 않는다. 공개 파일명의 `800`은 실제 학습 이력을 검증한 epoch 수가 아니다. README 학습 예시의 700 epoch와 구분한다. 논문이나 README의 성능 수치는 결과 표에 대입하지 않는다.

## 실행 조건

| 항목 | 값 |
|---|---|
| 공식 commit | `2dbf67397ab5c10a1494e5ae70ab59a25d7c35ef` |
| GPU | NVIDIA GeForce RTX 5080 |
| 입력 | 256×256, OpenCV BGR, `/255` |
| 네트워크 | 공식 reconstructive + discriminative subnet, 카테고리별 두 가중치 |
| 정밀도 | FP32, AMP 없음, TF32 비활성 |
| batch | 1 |
| 이미지 점수 | 이상 softmax 채널에 21×21 average pooling 후 최댓값 |
| 픽셀 점수 | pooling 전 이상 softmax 채널 |
| mask | 공식 OpenCV 기본 linear resize → `/255` → 픽셀 지표에서 uint8 절삭 |
| 공식 지표 | 이미지 AUROC/AP, 픽셀 AUROC/AP |
| 추가 지표 | 이미지·픽셀 F1-max, APRIL-GAN 공식 함수의 200 threshold AUPRO(FPR<0.3) |
| RsCIN | 공식 MuSc Mobile_RsCIN, effective windows `[1,2,3]` |
| RsCIN feature | 이 PC에서 이미 추출한 공식 MuSc ViT-L-14-336 OpenAI class token, input 518 |

현재 NumPy 2 환경에서 학습 전용 `imgaug` import가 평가를 막으므로 AST로 공식 test dataset class만 그대로 불러온다. 해당 클래스의 이미지·마스크 처리 문장은 변경하지 않는다. 공식 `test()` 자체를 실행하며 line trace로 반올림 전 결과를 수집한다. 평가 메모리를 줄이기 위해 `torch.no_grad()`를 적용한다. 공식 저장소의 추적 파일은 수정하지 않는다. 최신 PyTorch/OpenCV 환경이므로 저자 환경과 bitwise 동일성을 주장하지 않는다.

- commit: `2dbf67397ab5c10a1494e5ae70ab59a25d7c35ef`
- sh: [run_draem.sh](../source/run_draem.sh), [다운로드](../source/prepare_draem.sh)
- result: [카테고리 CSV](../source/results/mvtec_public_20261007/category_metrics.csv), [환경·가중치 SHA-256](../source/results/mvtec_public_20261007/environment.json), [검증](../source/results/mvtec_public_20261007/verification.json)

실행:

```bash
bash /mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method12/source/run_draem.sh
/home/test/miniforge3/envs/patchcore-gpu/bin/python /mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method12/source/evaluate_rscin.py
```

원시 맵·마스크는 `/home/test/draem_results/mvtec_public_20261007/<category>/raw_predictions.npz`, 가중치는 `/home/test/draem_weights/DRAEM_checkpoints`에 보존한다. 대용량 파일은 연구 Git 저장소 밖에 둔다. CSV·환경·파일 SHA-256은 [결과 폴더](../source/results/mvtec_public_20261007)에 보존한다.

## 자체 측정 결과

MVTec AD 15개 category, test 1,725장 전체 완료. 단위는 %. Mean은 이미지 수로 가중하지 않은 카테고리별 지표의 단순 평균이다. 단일 공개 체크포인트 평가이므로 seed 반복 표준편차는 보고하지 않는다. F1-max는 평가 라벨로 threshold를 탐색한 값이다.

| Category | Images | Image AUROC | Image F1-max | Image AP | Pixel AUROC | Pixel F1-max | Pixel AP | AUPRO |
|---|---|---|---|---|---|---|---|---|
| capsule | 132 | 96.0909 | 95.8525 | 99.2265 | 94.0436 | 46.9147 | 42.7090 | 89.5780 |
| bottle | 83 | 96.9048 | 98.4375 | 98.5985 | 99.2689 | 83.9567 | 89.7101 | 96.8980 |
| carpet | 117 | 96.2681 | 95.0820 | 98.8270 | 96.2353 | 61.2116 | 63.7994 | 91.4759 |
| leather | 124 | 100.0000 | 100.0000 | 100.0000 | 98.8812 | 62.4742 | 69.0306 | 97.4722 |
| pill | 167 | 96.7540 | 97.2028 | 99.4038 | 97.6219 | 61.2348 | 45.9213 | 80.7718 |
| transistor | 100 | 94.1667 | 84.3373 | 93.5037 | 89.9719 | 53.5502 | 51.2825 | 77.6839 |
| tile | 117 | 100.0000 | 100.0000 | 100.0000 | 99.5005 | 91.4571 | 96.8538 | 98.8852 |
| cable | 150 | 93.4033 | 90.3226 | 95.7752 | 95.4296 | 60.0233 | 62.4131 | 73.5983 |
| zipper | 151 | 99.6849 | 98.7342 | 99.9170 | 98.6376 | 67.7513 | 71.9474 | 94.1570 |
| toothbrush | 42 | 99.7222 | 98.3607 | 99.8925 | 98.1218 | 49.3054 | 53.3440 | 90.2533 |
| metal_nut | 115 | 99.3646 | 99.4595 | 99.8681 | 98.7272 | 84.3549 | 91.4029 | 93.0705 |
| hazelnut | 110 | 100.0000 | 100.0000 | 100.0000 | 99.5470 | 80.4586 | 87.6923 | 96.8259 |
| screw | 160 | 99.2416 | 98.3193 | 99.7262 | 99.6807 | 64.8238 | 70.5493 | 97.2315 |
| grid | 78 | 100.0000 | 100.0000 | 100.0000 | 99.5586 | 58.2713 | 55.2671 | 97.6739 |
| wood | 79 | 99.2105 | 98.3333 | 99.7466 | 97.1669 | 74.9492 | 80.8367 | 93.4084 |
| macro_mean | 1725 | 98.0541 | 96.9628 | 98.9657 | 97.4928 | 66.7158 | 68.8506 | 91.2656 |

[반올림 전 전체 평균 CSV](../source/results/mvtec_public_20261007/mean_metrics.csv)

AUPRO는 DRAEM의 공식 test 출력에 없는 보조 지표다. APRIL-GAN 함수의 200 threshold, FPR<0.3 및 선택 FPR 범위를 min-max 정규화하는 구현을 적용했다. 원문 저자의 AUPRO와 동일하다고 단정하지 않는다.

## RsCIN 적용 전후

같은 DRAEM 이미지 점수에만 공식 Mobile_RsCIN을 적용했다. 분할 맵·픽셀 지표는 다시 생성하지 않는다. 공유 MuSc class token을 원래 DRAEM 이미지 순서로 정렬하고 이미지 경로·라벨·SHA-256을 모두 대조했다.

| Category | w/o AUROC | w AUROC | Δ AUROC (pp) | w/o F1-max | w F1-max | w/o AP | w AP |
|---|---|---|---|---|---|---|---|
| capsule | 96.0909 | 94.8544 | -1.2365 | 95.8525 | 94.6903 | 99.2265 | 98.9270 |
| bottle | 96.9048 | 99.5238 | 2.6190 | 98.4375 | 99.2126 | 98.5985 | 99.8450 |
| carpet | 96.2681 | 98.3146 | 2.0465 | 95.0820 | 96.7391 | 98.8270 | 99.4591 |
| leather | 100.0000 | 99.8302 | -0.1698 | 100.0000 | 99.4595 | 100.0000 | 99.9403 |
| pill | 96.7540 | 90.5619 | -6.1920 | 97.2028 | 95.2381 | 99.4038 | 98.0918 |
| transistor | 94.1667 | 94.9167 | 0.7500 | 84.3373 | 86.0465 | 93.5037 | 94.3066 |
| tile | 100.0000 | 100.0000 | 0.0000 | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| cable | 93.4033 | 94.0592 | 0.6559 | 90.3226 | 90.7104 | 95.7752 | 96.6133 |
| zipper | 99.6849 | 100.0000 | 0.3151 | 98.7342 | 100.0000 | 99.9170 | 100.0000 |
| toothbrush | 99.7222 | 99.7222 | 0.0000 | 98.3607 | 98.3607 | 99.8925 | 99.8925 |
| metal_nut | 99.3646 | 98.4848 | -0.8798 | 99.4595 | 98.3607 | 99.8681 | 99.6884 |
| hazelnut | 100.0000 | 100.0000 | 0.0000 | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| screw | 99.2416 | 98.1964 | -1.0453 | 98.3193 | 95.9350 | 99.7262 | 99.3738 |
| grid | 100.0000 | 100.0000 | 0.0000 | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| wood | 99.2105 | 100.0000 | 0.7895 | 98.3333 | 100.0000 | 99.7466 | 100.0000 |
| macro_mean | 98.0541 | 97.8976 | -0.1565 | 96.9628 | 96.9835 | 98.9657 | 99.0759 |

[RsCIN 카테고리 CSV](../source/results/mvtec_public_20261007/rscin/category_metrics.csv) · [평균 CSV](../source/results/mvtec_public_20261007/rscin/mean_metrics.csv) · [이미지·특징 정렬 검증](../source/results/mvtec_public_20261007/rscin/dataset_alignment.csv) · [RsCIN provenance](../source/results/mvtec_public_20261007/rscin/provenance.json)

## 결과 해석과 검증

전체 이미지 점수와 픽셀 맵이 유효하게 생성되어 실행 전 가설의 통합 동작 부분을 지지한다. RsCIN 적용 후 평균 Image AUROC는 -0.1565 percentage point 변했다. 이는 이 공개 가중치와 로컬 공유 특징 조건에서의 관찰이며 저자의 Table 11 수치 재현이나 카테고리별 개선 보장은 아니다.

원시 NPZ에서 공식 4개 지표를 다시 계산하고 실행 중 수집된 값과 절대 오차 1e-12 이내 일치를 확인했다. 원시 픽셀 맵에 공식 21×21 pooling을 다시 적용한 이미지 점수는 저장된 FP32 점수와 정확히 일치했다. 30개 가중치의 해시, 15개 category·1,725장 수, 현재 원본 이미지의 해시, 비가중 평균, RsCIN 특징의 라벨·해시 정렬을 검증했다. 보조 AUPRO와 F1-max의 저자 환경 수치 동등성은 검증하지 않았다.

공식 평가와 결과 수집·압축·보조 지표 계산을 포함한 wall time은 5.09분이다. 다운로드·검증·RsCIN·보고서 시간은 제외한다. 순수 추론 시간으로 해석하지 않는다. PyTorch peak allocated는 992.93 MiB다.

한계: 공개 가중치를 사용했으므로 현재 PC에서 DRAEM의 학습을 재현한 결과는 아니다. 학습 epoch·학습 seed·저자의 개별 train 이력은 가중치 파일명만으로 확인할 수 없다. 현대 PyTorch/OpenCV와 저자 의존성의 차이가 있고, MuSc Table 11에서 저자가 사용한 개별 DRAEM 가중치·class token 파일과의 동일성을 확인하지 않았다. 따라서 표의 상태를 조건 대조 필요로 유지한다.

[MuSc 전체 비교표 Table 11](../../method10/markdown/MuSc_all_reproduction_tables.md#table-11-비교군에-rscin을-적용한-classification-결과)

## 실행 후 자료 정리

2026-10-07 사용자 요청으로 공식 체크아웃·공개 DRAEM 가중치·중복 및 벤치마크 임시 자료를 삭제했다. 완료된 결과·원시 예측·검증 기록과 직접 학습한 가중치는 보존했다. 재실행에는 삭제된 공식 체크아웃 등을 다시 준비해야 한다. [삭제·보존 내역](../../related_work/markdown/MuSc_comparators_cleanup_20261007.md)을 참조한다.
