# NSA MVTec AD 자체 학습·평가

| Item | Content |
|---|---|
| Title | Natural Synthetic Anomalies for Self-Supervised Anomaly Detection and Localization |
| Authors | Hannah M. Schlüter, Jeremy Tan, Benjamin Hou, Bernhard Kainz |
| Conference / Journal | ECCV |
| Year | 2022; arXiv 초판 2021 |
| Paper link | [원문](https://arxiv.org/abs/2109.15222) |
| GitHub / Official code | [hmsch/natural-synthetic-anomalies](https://github.com/hmsch/natural-synthetic-anomalies) |
| Reason for investigation | MuSc Table 2 NSA의 Image AUROC·Pixel AUROC를 PC 실측값으로 확보 |

질문: 공식 NSA logistic 학습·평가를 RTX 5080에서 MVTec 15개 category 모두 실행할 수 있는가? 실행 전 가설은 원래 이미지 합성·네트워크·최적화·평가 규칙을 유지하면서 현대 의존성의 API 차이만 보완하면 유효한 전체 결과를 생성할 수 있다는 것이다. 결과가 생성되면 실행 가능성을 지지하며 저자 성능의 수치 동등성을 직접 입증하는 것은 아니다.

공식 README에서 공개 평가 가중치 배포 링크를 확인하지 못했으므로 직접 학습한다. Object는 `Shift-Intensity-923874273`, texture는 mixed gradients를 쓰는 `Shift-Intensity-M-923874273`이다. Seed 923874273은 공식 예시 설정에 포함된 seed다. 이번 결과는 단일 seed 실행이며 원문 반복 평균과 구분한다.

| 항목 | 실행 조건 |
|---|---|
| commit | `919591685307ce030fe27cb77687509dc277189c` |
| 학습 | 정상 이미지 전체, category별 320 epoch; hazelnut·metal_nut·screw 560 epoch |
| batch | 64; GPU forward/backward 사전 확인 통과 |
| workers | 8; OpenCV thread 1 |
| 네트워크 | 공식 resnet18_enc_dec, pool=True, preact=False, sigmoid |
| optimizer | Adam 0.001, cosine scheduling, eta_min 1e-6 |
| 정밀도 | FP32, AMP 없음, TF32 비활성 |
| 평가 | 공식 notebook의 test_real_anomalies, batch16, input256 |
| Object 평가 | center crop224 추론 후 zero pad16; 이미지 점수는 pad 전 평균 |
| Texture 평가 | 256 전체 추론, 이미지 점수 평균 |
| 지표 | 공식 Image AUROC/AP, Pixel AUROC/AP, 공식 AUPRO; F1-max 추가 계산 |

Pillow `ANTIALIAS`는 원래 의미인 LANCZOS로 alias를 복원하고, 시각화용 torchvision `make_grid`의 `range` 인자를 `value_range`로 연결한다. 공식 추적 파일은 수정하지 않는다. 진행률 표시와 ETA 계산은 비활성화했다. 학습 loss와 복구용 checkpoint는 파일로 보존하며 사용자에게 반복 출력하지 않는다. 80 epoch마다 저장하는 복구 checkpoint는 연구 자료 보존용이며 자동 중간 epoch 재개는 구현하지 않았다. 완료된 category는 재실행 시 최종 checkpoint를 재사용한다.

- commit: `919591685307ce030fe27cb77687509dc277189c`
- sh: [run_nsa.sh](../source/run_nsa.sh)
- result: [실행 상태·환경](../source/results/logistic_seed923874273_20261007/environment.json). 완료 전 수치는 비교표에 넣지 않는다.

```bash
bash /mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method13/source/run_nsa.sh
```

가중치·복구 checkpoint·원시 예측은 `/home/test/nsa_results/logistic_seed923874273_20261007`, 실행 log는 `/home/test/nsa_results/run_20261007.log`에 둔다. 기존 MVTec 원본은 symlink로 참조하며 복사하지 않는다. 완료 후 [finish_nsa.py](../source/finish_nsa.py)가 원시 지표를 검증하고 이 문서 및 MuSc Table 2를 자동 갱신한다. Obsidian 탐색 목록도 완료 후 갱신한다.

## 자체 측정 결과

15개 category, 1,725장 전체 학습·평가 완료. 단위 %. Mean은 category 비가중 평균. 단일 seed로 반복 표준편차를 보고하지 않는다.

| Category | Images | image_auroc | image_f1_max | image_ap | pixel_auroc | pixel_f1_max | pixel_ap | aupro |
|---|---|---|---|---|---|---|---|---|
| bottle | 83 | 97.1429 | 98.4127 | 98.7925 | 98.3714 | 78.5165 | 82.9568 | 93.2496 |
| cable | 150 | 94.5840 | 93.8776 | 93.1760 | 95.1605 | 43.3805 | 38.0885 | 88.3496 |
| capsule | 132 | 93.6179 | 93.7198 | 98.7259 | 97.8239 | 56.9278 | 57.9360 | 92.8738 |
| carpet | 117 | 96.8299 | 96.0452 | 99.1352 | 95.6761 | 49.3732 | 52.9808 | 86.6554 |
| grid | 78 | 98.4127 | 99.1150 | 99.5614 | 98.9377 | 53.9401 | 51.8262 | 95.9762 |
| hazelnut | 110 | 96.6429 | 93.6170 | 98.2247 | 97.1476 | 58.1229 | 53.0871 | 93.6500 |
| leather | 124 | 99.8981 | 99.4595 | 99.9646 | 99.5218 | 59.9511 | 56.0240 | 98.5588 |
| metal_nut | 115 | 99.1202 | 97.8723 | 99.7946 | 98.0821 | 86.0425 | 92.7670 | 93.9896 |
| pill | 167 | 99.1271 | 98.2332 | 99.8380 | 98.6420 | 72.5995 | 78.2792 | 94.4109 |
| screw | 160 | 88.0098 | 87.6404 | 96.1779 | 95.7496 | 46.7004 | 45.0220 | 88.1337 |
| tile | 117 | 99.8918 | 98.8235 | 99.9581 | 98.9641 | 84.3843 | 92.8293 | 93.7908 |
| toothbrush | 42 | 100.0000 | 100.0000 | 100.0000 | 94.9747 | 49.9224 | 42.6930 | 86.9058 |
| transistor | 100 | 94.0000 | 88.0952 | 93.2027 | 85.7832 | 52.9859 | 48.1923 | 71.6908 |
| wood | 79 | 95.9649 | 95.2381 | 98.5693 | 89.2020 | 51.8336 | 50.9421 | 84.1172 |
| zipper | 151 | 99.8162 | 99.5781 | 99.9533 | 93.9550 | 62.1311 | 66.9176 | 88.3610 |
| macro_mean | 1725 | 96.8705 | 95.9818 | 98.3383 | 95.8661 | 60.4541 | 60.7028 | 90.0475 |

[환경](../source/results/logistic_seed923874273_20261007/environment.json) · [검증](../source/results/logistic_seed923874273_20261007/verification.json) · [카테고리 CSV](../source/results/logistic_seed923874273_20261007/category_metrics.csv)

공식 평가 함수의 AUROC/AP 4개 지표를 보존된 원시 NPZ에서 다시 계산해 1e-12 이내 일치를 검증했다. 공식 NSA AUPRO는 FPR 0.3에서 보간·정규화하는 구현이며, 다른 방법에서 사용한 200 threshold 근사 AUPRO와 구현 차이가 있다. 학습 seed, worker 수 및 현대 의존성은 이 PC의 실행 조건으로 기록하며, 저자 수치 동등성을 주장하지 않는다.

## 실행 후 자료 정리

2026-10-07 사용자 요청으로 공식 체크아웃·공개 DRAEM 가중치·중복 및 벤치마크 임시 자료를 삭제했다. 완료된 결과·원시 예측·검증 기록과 직접 학습한 가중치는 보존했다. 재실행에는 삭제된 공식 체크아웃 등을 다시 준비해야 한다. [삭제·보존 내역](../../related_work/markdown/MuSc_comparators_cleanup_20261007.md)을 참조한다.
