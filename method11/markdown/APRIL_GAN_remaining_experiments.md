# APRIL-GAN — MuSc 비교 실험 전체 진행과 결과

작성일: 2026-10-06. 기본 L14 12개 실행은 공개 checkpoint 평가다. Table 13의 작은 backbone은 별도 보조 데이터 학습·평가 결과로 구분한다. 논문 성능 숫자는 결과 셀에 복사하지 않는다.

4-shot 반복 seed는 42·43·44다. 원문에서 정확한 반복 seed 목록을 확인하지 못해 자체 재현 설정으로 지정했다. 표준편차는 3회 macro metric의 표본 표준편차(ddof=1)다. 3회 미완료 상태의 평균은 진행 중인 값이며 최종 반복 결과가 아니다.

## 실행 상태

| Dataset | Mode | Seed | 상태 | 추론 완료 장수 | 완료 category | 근거 |
|---|---|---|---|---|---|---|
| mvtec | zero_shot | 42 | completed | 1725 | 15 | [환경](../source/results/mvtec_zero_shot_20261006/environment.json) |
| mvtec | few_shot | 42 | completed | 1725 | 15 | [환경](../source/results/mvtec_4shot_seed42_20261006/environment.json) |
| mvtec | few_shot | 43 | completed | 1725 | 15 | [환경](../source/results/mvtec_4shot_seed43_20261006/environment.json) |
| mvtec | few_shot | 44 | completed | 1725 | 15 | [환경](../source/results/mvtec_4shot_seed44_20261006/environment.json) |
| visa | zero_shot | 42 | completed | 2162 | 12 | [환경](../source/results/visa_0shot_seed42_20261006/environment.json) |
| visa | few_shot | 42 | completed | 2162 | 12 | [환경](../source/results/visa_4shot_seed42_20261006/environment.json) |
| visa | few_shot | 43 | completed | 2162 | 12 | [환경](../source/results/visa_4shot_seed43_20261006/environment.json) |
| visa | few_shot | 44 | completed | 2162 | 12 | [환경](../source/results/visa_4shot_seed44_20261006/environment.json) |
| btad | zero_shot | 42 | completed | 741 | 3 | [환경](../source/results/btad_0shot_seed42_20261006/environment.json) |
| btad | few_shot | 42 | completed | 741 | 3 | [환경](../source/results/btad_4shot_seed42_20261006/environment.json) |
| btad | few_shot | 43 | completed | 741 | 3 | [환경](../source/results/btad_4shot_seed43_20261006/environment.json) |
| btad | few_shot | 44 | completed | 741 | 3 | [환경](../source/results/btad_4shot_seed44_20261006/environment.json) |

## 성능 집계

| Dataset | Setting | n_runs | Image AUROC | Image F1-max | Image AP | Pixel AUROC | Pixel F1-max | Pixel AP | AUPRO | 상태 |
|---|---|---|---|---|---|---|---|---|---|---|
| mvtec | 0-shot | 1 | 86.1250 | 90.3505 | 93.5605 | 87.6288 | 43.2758 | 40.7849 | 44.0371 | complete·원문 조건 대조 필요 |
| mvtec | 4-shot | 3 | 92.6505 ± 0.0724 | 92.7513 ± 0.0713 | 96.2305 ± 0.0077 | 95.9100 ± 0.0334 | 56.6658 ± 0.4731 | 54.4308 ± 0.2771 | 91.9175 ± 0.0340 | complete·원문 조건 대조 필요 |
| visa | 0-shot | 1 | 77.5213 | 78.5613 | 80.9004 | 94.1977 | 32.2806 | 25.7560 | 86.5940 | complete·원문 조건 대조 필요 |
| visa | 4-shot | 3 | 92.5687 ± 0.1498 | 88.2393 ± 0.2682 | 94.4684 ± 0.0689 | 96.1837 ± 0.0194 | 39.6256 ± 0.3260 | 32.0048 ± 0.1952 | 90.0917 ± 0.1452 | complete·원문 조건 대조 필요 |
| btad | 0-shot | 1 | 73.7895 | 68.7930 | 69.8567 | 91.4034 | 37.4302 | 32.4077 | 21.9031 | complete·BTAD 조건 미확정 |
| btad | 4-shot | 3 | 91.9072 ± 0.1170 | 90.7082 ± 0.6579 | 93.2812 ± 0.1939 | 96.2023 ± 0.0943 | 50.2043 ± 0.1172 | 49.4509 ± 0.2676 | 80.9603 ± 0.6314 | complete·BTAD 조건 미확정 |

## RsCIN 적용 전후 — Table 11

| Dataset | Setting | RsCIN | n_runs | Image AUROC | Image F1-max | Image AP |
|---|---|---|---|---|---|---|
| mvtec | 0-shot | w/o | 1 | 86.1250 | 90.3505 | 93.5605 |
| mvtec | 0-shot | w | 1 | 86.1692 | 90.8962 | 93.6730 |
| mvtec | 4-shot | w/o | 3 | 92.6505 ± 0.0724 | 92.7513 ± 0.0713 | 96.2305 ± 0.0077 |
| mvtec | 4-shot | w | 3 | 93.3863 ± 0.0492 | 93.3749 ± 0.2192 | 96.7462 ± 0.0207 |
| visa | 0-shot | w/o | 1 | 77.5213 | 78.5613 | 80.9004 |
| visa | 0-shot | w | 1 | 78.9014 | 80.0743 | 82.0228 |
| visa | 4-shot | w/o | 3 | 92.5687 ± 0.1498 | 88.2393 ± 0.2682 | 94.4684 ± 0.0689 |
| visa | 4-shot | w | 3 | 94.5818 ± 0.1753 | 90.4776 ± 0.2696 | 95.7089 ± 0.1140 |

## 단독 속도·메모리 측정과 작은 backbone 학습

| Backbone | 상태 | ms/image | GPU allocated (MiB) | GPU reserved (MiB) | 근거 |
|---|---|---|---|---|---|
| ViT-L-14-336 | completed | 50.8219 | 2490.3623 | 2904.0000 | [계측](../source/results/benchmark_ViT-L-14-336_20261006/benchmark.json) |
| ViT-B-16-plus-240 | completed | 17.4080 | 1488.1372 | 1704.0000 | [계측](../source/results/benchmark_ViT-B-16-plus-240_20261006/benchmark.json) |

작은 backbone 학습 상태: **completed**. [학습 환경·설정](../source/results/train_vit_b16_plus_240_20261006/environment.json).

작은 backbone MVTec 15개 카테고리 성능 평가: **completed**.
Image AUROC 88.1834%, Pixel AUROC 87.5279%, AUPRO 37.7854%. [7개 지표 CSV](../source/results/mvtec_0shot_vit_b16_plus_240_20261006/mean_metrics.csv). 이 추가 평가는 위의 L14 12개 실행 집계와 구분하며 Table 13의 작은 backbone 성능 칸에만 연결한다.

## 조건과 남은 확인

- 공식 APRIL-GAN commit은 f13b8a634e04f9fde8fa03db125b25af5695d8e1이다. scoring·mask·전처리·few-shot 선택 알고리즘은 수정하지 않았다. 자체 wrapper가 실제 배열과 CLIP 특징, 선택 reference를 수집한다.
- MVTec AD에는 VisA 학습 checkpoint, VisA에는 MVTec AD 학습 checkpoint를 쓴다. 입력 518, ViT-L-14-336/OpenAI, 특징 block 6·12·18·24, 4-shot memory block도 6·12·18·24다.
- 공식 reference 선택은 torch.randint에 따른 중복 허용이다. references.json과 공식 k_shot.txt를 보존한다. 4-shot image score는 text score와 category 내 min-max 정규화한 anomaly-map 최대값의 평균이다.
- RsCIN은 MuSc 공식 Mobile_RsCIN을 호출한다. 공개 논문 특징 파일을 쓰지 않고, MuSc 공식 backbone·loader로 이번 PC에서 별도 추출한 공통 CLIP 특징을 사용한다. APRIL-GAN 자체 특징과는 positional embedding 보간 구현이 달라 이름만 같다고 재사용하지 않았다. 기본 windows는 MVTec [1,2,3], VisA [1,8,9]다. 1은 자기 score를 보존하는 창이다.
- 먼저 계산한 MVTec RsCIN 파일은 APRIL 자체 CLIP 특징을 쓴 pilot이어서 rscin_april_feature_pilot에 분리 보존했다. Table 11의 최종 집계는 공통 MuSc 특징으로 재계산한 rscin 폴더만 읽는다.
- BTAD는 공식 배포의 원본 category 01·02·03, 총 741장 test를 쓴다. metadata를 APRIL-GAN VisaDataset 스키마로 연결했다. VisA 학습 checkpoint와 category 식별자를 prompt로 쓰는 선택은 원문에서 세부 조건이 확인되지 않아 잠정 설정이다. 따라서 BTAD는 실행 완료와 논문 조건 일치 확인을 구분한다.
- 초기 병렬 평가의 WSL 프로세스들이 함께 종료됐다. 두 미완료 시도는 _interrupted1 폴더와 로그에 보존했다. 이후에는 scoring과 metric 수식을 유지하면서 원래 배열을 디스크에 저장하는 container를 사용하고, 한 번에 하나의 평가만 실행한다. 메모리 압박은 원인 가설이며 확정 진단은 아니다.
- 일반 평가 중 수집한 inference_times/benchmark.json은 특징 저장·디스크 cache·계측 비용이 있어 Table 13의 독립 속도 결과로 채택하지 않는다. 별도의 단독 실행 benchmark 결과만 채운다.
- Table 13의 ViT-B-16-plus-240 projection checkpoint는 공식 배포에 없어 자체 학습한다. 공식 VisA 15-epoch 학습 절차를 해당 backbone에 적용하고, MuSc 공식 backbone script의 LAION-400M e31과 4 stage [3,6,9,12], 518 입력을 사용한다. 원문의 작은 backbone 학습 세부 조건은 확인되지 않아 자체 조건으로 구분한다. PyTorch 역전파 호환을 위해 정규화의 제자리 할당만 같은 식의 새 변수 할당으로 바꿨으며 원본 checkout은 보존한다.
- 근거: [평가 코드](../source/run_aprilgan_experiment.py), [직렬 실행 큐](../source/run_aprilgan_serial_remaining.py), [공통 특징 추출](../source/extract_shared_musc_features.py), [RsCIN 코드](../source/evaluate_aprilgan_rscin.py), [속도 측정 코드](../source/benchmark_aprilgan.py), [작은 backbone 학습 코드](../source/train_aprilgan_small_backbone.py), [집계 코드](../source/summarize_aprilgan_remaining.py), [집계 CSV](../source/results/aprilgan_remaining_20261006/aggregates.csv), [진행 CSV](../source/results/aprilgan_remaining_20261006/run_status.csv).
- 대용량 raw 예측·특징·시각화는 /home/test/aprilgan_results/<tag>/에 보존한다. dataset/weights는 연구 repo와 Obsidian에 복사하지 않는다.

- 저장 공간을 위해 완료된 실행의 중복 inference_cache .npy는 최종 압축 원시 예측·특징과 dtype 및 모든 값을 대조한 뒤 정리한다. 원시 예측 NPZ와 카테고리 특징 NPY, 이미지 순서 JSONL을 보존하며, 각 실행의 duplicate_cache_verification.json에 검증 결과를 남긴다. [중복 캐시 검증 코드](../source/compact_aprilgan_cache.py).

- BTAD 재현 입력: [공식 배포·해시·설정](../source/results/btad_dataset_provenance.json), [metadata 생성·기존 순서 대조 코드](../source/prepare_btad_metadata.py), [현재 metadata 검증](../source/results/btad_metadata_verification.json).

## 최종 검증

검증 상태: **passed**. 기본 12개 평가와 작은 backbone 추가 평가를 합한 13개 실행, 6개 설정 집계, 정상 train 참조 선택, 반복 간 이미지·라벨·SHA-256 일치, 평균·표준편차, RsCIN 적용 전 원시 지표 일치, 두 backbone benchmark와 학습 가중치 유한성 검증을 통과했다.

AP가 완벽한 순위에서 1.0000000000000002로 계산되는 float 반올림 사례가 있어 범위 검사에만 10⁻¹² 허용오차를 쓴다. 원시 측정값은 clipping하거나 수정하지 않았다. 논문 조건의 완전 일치는 인증하지 않는다.

[최종 검증 JSON](../source/results/aprilgan_remaining_20261006/final_verification.json) · [검증 코드](../source/verify_aprilgan_remaining.py) · [중단 전후 VisA 7개 카테고리 원시 배열 완전 일치 검증](../source/results/streaming_recovery_verification.json)

## 실행 후 자료 정리

2026-10-07 사용자 요청으로 공식 체크아웃·공개 DRAEM 가중치·중복 및 벤치마크 임시 자료를 삭제했다. 완료된 결과·원시 예측·검증 기록과 직접 학습한 가중치는 보존했다. 재실행에는 삭제된 공식 체크아웃 등을 다시 준비해야 한다. [삭제·보존 내역](../../related_work/markdown/MuSc_comparators_cleanup_20261007.md)을 참조한다.
