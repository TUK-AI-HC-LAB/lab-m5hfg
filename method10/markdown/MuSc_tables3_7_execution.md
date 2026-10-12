# MuSc Table 3–7 자체 실행

## 질문과 실행 조건

MuSc ICLR2024의 aggregation degree, MSM reduction, RsCIN, pool 분할과 효율 조건을 공식 구현으로 실행해 이 PC의 값을 확보한다. 실행 전 가설은 기본 r={1,3,5}, 하위30% 평균, RsCIN이 양 데이터셋에서 유용하지만 pool 분할이 계산량과 성능을 함께 바꾼다는 것이다. 결과를 얻기 전 개선을 주장하지 않는다.

고정 commit `b76b93da8bd3096a99964a96ae29d46f197a0651`, CLIP ViT-L-14-336/OpenAI, 입력518, 0-based layers=[5,11,17,23], batch4, seed42. native FP16 autocast와 FP64 점수 집계, FP32 RsCIN을 사용한다. TF32는 기존 공식 실행과 같이 끈다. 추가 학습은 없다. MVTec15 category/1,725장, VisA12 category/2,162장을 사용한다.

| Table | 실행 조건 | 지표 |
|---|---|---|
| 3 | r={1},{3},{5},{1,3},{3,5},{1,3,5} | 양 dataset Image/Pixel AUROC |
| 4 | 전체 min/max/mean; 하위30% min/max/mean | 양 dataset Image/Pixel AUROC |
| 5 | r={1,3,5}, 하위30% 평균의 동일 점수에 RsCIN 전/후 | 양 dataset Image AUROC/F1-max/AP |
| 6 | 공식 pipeline, VisA candle200장, s=1/2/3, 각 warmup1+반복3 | ms/image, peak allocated/reserved bytes·MB·MiB |
| 7 | 공식 loader의 s=1/2/3 subset, 모든 예측을 원래 이미지 순서로 합침 | 양 dataset Image/Pixel AUROC, 실제 subset 크기 범위 |

## 원본 코드와 공유 계산

backbone/loader/LNAMD/RsCIN은 공식 코드를 사용한다. Table4의 공식 배포 코드에 없는 다섯 reduction은 논문 정의대로 확장한다. 동일 native fast cdist의 patch-to-image 최소거리를 한번 계산해 여섯 reduction에 공유한다. 각 category/r/layer/subset의 첫 query를 공식 compute_scores_fast와 대조하고, 여섯 reduction을 CPU NumPy로 독립 대조한다. score 평균은 공식처럼 FP64로 수행한다.

Table3은 같은 r별 점수의 조합 평균으로, Table5는 같은 기본 점수의 RsCIN 전/후로 구하므로 중복 backbone 추론하지 않는다. Table7은 공식 loader의 anomaly-type별 seed42 분할을 따르며 고정 backbone 특징을 공유한다. 각 category의 subset 첫 batch를 별도 추론해 특징을 대조한다. 이 대조는 rtol0.01/atol0.005의 특징 검증이며 batch 구성 변화에 대한 bitwise 동등성 주장은 아니다. 점수·성능의 논문 동등성도 주장하지 않는다. RsCIN은 공식처럼 subset을 합친 category 전체에서 적용한다.

원본 예측은 압축 patch 점수·class feature·원본 loader mask로 저장한다. FP64 patch map을 torch bilinear(518, align_corners=True)로 복원하면 평가 map을 다시 생성할 수 있다. 복원된 모든 조건의 Image/Pixel AUROC와 이미지 F1/AP, input/raw SHA, subset coverage를 검증한 뒤 최종 표에 반영한다. raw 및 모델 캐시는 보존한다.

## 효율 측정 범위

Table6은 공유 거리 계산으로 가속한 ablation runner를 재는 대신 공식 make_category_data를 그대로 사용한다. 최대 크기의 VisA candle200 pool, 각 s의 전체 warmup1회 뒤3회 측정, GPU synchronize 및 반복별 peak 초기화를 적용한다. data loading/전처리/encoder/LNAMD/MSM/interpolation을 포함하고 모델 로딩·metric·시각화·저장을 제외한다. RsCIN 제외 core와 포함 total 시간을 함께 기록한다. peak allocated와 reserved를 함께 저장하고 표의 MB는 allocated bytes/1e6이다. MiB도 별도 저장한다. 논문은 측정 범위를 완전히 명시하지 않아 GPU 종류뿐 아니라 범위 차이도 명시한다.

- commit: `b76b93da8bd3096a99964a96ae29d46f197a0651`
- sh: [run_musc_tables3_7.sh](../source/run_musc_tables3_7.sh)
- state: [environment.json](../source/results/tables3_7_seed42_20261012/environment.json)
- result: [결과 폴더](../source/results/tables3_7_seed42_20261012/)
- raw: `/home/test/musc_results/tables3_7_seed42_20261012`
- log: `/home/test/musc_results/tables3_7_seed42_20261012.log`

실행 시작은 완료가 아니다. 최종 verification.json이 passed인 수치만 MuSc_all_reproduction_tables.md에 반영한다. IGD는 사용자 취소 상태이며 재개하지 않는다.

## 최종 결과

27개 category/3,887장의 459개 category 조건과 속도9개 반복 측정을 검증했다. Table3/4/5/7 수치는 category별 비가중 평균이며 Table6는 RTX5080 VisA candle200 pool이다. 논문 수치를 실험값으로 대입하지 않았다.

[평균 CSV](../source/results/tables3_7_seed42_20261012/mean_metrics.csv) · [검증](../source/results/tables3_7_seed42_20261012/verification.json) · [속도 CSV](../source/results/tables3_7_seed42_20261012/benchmark_summary.csv)
