# MuSc 논문 전체 표 — 내 PC 재현 결과와 실행 목록

작성일: 2026-10-06. **본문·부록 Table 1–18의 모든 행과 metric을 Markdown으로 구성했다.** 논문의 성능·시간·메모리 수치를 결과 셀에 복사하지 않았다. 채운 숫자는 보존된 자체 실행 CSV에서만 가져왔다.

원문의 다단 헤더와 좌우 배치를 한 표로 풀고 Dataset·상태·근거 열을 추가했다. 원문의 숫자 순위에 따른 bold·underline와 개선폭을 가져오지 않았다. 자체 비교군이 모두 실행된 뒤 다시 계산한다.

- `미실행`: 해당 표 조건으로 채울 직접 측정값이 아직 이 집계에 없다. 다른 실험 결과가 없다는 뜻은 아니다.
- `측정값 있음·조건 대조 필요`: 자체 실행 근거가 있으나 논문 조건과 모든 항목이 같다고 확정하지 않았다. 완료 인증이 아니다.
- 기존 WinCLIP·PatchCore·PaDiM 결과는 보유해도 dataset·shot·backbone·reference·seed 조건의 일치가 검증되기 전에는 해당 행에 넣지 않는다.
- 단위: 성능은 %, 차이는 %p, 추론 시간은 ms/image, GPU 메모리는 MiB(bytes/1024²). 기본 AC는 Image AUROC, AS는 Pixel AUROC이며 Table 13만 AS=AUPRO다.
- 각 방법의 원래 정보 조건을 유지한다. 모든 비교군의 backbone·resize를 MuSc와 강제로 맞추지 않는다. 학습이 필요한 방법은 보조 데이터/target train 사용 조건을 기록한다.
- 공개 checkpoint 평가는 학습부터 재현한 결과와 구분한다. 4-shot 반복 횟수·reference 선택·표준편차 정의는 원문·공식 코드에서 확정한 뒤 실행한다. 단일 seed에 원문의 ±를 붙이지 않는다.

## 기존 측정값의 조건과 한계

MuSc는 MVTec AD 15개 category·1,725장·seed 42·CLIP ViT-L-14-336·518 입력·r={1,3,5}·s=1 실행이다. 기존 실행의 RsCIN 창 [1,2,3]과 원문 본문의 [2,3] 관계를 공식 구현과 함께 대조해야 한다. 각 표의 기본 행에 같은 기존 측정값을 재사용한 것은 신규 제거 실험을 수행했다는 뜻이 아니다.

APRIL-GAN은 같은 1,725장에 VisA 학습 공개 checkpoint를 사용한 MVTec AD 0-shot 평가다. MuSc와 원본 이미지·라벨·SHA-256은 일치하나 mask 처리와 방법별 scoring이 다르다. 단일 seed이며 보조 데이터 학습을 이번 PC에서 다시 수행하지 않았다.

표의 빈칸을 채우는 것과 논문 전체 재현 완료는 다르다. Figure 6–10 등의 qualitative 결과·곡선과 구현 검증은 별도 작업이다. 이번 요청에서는 Table 1–18을 만들었다.

## 근거와 재생성

- [MuSc 원문 PDF](../paper/ICLR24_MuSc_Zero_Shot_Industrial_Anomaly_Classification_and_Segmentation_with_Mutual_Scoring_of_the_Unlabeled_Images.pdf)
- [MuSc 공식 실행 CSV](../source/results/mvtec_all_paper_20261002/category_metrics.csv)
- [MuSc category 설정 예시](../source/results/mvtec_all_paper_20261002/bottle/config.json)
- [APRIL-GAN 공식 실행 보고서](../../method11/markdown/APRIL_GAN_mvtec_official_execution.md)
- [APRIL-GAN 나머지 실험 최신 상태·반복·RsCIN·benchmark](../../method11/markdown/APRIL_GAN_remaining_experiments.md)
- [DRAEM 공개 가중치 자체 평가·RsCIN 비교](../../method12/markdown/DRAEM_mvtec_official_execution.md)
- [NSA 자체 학습·평가 실행](../../method13/markdown/NSA_mvtec_execution.md)
- [IGD 자체 학습·평가 실행](../../method14/markdown/IGD_mvtec_execution.md)
- [집계 코드](../source/build_musc_reproduction_tables.py)
- [표의 모든 행 JSON/CSV](../source/results/musc_reproduction_tables/table_rows.csv)
- [집계 provenance](../source/results/musc_reproduction_tables/provenance.json)

## 전체 표 목록

| Table | 내용 | 원문 쪽 | 재현 행 수 |
|---|---|---|---|
| 1 | MVTec AD·VisA의 zero/few-shot 비교 | 7 | 17 |
| 2 | MVTec AD의 many-shot 방법 비교 | 8 | 7 |
| 3 | LNAMD aggregation degree 제거 실험 | 8 | 6 |
| 4 | MSM sample strategy 제거 실험 | 8 | 6 |
| 5 | MuSc의 RsCIN 적용 전후 | 8 | 4 |
| 6 | MuSc의 이미지당 추론 시간·최대 GPU 메모리 | 9 | 3 |
| 7 | test pool 분할 수에 따른 성능·부분집합 크기 | 9 | 6 |
| 8 | 사전학습 방식·backbone별 MuSc 성능 | 14 | 11 |
| 9 | 더 큰 aggregation degree 조합 | 15 | 6 |
| 10 | MSM percentage interval 선택 | 16 | 6 |
| 11 | 비교군에 RsCIN을 적용한 classification 결과 | 17 | 24 |
| 12 | 방향·크기가 일정하지 않은 category 비교 | 17 | 7 |
| 13 | 비교군의 추론 시간·GPU 메모리·성능 | 18 | 10 |
| 14 | BTAD의 zero/few/full-shot 비교 | 18 | 10 |
| 15 | MVTec AD의 MuSc few/many-shot 확장 | 19 | 8 |
| 16 | MVTec AD의 MuSc+ few/many-shot 확장 | 19 | 8 |
| 17 | MuSc의 MVTec AD 15개 category 상세 결과 | 20 | 16 |
| 18 | MuSc의 VisA 12개 category 상세 결과 | 20 | 13 |

## Table 1. MVTec AD·VisA의 zero/few-shot 비교

원문 7쪽. 수치는 자체 측정값이며 단위는 별도 표기 외 %.

| Dataset | 비교 블록 | Method | Setting | Image AUROC | Image F1-max | Image AP | Pixel AUROC | Pixel F1-max | Pixel AP | AUPRO | 상태 | 근거 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MVTec AD | 0-shot 비교 | WinCLIP | 0-shot | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | — |
| MVTec AD | 0-shot 비교 | APRIL-GAN | 0-shot | 86.1250 | 90.3505 | 93.5605 | 87.6288 | 43.2758 | 40.7849 | 44.0371 | 1회·complete·조건 대조 필요 | [추가 실험 집계](../../method11/markdown/APRIL_GAN_remaining_experiments.md) |
| MVTec AD | 0-shot 비교 | ACR | 0-shot | 86.5904 | 91.8947 | 93.5808 | 92.5731 | 44.2887 | 38.5545 | 73.9600 | 측정값 있음·조건 대조 필요 | [ACR CSV](../../method15/source/results/mvtec_seed42_20261007/category_metrics.csv) |
| MVTec AD | 0-shot 비교 | MuSc | 0-shot | 97.7694 | 97.3219 | 99.0706 | 97.1149 | 62.1593 | 62.2587 | 93.4755 | 측정값 있음·조건 대조 필요 | [MuSc CSV](../source/results/mvtec_all_paper_20261002/category_metrics.csv) |
| VisA | 0-shot 비교 | WinCLIP | 0-shot | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | — |
| VisA | 0-shot 비교 | APRIL-GAN | 0-shot | 77.5213 | 78.5613 | 80.9004 | 94.1977 | 32.2806 | 25.7560 | 86.5940 | 1회·complete·조건 대조 필요 | [추가 실험 집계](../../method11/markdown/APRIL_GAN_remaining_experiments.md) |
| VisA | 0-shot 비교 | MuSc | 0-shot | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | — |
| MVTec AD | 4-shot 비교 | RegAD | 4-shot | 88.8213 ± 1.0163 | 92.2460 ± 0.3467 | 94.7066 ± 0.7681 | 96.1330 ± 0.2160 | 51.3182 ± 0.8179 | 47.8845 ± 0.9246 | 87.9181 ± 0.4449 | 공개 가중치·support10round(capsule/grid 자체 생성)·재학습 없음 | [RegAD CSV](../../method16/source/results/mvtec_public_20261007/category_metrics.csv) |
| MVTec AD | 4-shot 비교 | PatchCore | 4-shot | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | — |
| MVTec AD | 4-shot 비교 | WinCLIP | 4-shot | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | — |
| MVTec AD | 4-shot 비교 | APRIL-GAN | 4-shot | 92.6505 ± 0.0724 | 92.7513 ± 0.0713 | 96.2305 ± 0.0077 | 95.9100 ± 0.0334 | 56.6658 ± 0.4731 | 54.4308 ± 0.2771 | 91.9175 ± 0.0340 | 3회·complete·조건 대조 필요 | [추가 실험 집계](../../method11/markdown/APRIL_GAN_remaining_experiments.md) |
| MVTec AD | 4-shot 비교 | GraphCore | 4-shot | 80.9780 ± 1.0801 | 89.4086 ± 0.3108 | 89.7765 ± 0.5897 | 88.7568 ± 0.4668 | 36.5777 ± 0.6149 | 27.6922 ± 0.5830 | 75.3666 ± 0.4896 | 저자 benchmark·공개 PyramidViG·support10round·feature층 선택/구조 차이 있음 | [GraphCore CSV](../../method17/source/results/mvtec_pvig_fp32_20261008/category_metrics.csv) |
| MVTec AD | 4-shot 비교 | MuSc | 0-shot | 97.7694 | 97.3219 | 99.0706 | 97.1149 | 62.1593 | 62.2587 | 93.4755 | 측정값 있음·조건 대조 필요 | [MuSc CSV](../source/results/mvtec_all_paper_20261002/category_metrics.csv) |
| VisA | 4-shot 비교 | PatchCore | 4-shot | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | — |
| VisA | 4-shot 비교 | WinCLIP | 4-shot | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | — |
| VisA | 4-shot 비교 | APRIL-GAN | 4-shot | 92.5687 ± 0.1498 | 88.2393 ± 0.2682 | 94.4684 ± 0.0689 | 96.1837 ± 0.0194 | 39.6256 ± 0.3260 | 32.0048 ± 0.1952 | 90.0917 ± 0.1452 | 3회·complete·조건 대조 필요 | [추가 실험 집계](../../method11/markdown/APRIL_GAN_remaining_experiments.md) |
| VisA | 4-shot 비교 | MuSc | 0-shot | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | — |

원문의 ±는 반복 실험의 변동량이다. 단일 seed 결과에 ±를 붙이지 않는다. 원문에서 미보고한 metric도 자체 측정 가능하면 이후 채우되 그 사실을 기록한다.

## Table 2. MVTec AD의 many-shot 방법 비교

원문 8쪽. 수치는 자체 측정값이며 단위는 별도 표기 외 %.

| Method | Setting | AC | AS | 상태 | 근거 |
| --- | --- | --- | --- | --- | --- |
| CutPaste | full-shot | 미실행 | 미실행 | 전체 학습 시작·최종 검증 미완료 | [CutPaste 실행](../../method22/markdown/CutPaste_mvtec_execution.md) |
| NSA | full-shot | 96.8705 | 95.8661 | 자체 학습·15개 category·단일 seed | [NSA 실행](../../method13/markdown/NSA_mvtec_execution.md) |
| IGD | full-shot | 미실행 | 미실행 | 미실행 | — |
| PatchCore | full-shot | 미실행 | 미실행 | 미실행 | — |
| RegAD | 32-shot | 미실행 | 미실행 | 미실행 | — |
| GraphCore | 8-shot | 82.0203 ± 0.4270 | 89.8711 ± 0.3309 | 저자 benchmark·PyramidViG·10support round·구현 조건 차이 있음 | [GraphCore 실행·조건 대조](../../method17/markdown/GraphCore_mvtec_execution.md) |
| MuSc | 0-shot | 97.7694 | 97.1149 | 측정값 있음·조건 대조 필요 | [MuSc CSV](../source/results/mvtec_all_paper_20261002/category_metrics.csv) |



## Table 3. LNAMD aggregation degree 제거 실험

원문 8쪽. 수치는 자체 측정값이며 단위는 별도 표기 외 %.

| 조건 | MVTec AC | MVTec AS | VisA AC | VisA AS | 상태 |
| --- | --- | --- | --- | --- | --- |
| {1} | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| {3} | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| {5} | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| {1,3} | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| {3,5} | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| {1,3,5} | 97.7694 | 97.1149 | 미실행 | 미실행 | MVTec 기본 측정값 있음·조건 대조 필요 |

AC=Image AUROC, AS=Pixel AUROC. 기본 행에만 기존 측정값을 연결했다. 다른 r 조합은 별도 실행한다.

## Table 4. MSM sample strategy 제거 실험

원문 8쪽. 수치는 자체 측정값이며 단위는 별도 표기 외 %.

| 조건 | MVTec AC | MVTec AS | VisA AC | VisA AS | 상태 |
| --- | --- | --- | --- | --- | --- |
| (a) min | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| (b) max | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| (c) mean | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| (d) 30% + min | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| (e) 30% + max | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| (f) 30% + mean | 97.7694 | 97.1149 | 미실행 | 미실행 | MVTec 기본 측정값 있음·조건 대조 필요 |



## Table 5. MuSc의 RsCIN 적용 전후

원문 8쪽. 수치는 자체 측정값이며 단위는 별도 표기 외 %.

| Dataset | RsCIN | AUROC | F1-max | AP | 상태 |
| --- | --- | --- | --- | --- | --- |
| MVTec AD | w/o | 미실행 | 미실행 | 미실행 | 미실행 |
| MVTec AD | w | 97.7694 | 97.3219 | 99.0706 | 측정값 있음·조건 대조 필요 |
| VisA | w/o | 미실행 | 미실행 | 미실행 | 미실행 |
| VisA | w | 미실행 | 미실행 | 미실행 | 미실행 |

w/o는 동일 raw image score에서 RsCIN만 제거한 평가가 필요하다. 현재 w 행은 기존 공식 MuSc 실행값이다.

## Table 6. MuSc의 이미지당 추론 시간·최대 GPU 메모리

원문 9쪽. 수치는 자체 측정값이며 단위는 별도 표기 외 %.

| s | Time (ms/image) | GPU cost (MB) | 상태 |
| --- | --- | --- | --- |
| 1 | 미실행 | 미실행 | 미실행 |
| 2 | 미실행 | 미실행 | 미실행 |
| 3 | 미실행 | 미실행 | 미실행 |

논문은 RTX 3090에서 측정했다. 재현 표는 내 RTX 5080에서 측정한다. 모델 로딩·시각화·metric·raw 저장을 포함한 wall time을 이 칸에 넣지 않는다. 원문은 최대 200장인 pool의 메모리를 측정하므로 같은 pool 조건부터 맞춘다.

## Table 7. test pool 분할 수에 따른 성능·부분집합 크기

원문 9쪽. 수치는 자체 측정값이며 단위는 별도 표기 외 %.

| Dataset | s | AC | AS | Size of subsets | 상태 |
| --- | --- | --- | --- | --- | --- |
| MVTec AD | 1 | 97.7694 | 97.1149 | 42–167 | 측정값 있음·조건 대조 필요 |
| MVTec AD | 2 | 미실행 | 미실행 | 미실행 | 미실행 |
| MVTec AD | 3 | 미실행 | 미실행 | 미실행 | 미실행 |
| VisA | 1 | 미실행 | 미실행 | 미실행 | 미실행 |
| VisA | 2 | 미실행 | 미실행 | 미실행 | 미실행 |
| VisA | 3 | 미실행 | 미실행 | 미실행 | 미실행 |

분할 후 모든 test 이미지의 예측을 합쳐 category metric을 계산한다. subset별 AUROC의 평균으로 대체하지 않는다.

## Table 8. 사전학습 방식·backbone별 MuSc 성능

원문 14쪽. 수치는 자체 측정값이며 단위는 별도 표기 외 %.

| Pre-training | Arch. | Pre-training dataset | MVTec AC | MVTec AS | VisA AC | VisA AS | 상태 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DINO | ViT-B-16 | ImageNet-1k | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| DINO | ViT-B-8 | ImageNet-1k | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| DINOv2 | ViT-B-14 | LVD-142M | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| DINOv2 | ViT-L-14 | LVD-142M | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| CLIP | ViT-B-32 | WIT-400M | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| CLIP | ViT-B-16 | WIT-400M | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| CLIP | ViT-B-16-plus-240 | LAION-400M | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| CLIP | ViT-L-14 | WIT-400M | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| CLIP | ViT-L-14-336 | WIT-400M | 97.7694 | 97.1149 | 미실행 | 미실행 | 측정값 있음·조건 대조 필요 |
| — | Swin-B-4 | ImageNet-22k | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| — | Swin-L-4 | ImageNet-22k | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |

사전학습 데이터명은 실험 조건이며 성능 수치가 아니다. ViT-Base는 3층씩, ViT-Large는 6층씩 4 stage를 구성하고 Swin은 자체 stage를 따른다.

## Table 9. 더 큰 aggregation degree 조합

원문 15쪽. 수치는 자체 측정값이며 단위는 별도 표기 외 %.

| 조건 | MVTec AC | MVTec AS | VisA AC | VisA AS | 상태 |
| --- | --- | --- | --- | --- | --- |
| {1} | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| {1,3} | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| {1,3,5} | 97.7694 | 97.1149 | 미실행 | 미실행 | MVTec 기본 측정값 있음·조건 대조 필요 |
| {1,3,5,7} | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| {1,3,5,7,9} | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| {1,3,5,7,9,13} | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |



## Table 10. MSM percentage interval 선택

원문 16쪽. 수치는 자체 측정값이며 단위는 별도 표기 외 %.

| 조건 | MVTec AC | MVTec AS | VisA AC | VisA AS | 상태 |
| --- | --- | --- | --- | --- | --- |
| 0%–30% | 97.7694 | 97.1149 | 미실행 | 미실행 | MVTec 기본 측정값 있음·조건 대조 필요 |
| 2%–30% | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| 4%–30% | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| 6%–30% | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| 8%–30% | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| 10%–30% | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |

오름차순 score의 하위 Y%를 제외하고 Y%–30% 구간 평균을 사용한다. Table 4의 sample strategy 실험과 구분한다.

## Table 11. 비교군에 RsCIN을 적용한 classification 결과

원문 17쪽. 수치는 자체 측정값이며 단위는 별도 표기 외 %.

| Dataset | Method | Setting | RsCIN | AUROC | F1-max | AP | 상태 | 근거 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MVTec AD | SPADE | full-shot | w/o | 미실행 | 미실행 | 미실행 | 미실행 | — |
| MVTec AD | SPADE | full-shot | w | 미실행 | 미실행 | 미실행 | 미실행 | — |
| MVTec AD | DRAEM | full-shot | w/o | 98.0541 | 96.9628 | 98.9657 | 공개 가중치 자체 평가·15개 category·재학습 없음·조건 대조 필요 | [DRAEM 실행](../../method12/markdown/DRAEM_mvtec_official_execution.md) |
| MVTec AD | DRAEM | full-shot | w | 97.8976 | 96.9835 | 99.0759 | 공개 가중치 자체 평가·15개 category·재학습 없음·조건 대조 필요 | [DRAEM 실행](../../method12/markdown/DRAEM_mvtec_official_execution.md) |
| MVTec AD | STPM | full-shot | w/o | 미실행 | 미실행 | 미실행 | 미실행 | — |
| MVTec AD | STPM | full-shot | w | 미실행 | 미실행 | 미실행 | 미실행 | — |
| MVTec AD | APRIL-GAN | 0-shot | w/o | 86.1250 | 90.3505 | 93.5605 | 1회·자체 RsCIN 측정·조건 대조 필요 | [RsCIN 집계](../../method11/markdown/APRIL_GAN_remaining_experiments.md) |
| MVTec AD | APRIL-GAN | 0-shot | w | 86.1692 | 90.8962 | 93.6730 | 1회·자체 RsCIN 측정·조건 대조 필요 | [RsCIN 집계](../../method11/markdown/APRIL_GAN_remaining_experiments.md) |
| MVTec AD | APRIL-GAN | 4-shot | w/o | 92.6505 ± 0.0724 | 92.7513 ± 0.0713 | 96.2305 ± 0.0077 | 3회·자체 RsCIN 측정·조건 대조 필요 | [RsCIN 집계](../../method11/markdown/APRIL_GAN_remaining_experiments.md) |
| MVTec AD | APRIL-GAN | 4-shot | w | 93.3863 ± 0.0492 | 93.3749 ± 0.2192 | 96.7462 ± 0.0207 | 3회·자체 RsCIN 측정·조건 대조 필요 | [RsCIN 집계](../../method11/markdown/APRIL_GAN_remaining_experiments.md) |
| VisA | APRIL-GAN | 0-shot | w/o | 77.5213 | 78.5613 | 80.9004 | 1회·자체 RsCIN 측정·조건 대조 필요 | [RsCIN 집계](../../method11/markdown/APRIL_GAN_remaining_experiments.md) |
| VisA | APRIL-GAN | 0-shot | w | 78.9014 | 80.0743 | 82.0228 | 1회·자체 RsCIN 측정·조건 대조 필요 | [RsCIN 집계](../../method11/markdown/APRIL_GAN_remaining_experiments.md) |
| MVTec AD | PatchCore | full-shot | w/o | 미실행 | 미실행 | 미실행 | 미실행 | — |
| MVTec AD | PatchCore | full-shot | w | 미실행 | 미실행 | 미실행 | 미실행 | — |
| MVTec AD | DSR | full-shot | w/o | 미실행 | 미실행 | 미실행 | 미실행 | — |
| MVTec AD | DSR | full-shot | w | 미실행 | 미실행 | 미실행 | 미실행 | — |
| MVTec AD | RegAD | 2-shot | w/o | 85.3483 ± 0.9444 | 91.5156 ± 0.4033 | 92.6173 ± 0.5601 | 공개 가중치·15개 category·support10round(capsule/grid 자체 생성)·자체 RsCIN | [RegAD 실행](../../method16/markdown/RegAD_mvtec_execution.md) |
| MVTec AD | RegAD | 2-shot | w | 86.4475 ± 0.8961 | 92.3205 ± 0.4738 | 93.0947 ± 0.5377 | 공개 가중치·15개 category·support10round(capsule/grid 자체 생성)·자체 RsCIN | [RegAD 실행](../../method16/markdown/RegAD_mvtec_execution.md) |
| MVTec AD | RegAD | 4-shot | w/o | 88.8213 ± 1.0163 | 92.2460 ± 0.3467 | 94.7066 ± 0.7681 | 공개 가중치·15개 category·support10round(capsule/grid 자체 생성)·자체 RsCIN | [RegAD 실행](../../method16/markdown/RegAD_mvtec_execution.md) |
| MVTec AD | RegAD | 4-shot | w | 89.9615 ± 1.0092 | 93.2572 ± 0.4279 | 95.1241 ± 0.7279 | 공개 가중치·15개 category·support10round(capsule/grid 자체 생성)·자체 RsCIN | [RegAD 실행](../../method16/markdown/RegAD_mvtec_execution.md) |
| MVTec AD | RegAD | 8-shot | w/o | 91.3723 ± 0.4621 | 93.3837 ± 0.2852 | 95.7999 ± 0.2973 | 공개 가중치·15개 category·support10round(capsule/grid 자체 생성)·자체 RsCIN | [RegAD 실행](../../method16/markdown/RegAD_mvtec_execution.md) |
| MVTec AD | RegAD | 8-shot | w | 92.0118 ± 0.4007 | 94.1207 ± 0.2536 | 96.0918 ± 0.2877 | 공개 가중치·15개 category·support10round(capsule/grid 자체 생성)·자체 RsCIN | [RegAD 실행](../../method16/markdown/RegAD_mvtec_execution.md) |
| VisA | APRIL-GAN | 4-shot | w/o | 92.5687 ± 0.1498 | 88.2393 ± 0.2682 | 94.4684 ± 0.0689 | 3회·자체 RsCIN 측정·조건 대조 필요 | [RsCIN 집계](../../method11/markdown/APRIL_GAN_remaining_experiments.md) |
| VisA | APRIL-GAN | 4-shot | w | 94.5818 ± 0.1753 | 90.4776 ± 0.2696 | 95.7089 ± 0.1140 | 3회·자체 RsCIN 측정·조건 대조 필요 | [RsCIN 집계](../../method11/markdown/APRIL_GAN_remaining_experiments.md) |

원문의 *는 VisA를 뜻하며 여기서는 Dataset 열로 풀었다. 원문에 명시하지 않은 full-shot 표기는 해당 방법의 기본 조건으로 정리한 것이므로 세부 학습 설정을 추가 확인한다. RsCIN 입력은 ViT-L-14-336 class token으로 통일한다. 기존 PatchCore 실행을 이 실험의 w/o 행으로 자동 대입하지 않는다.

## Table 12. 방향·크기가 일정하지 않은 category 비교

원문 17쪽. 수치는 자체 측정값이며 단위는 별도 표기 외 %.

| Category | Dataset | WinCLIP AC | WinCLIP AS | APRIL-GAN AC | APRIL-GAN AS | MuSc AC | MuSc AS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| screw | MVTec AD | 미실행 | 미실행 | 84.9047 | 97.7630 | 82.5579 | 98.7672 |
| hazelnut | MVTec AD | 미실행 | 미실행 | 89.3750 | 96.1228 | 99.6071 | 99.3755 |
| metal_nut | MVTec AD | 미실행 | 미실행 | 68.4751 | 65.4553 | 96.5787 | 86.1210 |
| capsules | VisA | 미실행 | 미실행 | 62.3417 | 97.4573 | 미실행 | 미실행 |
| macaroni2 | VisA | 미실행 | 미실행 | 65.5850 | 97.8123 | 미실행 | 미실행 |
| mean | 선택된 5개 category | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| mean-ALL | MVTec AD | 미실행 | 미실행 | 86.1250 | 87.6288 | 97.7694 | 97.1149 |

metal nut은 파일 category metal_nut으로 썼다. capsules는 VisA이며 MVTec의 capsule과 다르다. mean은 5개 category가 모두 측정된 뒤 계산한다. mean-ALL은 원문의 값이 MVTec 전체 평균을 가리키므로 MVTec AD로 표시했다. 모든 채운 값은 조건 대조가 남아 있다.

## Table 13. 비교군의 추론 시간·GPU 메모리·성능

원문 18쪽. 수치는 자체 측정값이며 단위는 별도 표기 외 %.

| Method | Backbone | Setting | s | Training | Time (ms/image) | GPU allocated (MiB) | AC | AS=AUPRO | 상태 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RegAD | ResNet-18 | 4-shot | — | yes | 미실행 | 미실행 | 88.8213 ± 1.0163 | 87.9181 ± 0.4449 | 측정값 있음·조건 대조 필요·속도 미측정 |
| APRIL-GAN | ViT-L-14-336 | 0-shot | — | yes | 50.8219 | 2490.3623 | 86.1250 | 44.0371 | 단독 속도 측정·조건 대조 필요 |
| MuSc | ViT-L-14-336 | 0-shot | 1 | no | 미실행 | 미실행 | 97.7694 | 93.4755 | 측정값 있음·조건 대조 필요·속도 미측정 |
| MuSc | ViT-L-14-336 | 0-shot | 2 | no | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| MuSc | ViT-L-14-336 | 0-shot | 3 | no | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| APRIL-GAN | ViT-B-16-plus-240 | 0-shot | — | yes | 17.4080 | 1488.1372 | 88.1834 | 37.7854 | 단독 속도 측정·조건 대조 필요·자체 학습 checkpoint |
| WinCLIP | ViT-B-16-plus-240 | 0-shot | — | no | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| MuSc | ViT-B-16-plus-240 | 0-shot | 1 | no | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| MuSc | ViT-B-16-plus-240 | 0-shot | 2 | no | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| MuSc | ViT-B-16-plus-240 | 0-shot | 3 | no | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |

**이 표의 AS는 Pixel AUROC가 아니라 AUPRO다.** Training=yes는 방법에 학습된 모듈이 있다는 뜻이며 이번 실행에서 추가 학습했는지와 구분한다. 원문에서 미보고한 셀은 재현에서도 직접 측정하기 전에는 채우지 않는다.

## Table 14. BTAD의 zero/few/full-shot 비교

원문 18쪽. 수치는 자체 측정값이며 단위는 별도 표기 외 %.

| Method | Setting | AC | AS | 상태 |
| --- | --- | --- | --- | --- |
| VT-ADL | full-shot | 58.2809 | 79.7399 | 자체 학습·제품3개·400epoch·Gaussian150·공개코드 수정 명시 |
| P-SVDD | full-shot | 69.7564 | 78.0765 | 자체 학습·제품3개·299학습epoch·공식 기본λ1·설정 차이 명시 |
| SPADE | full-shot | 87.1499 | 97.6713 | 제품3개·K50·논문 수식 재현·비공식 구현·조건 차이 명시 |
| PaDiM | full-shot | 미실행 | 미실행 | 미실행 |
| PyramidFlow | full-shot | 91.2218 | 95.6597 | 제품3개·Res18·15epoch·최종모델·설정 차이 명시 |
| PatchCore | 4-shot | 미실행 | 미실행 | 미실행 |
| RegAD | 4-shot | 미실행 | 미실행 | 미실행 |
| APRIL-GAN | 4-shot | 91.9072 ± 0.1170 | 96.2023 ± 0.0943 | 3회·측정값 있음·BTAD 세부 조건 잠정 |
| APRIL-GAN | 0-shot | 73.7895 | 91.4034 | 1회·측정값 있음·BTAD 세부 조건 잠정 |
| MuSc | 0-shot | 미실행 | 미실행 | 미실행 |

MVTec AD 조건을 BTAD에 적용하는 원문 protocol을 따르고 BTAD test 데이터로 hyperparameter를 조정하지 않는다. VT-ADL: [실행·설정 차이](../../method18/markdown/VT_ADL_btad_execution.md). P-SVDD: [실행·설정 차이](../../method19/markdown/P_SVDD_btad_execution.md). SPADE: [실행·설정 차이](../../method20/markdown/SPADE_btad_execution.md). PyramidFlow: [실행·설정 차이](../../method21/markdown/PyramidFlow_btad_execution.md).

## Table 15. MVTec AD의 MuSc few/many-shot 확장

원문 19쪽. 수치는 자체 측정값이며 단위는 별도 표기 외 %.

| Setting | AC | AS | 상태 |
| --- | --- | --- | --- |
| 0-shot | 97.7694 | 97.1149 | 측정값 있음·조건 대조 필요 |
| 1-shot | 미실행 | 미실행 | 미실행 |
| 2-shot | 미실행 | 미실행 | 미실행 |
| 4-shot | 미실행 | 미실행 | 미실행 |
| 8-shot | 미실행 | 미실행 | 미실행 |
| 16-shot | 미실행 | 미실행 | 미실행 |
| 32-shot | 미실행 | 미실행 | 미실행 |
| full-shot | 미실행 | 미실행 | 미실행 |

0-shot은 MuSc 기본 결과다. MuSc는 정상 reference를 기존 test pool 상호 비교에 추가한다. MuSc+는 정상 reference만으로 patch score를 계산하므로 별도 구현·실험이 필요하다.

## Table 16. MVTec AD의 MuSc+ few/many-shot 확장

원문 19쪽. 수치는 자체 측정값이며 단위는 별도 표기 외 %.

| Setting | AC | AS | 상태 |
| --- | --- | --- | --- |
| 0-shot | 97.7694 | 97.1149 | 측정값 있음·조건 대조 필요 |
| 1-shot | 미실행 | 미실행 | 미실행 |
| 2-shot | 미실행 | 미실행 | 미실행 |
| 4-shot | 미실행 | 미실행 | 미실행 |
| 8-shot | 미실행 | 미실행 | 미실행 |
| 16-shot | 미실행 | 미실행 | 미실행 |
| 32-shot | 미실행 | 미실행 | 미실행 |
| full-shot | 미실행 | 미실행 | 미실행 |

0-shot은 MuSc 기본 결과다. MuSc는 정상 reference를 기존 test pool 상호 비교에 추가한다. MuSc+는 정상 reference만으로 patch score를 계산하므로 별도 구현·실험이 필요하다.

## Table 17. MuSc의 MVTec AD 15개 category 상세 결과

원문 20쪽. 수치는 자체 측정값이며 단위는 별도 표기 외 %.

| Class | Image AUROC | Image F1-max | Image AP | Pixel AUROC | Pixel F1-max | Pixel AP | AUPRO | 상태 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bottle | 99.9206 | 99.2126 | 99.9752 | 98.4773 | 79.1673 | 83.0416 | 96.1027 | 측정값 있음·조건 대조 필요 |
| cable | 99.1004 | 97.2973 | 99.4938 | 95.7627 | 60.9783 | 57.6998 | 89.6186 | 측정값 있음·조건 대조 필요 |
| capsule | 96.6494 | 94.9309 | 99.3325 | 98.9564 | 49.7988 | 48.4481 | 95.5044 | 측정값 있음·조건 대조 필요 |
| carpet | 99.8395 | 98.8889 | 99.9499 | 99.4536 | 73.3367 | 76.0552 | 97.5819 | 측정값 있음·조건 대조 필요 |
| grid | 98.6633 | 96.4912 | 99.5372 | 98.1653 | 43.9301 | 38.2309 | 93.9317 | 측정값 있음·조건 대조 필요 |
| hazelnut | 99.6071 | 98.5507 | 99.7907 | 99.3755 | 73.4079 | 73.2828 | 92.2300 | 측정값 있음·조건 대조 필요 |
| leather | 100.0000 | 100.0000 | 100.0000 | 99.7170 | 62.8446 | 64.4696 | 98.7349 | 측정값 있음·조건 대조 필요 |
| metal_nut | 96.5787 | 97.3822 | 99.1445 | 86.1210 | 46.2250 | 47.5429 | 89.3400 | 측정값 있음·조건 대조 필요 |
| pill | 96.3448 | 95.8904 | 99.3325 | 97.4723 | 65.5317 | 67.2379 | 98.0108 | 측정값 있음·조건 대조 필요 |
| screw | 82.5579 | 88.4462 | 91.0461 | 98.7672 | 41.8867 | 36.1239 | 94.3990 | 측정값 있음·조건 대조 필요 |
| tile | 100.0000 | 100.0000 | 100.0000 | 97.8983 | 74.7124 | 78.9029 | 94.6450 | 측정값 있음·조건 대조 필요 |
| toothbrush | 100.0000 | 100.0000 | 100.0000 | 99.5346 | 70.1953 | 67.7893 | 95.4786 | 측정값 있음·조건 대조 필요 |
| transistor | 99.3750 | 95.2381 | 99.1022 | 91.3843 | 59.2512 | 58.4143 | 77.2170 | 측정값 있음·조건 대조 필요 |
| wood | 98.5088 | 98.3333 | 99.5248 | 97.2348 | 68.6367 | 74.7484 | 94.5024 | 측정값 있음·조건 대조 필요 |
| zipper | 99.3960 | 99.1667 | 99.8300 | 98.4039 | 62.4870 | 61.8920 | 94.8351 | 측정값 있음·조건 대조 필요 |
| Mean | 97.7694 | 97.3219 | 99.0706 | 97.1149 | 62.1593 | 62.2587 | 93.4755 | 측정값 있음·조건 대조 필요 |

기존 공식 구현체 실행의 실제 값이다. 15개 category의 비가중 평균을 썼다. 공통 framework 결과와 섞지 않았다. 근거: [MuSc CSV](../source/results/mvtec_all_paper_20261002/category_metrics.csv)

## Table 18. MuSc의 VisA 12개 category 상세 결과

원문 20쪽. 수치는 자체 측정값이며 단위는 별도 표기 외 %.

| Class | Image AUROC | Image F1-max | Image AP | Pixel AUROC | Pixel F1-max | Pixel AP | AUPRO | 상태 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| candle | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| capsules | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| cashew | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| chewinggum | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| fryum | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| macaroni1 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| macaroni2 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| pcb1 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| pcb2 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| pcb3 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| pcb4 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| pipe_fryum | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |
| Mean | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 | 미실행 |


