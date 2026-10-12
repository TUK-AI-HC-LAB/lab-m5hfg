# MuSc 관련 연구 및 내 PC 재현 근거

작성일: 2026-10-12. 이 문서는 MuSc의 연구 위치와 사용자가 작성한 12페이지 재현 PDF를 함께 설명한다. **원 논문의 주장, 이 PC의 측정, 미실험, 출처 미확인을 구분한다.** 숫자의 근거는 PDF에 적혔다는 사실보다 실행 CSV·설정·검증 기록을 우선한다. 사용자 PDF를 수정하거나 연구 자료의 Obsidian 사본을 만들지 않는다.

**재현 범위는 사용자 지정에 따라 본문 시작부터 `6 Acknowledgments`까지다.** 본문 Table1–7 및 해당 본문 그림·서술을 대상으로 한다. References는 인용 확인용이며 별도 구현 대상이 아니다. Appendix/Table8–18은 범위 밖이므로 미완료 목록이나 본문 재현 완료 판단에 포함하지 않는다. 본문에서 Appendix를 참조하는 문장은 원문 인용으로 구분하며 그 참조만으로 부록 구현을 요구하지 않는다.

## GitHub에서 결과 CSV 열기

아래 링크는 저장소 내부 파일을 가리킨다. 작은 평균 CSV는 GitHub에서 표로 확인할 수 있고, category CSV는 각 제품 결과를 확인하는 근거다. GitHub가 큰 파일의 미리보기를 생략하면 해당 파일 화면의 **Raw/Download raw file**로 열거나 내려받을 수 있다. 성능 CSV의 값은 0–1 단위이며 PDF와 그림에는 ×100한 %를 쓴다. benchmark CSV는 열 이름에 적힌 ms/image·bytes·MB·MiB 단위를 따른다.

| 실험 | 평균·요약 CSV | 상세 CSV |
|---|---|---|
| Table3–5·7 | [조건별 평균](../../method10/source/results/tables3_7_seed42_20261012/mean_metrics.csv) | [27개 category 결과](../../method10/source/results/tables3_7_seed42_20261012/category_metrics.csv) |
| Table6 효율 | [속도·메모리 요약](../../method10/source/results/tables3_7_seed42_20261012/benchmark_summary.csv) | [반복 측정값](../../method10/source/results/tables3_7_seed42_20261012/timing_samples.csv) |
| Figure7 MSM 비율 | [비율별 평균](../../method10/source/results/sweeps_seed42_20261012/msm_interval_ratio_mean_metrics.csv) | [category별 비율 결과](../../method10/source/results/sweeps_seed42_20261012/msm_interval_ratio_category_metrics.csv) |
| Figure8 RsCIN window | [window별 평균](../../method10/source/results/sweeps_seed42_20261012/rscin_single_window_mean_metrics.csv) | [category별 window 결과](../../method10/source/results/sweeps_seed42_20261012/rscin_single_window_category_metrics.csv) |
| 재현 PDF 수치 대조 | [선택 셀 대조](../../method10/source/results/user_pdf_evidence_audit_20261012/selected_pdf_cells.csv) | 26개 선택 셀의 대조이며 PDF 전체 수치 인증은 아님 |

이 문서와 위 CSV, 연결된 검증 기록·실행 source·figure 자료·재현 PDF를 함께 게시한다. WSL 외부 경로의 dataset·weight·대용량 raw NPZ는 게시 대상이 아니며, 해당 경로는 실행 보고서에 provenance로 남긴다.

## Paper Metadata

| Item | Content |
|---|---|
| Title | MuSc: Zero-Shot Industrial Anomaly Classification and Segmentation with Mutual Scoring of the Unlabeled Images |
| Authors | Xurui Li, Ziming Huang, Feng Xue, Yu Zhou — 원 논문 저자 |
| Conference / Journal | ICLR |
| Year | 2024 |
| Paper link | [출판 원문](https://proceedings.iclr.cc/paper_files/paper/2024/file/096b1019463f34eb241e87cfce8dfe16-Paper-Conference.pdf), [보존한 원문 PDF](../../method10/paper/ICLR24_MuSc_Zero_Shot_Industrial_Anomaly_Classification_and_Segmentation_with_Mutual_Scoring_of_the_Unlabeled_Images.pdf) |
| GitHub / Official code | [xrli-U/MuSc](https://github.com/xrli-U/MuSc), 실행 고정 commit `b76b93da8bd3096a99964a96ae29d46f197a0651` |
| Reason for investigation | 정상 reference와 prompt 없이 test 이미지끼리 비교하는 방법을 RTX5080에서 재현하고, 비교군·제거 실험·그림의 근거를 추적하기 위함 |
| 사용자 재현 PDF | [사용자 재현 PDF](../paper/Local26_MuSc_RTX5080_Main_Text_Reproduction.pdf), 12페이지. 원문과 구분한 로컬 작성본 |
| PDF 작성 소스 | 해당 PDF를 만든 `.tex`는 제공되거나 이번 참조 경로에서 확인되지 않음. 아래 실행 Python·shell 소스와 구분 |
| PDF 식별 | SHA-256 `245b703e0959fd92a6e2f3e4233d873f44d86f6a948816b966ca6b2d767c1e25` |

PDF의 표지에는 원 논문 저자명이 남아 있다. 따라서 이 문서는 사용자에게 원 방법의 저자성을 부여하는 대신 **MuSc의 로컬 재현·분석 작업**으로 설명한다. PDF에 남아 있는 연구비·HPC 감사문도 원문 서술이며 RTX5080 자체 실험의 환경 근거로 사용하지 않는다.

## 1. 관련 연구 속 MuSc의 위치

산업 이상 탐지는 정상 reference의 특징·분포와 test 이미지를 비교하는 방법, 정상 데이터를 이용해 모델을 학습하는 방법, 사전학습 모델의 언어 정보를 이용하는 zero-shot 방법으로 나눠 설명할 수 있다. MuSc는 같은 category의 **여러 무라벨 test 이미지**에서 서로 비교 대상을 찾는다 [1]. 사전학습 backbone은 사용하지만 MuSc 자체의 추가 학습이나 text prompt는 요구하지 않는다.

| 계열·예시 | 이용하는 정보·핵심 비교 | MuSc와의 관계 | 이 연구의 연결 자료 |
|---|---|---|---|
| 정상 특징·분포: PatchCore, PaDiM, SPADE | 정상 train의 feature bank·Gaussian·gallery와 query 비교 | 정상 reference를 쓰는 기준점. MuSc는 무라벨 test pool에서 이웃을 찾음 | [PatchCore 조사](CVPR22_PatchCore_Towards_Total_Recall_in_Industrial_Anomaly_Detection.md), [자체 비교 실행](../../method1/markdown/PatchCore_MuSc_comparison_execution.md), [PaDiM](../../method23/markdown/PaDiM_btad_execution.md), [SPADE](../../method20/markdown/SPADE_mvtec_rscin_execution.md) |
| 학습 기반: CutPaste, NSA, DRAEM, STPM, DSR, IGD | 합성 이상·재구성·teacher/student·one-class 목적에 따라 이상 표현 학습 | 학습 없이 사용하는 MuSc와 계산 비용·정보 조건이 다름. 이미 학습된 공개 모델의 평가도 가능하나 학습 완료와 구분 | [CutPaste](../../method22/markdown/CutPaste_mvtec_execution.md), [NSA](../../method13/markdown/NSA_mvtec_execution.md), [DRAEM](../../method12/markdown/DRAEM_mvtec_official_execution.md), [STPM](../../method24/markdown/STPM_mvtec_rscin_execution.md), [DSR](../../method25/markdown/DSR_mvtec_rscin_execution.md), [IGD 취소 기록](../../method14/markdown/IGD_mvtec_execution.md) |
| few-shot: RegAD, GraphCore | 소수 정상 reference와 방법별 학습된 표현 활용 | reference 수와 보조 학습 조건을 유지해서 비교해야 함 | [RegAD 조건별 실행](../../method16/markdown/RegAD_mvtec32_btad4_execution.md), [전체 조건 감사](../../method10/markdown/MuSc_remaining_work_audit_20261010.md) |
| 언어 기반 zero/few-shot: WinCLIP, APRIL-GAN | CLIP 이미지·text 정렬, prompt, 필요시 정상 reference/보조 데이터 학습 | MuSc는 prompt를 쓰지 않으며 test pool의 크기·구성에 의존 | [WinCLIP 조사](CVPR23_WinCLIP_Zero_Few_Shot_Anomaly_Classification_and_Segmentation.md), [WinCLIP 효율](../../method7/markdown/WinCLIP_speed_memory_20261011.md), [APRIL-GAN 실행](../../method11/markdown/APRIL_GAN_remaining_experiments.md) |

여기서 full-shot/few-shot/zero-shot은 target 데이터의 정보 조건이며, 모든 방법의 학습량·backbone이 같다는 뜻이 아니다. 공개 가중치를 평가한 DRAEM·STPM·DSR 등의 수치는 **RTX5080 평가 결과**로 기술하고, 이 PC에서 해당 모델을 학습했다고 쓰지 않는다. PatchCore 등의 정상 특징 구축도 gradient 학습과 구분한다. 각 방법의 상세 조건은 연결한 실행 보고서를 따른다.

## 2. 알아야 하는 방법·지표

- **LNAMD:** ViT patch token 주변 r×r 영역을 집계한다. 작은 r은 작은 결함, 큰 r은 넓은 결함을 표현하는 선택이며 기본 r={1,3,5}를 함께 사용한다 [1].
- **MSM:** query patch에 대해 다른 이미지마다 가장 가까운 patch 거리를 구하고, 그 이미지별 거리의 하위 일정 비율을 평균한다. 모든 이미지의 모든 patch를 바로 평균하는 방식과 다르다 [1].
- **RsCIN:** 이미지별 최대 patch 점수를 class-token similarity 기반 이웃 점수로 재조정한다. 여러 작은 window를 평균하며 최종 image score에 적용한다. pixel map 자체의 smoothing은 아니다 [1].
- **AC/AS:** 이번 본문 범위의 AC는 Image AUROC, AS는 Pixel AUROC다. 부록의 다른 지표 정의를 본문 표에 혼용하지 않는다.
- **F1-max:** threshold를 바꿔 얻은 최대 F1. 고정 threshold의 accuracy와 다르다.
- **category macro mean:** category별 지표의 비가중 평균이다. 전체 dataset 이미지나 픽셀을 한꺼번에 합쳐 계산한 AUROC와 다르다.
- **s:** 같은 category의 무라벨 test pool을 나누는 subset 수다. 각 subset을 독립적으로 scoring한 뒤 모든 예측을 합쳐 category 지표를 계산한다. subset별 AUROC의 평균을 쓰지 않는다.

## 3. 근거의 분류 기준

| 표시 | 뜻 | 사용 가능한 주장 |
|---|---|---|
| **자체 측정·검증** | 같은 조건의 CSV와 실행 설정, raw 계산 검증이 연결됨 | 이 PC에서 해당 조건으로 측정한 수치 |
| **원문 인용** | 원 논문 서술·도식·수치. 로컬 실험 근거를 대신하지 않음 | 원 연구가 보고한 사실로 인용 |
| **미실험/취소** | 필요한 조건의 완료·검증 결과가 없음 | 자체 수치·순위·개선폭을 주장하지 않음 |
| **출처 미확인/혼합** | PDF 값이 최신 CSV와 다르거나 작성 source로 추적하지 못함 | 원문 또는 다른 run의 수치일 가능성만 명시. 자체 수치로 확정하지 않음 |

원문 문장에 있는 “we/ours”와 로컬 PDF에 그림을 삽입했다는 사실만으로 자체 실험임을 판단하지 않는다. 사용자 PDF와 최신 재현 CSV는 별도 버전이며, PDF의 일부가 오래된 run을 사용했을 가능성도 남겨 둔다.

## 4. 사용자 PDF의 표·그림과 실험 근거 연결

아래 page는 사용자 PDF의 실제 페이지다. 원문·로컬 PDF 사이에 그림 번호가 달라진 항목은 번호만으로 같다고 판단하지 않는다.

| PDF 항목 | 분류·확인 내용 | 실행 source | 실험·검증 근거 |
|---|---|---|---|
| p1–3 방법의 배경·기여·관련 연구 | **원문 인용/서술**. 추가 학습 없는 방법이라는 설명과 원문 성능 개선폭을 구분 | [원문 조사](../../method10/markdown/ICLR24_MuSc_Zero_Shot_Industrial_Anomaly_Classification_and_Segmentation.md) | [원문 PDF](../../method10/paper/ICLR24_MuSc_Zero_Shot_Industrial_Anomaly_Classification_and_Segmentation_with_Mutual_Scoring_of_the_Unlabeled_Images.pdf) |
| p4–6 LNAMD 예시·score histogram·RsCIN 예시 | 해당 예시·분포의 **로컬 생성 근거 미확인**. 배경 그림으로 인용 가능하지만 자체 분석으로 확정하지 않음 | [공식 모듈](https://github.com/xrli-U/MuSc/tree/b76b93da8bd3096a99964a96ae29d46f197a0651/models/modules) | 로컬 histogram Figure4/5 전체를 생성·검증한 artifact는 이번 점검에서 확인되지 않음 |
| p7 구현 조건 | backbone·518입력과 stage는 로컬 설정으로 연결 가능. PDF의 VisA window “8,9”만으로는 실제 [1,8,9] 구현을 완전히 설명하지 못함 | [실행 코드](../../method10/source/run_musc_tables3_7.py) | [config](../../method10/source/results/tables3_7_seed42_20261012/config.json), [환경](../../method10/source/results/tables3_7_seed42_20261012/environment.json) |
| Table1 관련 비교·개선폭 서술 | 일부 비교군 결과가 있어도 PDF 문장의 전체 순위·개선폭은 **출처/조건 대조 필요**. VisA MuSc의 Pixel F1/AP/AUPRO 등 빈 항목은 자체 값으로 채우지 않음 | [비교표 생성](../../method10/source/build_musc_reproduction_tables.py) | [최신 비교표](../../method10/markdown/MuSc_all_reproduction_tables.md), [행별 source/provenance](../../method10/source/results/musc_reproduction_tables/provenance.json) |
| p8 Table2 IGD | **미완료·사용자 취소**. PDF 93.4/93.0을 로컬 결과로 쓰면 안 됨. 원문 인용 시 그 출처를 별도 표시 | [IGD source](../../method14/source/run_igd.py) | [취소 상태](../../method14/source/results/mvtec_batch8_seed42_20261011/environment.json), [실행 보고서](../../method14/markdown/IGD_mvtec_execution.md) |
| p8 Table3 LNAMD·Table4 MSM | 자체 실험은 **완료·검증**. PDF 셀 전부가 최신 자체 CSV라는 뜻은 아님; 일부 값은 불일치 | [제거 실험 runner](../../method10/source/run_musc_tables3_7.py) | [category CSV](../../method10/source/results/tables3_7_seed42_20261012/category_metrics.csv), [mean CSV](../../method10/source/results/tables3_7_seed42_20261012/mean_metrics.csv), [검증](../../method10/source/results/tables3_7_seed42_20261012/verification.json) |
| p8 Table5 RsCIN | 동일 raw score의 전/후 실험 **완료·검증**. PDF 일부 값은 원문 값에 가깝고 최신 CSV와 불일치 | [runner](../../method10/source/run_musc_tables3_7.py), [독립 검증](../../method10/source/finish_musc_tables3_7.py) | [Table5 포함 mean CSV](../../method10/source/results/tables3_7_seed42_20261012/mean_metrics.csv) |
| p9 Table6 효율 | 선택한 6개 시간·메모리 셀은 **최신 자체 측정과 소수1자리 일치** | [native benchmark runner](../../method10/source/run_musc_tables3_7.py) | [반복 raw CSV](../../method10/source/results/tables3_7_seed42_20261012/timing_samples.csv), [summary](../../method10/source/results/tables3_7_seed42_20261012/benchmark_summary.csv), [측정 검증](../../method10/source/results/tables3_7_seed42_20261012/benchmark_verification.json) |
| p9 Table7 pool 분할 | 선택한 12개 AC/AS 셀은 **최신 자체 측정과 소수1자리 일치**. 실제 subset 크기는 로컬 CSV를 따름 | [분할 runner](../../method10/source/run_musc_tables3_7.py) | [Table7 포함 mean CSV](../../method10/source/results/tables3_7_seed42_20261012/mean_metrics.csv), [검증](../../method10/source/results/tables3_7_seed42_20261012/verification.json) |
| p9 Figure7 비율 sweep | **자체 측정·검증**. 10/30/50/70/90/100% upper-ratio sweep | [sweep runner](../../method10/source/run_musc_sweeps.py), [plot source](../../method10/source/plot_musc_figures7_8.py) | [mean CSV](../../method10/source/results/sweeps_seed42_20261012/msm_interval_ratio_mean_metrics.csv), [PNG/PDF/SVG·plot data](../../method10/source/figures/figures7_8_local/README.md), [검증](../../method10/source/results/sweeps_seed42_20261012/verification.json) |
| p9 Figure8 window sweep | **자체 측정·검증**. k=2…9·full window·미적용·multi-window | [sweep runner](../../method10/source/run_musc_sweeps.py), [plot source](../../method10/source/plot_musc_figures7_8.py) | [mean CSV](../../method10/source/results/sweeps_seed42_20261012/rscin_single_window_mean_metrics.csv), [CPU 대조](../../method10/source/results/sweeps_seed42_20261012/rscin_verification.json), [그림](../../method10/source/figures/figures7_8_local/README.md) |
| Figure6 관련 정성 비교 서술 | 로컬 anomaly map은 있음. APRIL-GAN raw map 부재로 해당 이미지 열은 **미제작**. PDF에서 “MuSc가 비교군보다 전체 영역을 더 잘 찾음”을 일반화할 근거는 별도 필요 | [샘플 추출](../../method10/source/extract_figure6_examples.py), [render](../../method10/source/render_figure6_local.py) | [로컬 Figure6·선택 샘플·한계](../../method10/source/figures/figure6_local/README.md) |
| p9–10 Appendix A.4/A.6 참조·few-shot 확장 서술 | **원문 인용·부록 실험은 범위 밖**. 해당 참조는 본문에 남길 수 있으나 부록 결과를 자체 측정했다고 표현하지 않음 | [원문 전체](../../method10/paper/ICLR24_MuSc_Zero_Shot_Industrial_Anomaly_Classification_and_Segmentation_with_Mutual_Scoring_of_the_Unlabeled_Images.pdf) | 추가 부록 실행을 본문 재현의 미완료 작업으로 요구하지 않음 |

### 확인된 숫자 차이의 예

아래는 PDF 전체의 자동 인증이 아니라, p8·p9에서 눈으로 확인한 26개 선택 셀의 대조다. 최신 CSV의 0–1 지표는 %로 변환했다. PDF와 CSV 차이가 자동으로 버그나 조작을 의미하지는 않으며, run/source 구분이 필요하다는 뜻이다.

| 선택 셀 | PDF 값 (%) | 최신 자체 CSV (%) | 판정 |
|---|---:|---:|---|
| Table3 r={5}, MVTec AC | 95.4 | 94.5766 | 최신 결과와 다름 |
| Table3 r={1,3,5}, MVTec AS | 97.3 | 97.1149 | 이전 기본 run/원문과 최신 run 구분 필요 |
| Table3 r={1,3,5}, VisA AC | 92.8 | 92.4921 | 최신 결과와 다름 |
| Table5 MVTec w/RsCIN F1-max | 97.5 | 97.3219 | 최신 결과와 다름 |
| Table5 VisA w/RsCIN F1-max | 89.5 | 88.8657 | 최신 결과와 다름 |
| Table2 IGD AC/AS | 93.4 / 93.0 | 자체 완료 값 없음 | 원문 인용 또는 출처 미확인으로만 표시 |

- audit source: [audit_user_musc_pdf.py](../../method10/source/audit_user_musc_pdf.py)
- audit result: [선택 셀 CSV](../../method10/source/results/user_pdf_evidence_audit_20261012/selected_pdf_cells.csv), [PDF hash·검증 범위](../../method10/source/results/user_pdf_evidence_audit_20261012/verification.json)

## 5. 자체 결과로 말할 수 있는 것

기본 r={1,3,5}, 하위30%, s=1의 최신 자체 결과는 MVTec **Image AUROC97.7694%, Pixel AUROC97.1149%**, VisA **92.4921%, 98.7136%**다. 이는 전체 category별 비가중 평균이다. 근거는 [Table3/7 mean CSV](../../method10/source/results/tables3_7_seed42_20261012/mean_metrics.csv)와 [검증](../../method10/source/results/tables3_7_seed42_20261012/verification.json)이다.

RsCIN 전/후로 MVTec F1-max는 **96.7355→97.3219%(+0.5864%p)**, VisA Image AUROC는 **89.5456→92.4921%(+2.9465%p)**였다. PDF 본문의 “0.9% F1-max 개선”과 “2.8% AUROC 개선”은 이 최신 자체 측정의 개선폭이 아니다. 로컬 결론은 [Table5 mean CSV](../../method10/source/results/tables3_7_seed42_20261012/mean_metrics.csv)의 차이로 쓴다.

Figure7에서는 MVTec AC의 최대가30%, AS의 최대가50%; VisA AC의 최대가30%, AS의 최대가10%다. 따라서 “30%가 모든 지표에서 최적”이라고 쓰지 않는다. Figure8에서는 두 dataset의 기본 multi-window F1-max가 이번 single-window 점들보다 높다. MVTec은 k가 커질수록 대체로 감소하지만 VisA는 k=6–7까지 개선되어 **항상 감소**한다고 일반화하지 않는다. 큰 window에서 정상·이상 점수가 섞이는 설명은 가설이며 원인별 기여를 확정한 실험은 아니다. [원본 sweep CSV와 검증](../../method10/markdown/MuSc_sweeps_execution.md).

효율 Table6은 **VisA candle200장의 native pipeline**, 각 s warmup1회·측정3회의 평균이다. 726.4711/420.1018/327.7074ms/image와 7418.1699/7070.6990/6804.2424MB가 관측되었다. 메모리는 allocated bytes/1e6이며 MiB와 reserved를 별도 보존했다. 모델 로딩·metric·저장을 제외하고 data loading을 포함한 core 시간이며 RsCIN 포함 total 시간도 별도 저장했다. 원문 RTX3090 수치와 하드웨어만 바뀐 완전 동일 benchmark라고 주장하지 않는다. [측정 정의·raw](../../method10/markdown/MuSc_tables3_7_execution.md).

## 6. 본문 범위에서 아직 자체 결과로 채우면 안 되는 항목

| 항목 | 상태·이유 | 확인 경로 |
|---|---|---|
 | |
| MuSc VisA Pixel F1-max/AP/AUPRO | Table3–7 sweep에서 Pixel AUROC만 얻은 것과 7개 기본 지표 전체 확보는 다름 | [Figure6의 미측정 표시](../../method10/source/figures/figure6_local/README.md) |
| APRIL-GAN 정성 map | 자체 정량 결과가 있어도 현재 raw map을 찾지 못한 시각화는 따로 미제작 | [Figure6 한계](../../method10/source/figures/figure6_local/README.md) |
| seed 반복·분산 및 원인별 기여 | 단일 seed·batch 공유 검증만으로 통계적 우월성·동등성·window 하락 원인을 확정하지 않음 | [실험 설정·한계](../../method10/markdown/MuSc_tables3_7_execution.md) |

Table8–18의 backbone 변경·큰 r 조합·MSM 하한 sweep·BTAD·few-shot 확장 등은 **이번 재현 범위 밖**이다. 실행 여부와 무관하게 위의 본문 미완료 목록에 포함하지 않는다. 기존 전체 Table1–18 비교표는 확장 자료로만 참조한다.

일부 자동 생성 감사 문서에는 Table3–7·sweep·Figure6 제작 전의 상태 설명과 부록 작업이 섞여 있다. 이번 본문 범위의 최신 완료 판단은 위의 검증 JSON과 실제 CSV를 우선한다.

## 7. 구현 차이와 주장 범위

공식 코드의 핵심 모듈을 실행했지만 모든 경로를 원형 그대로 반복한 것은 아니다. Table4의 reduction을 확장하고 거리 계산을 공유했으며, Table7의 분할 간 backbone 특징을 재사용했다. native MSM·CPU reduction·RsCIN smoothing·raw metric·SHA를 대조했다. 특징 대조 허용오차와 mask 처리·seed·batch·환경 차이는 [실행 보고서](../../method10/markdown/MuSc_tables3_7_execution.md)에 기록되어 있다. **검증 passed는 정한 계산·데이터 확인의 통과이며 원 논문과의 bitwise 동등성이나 모든 실험 재현 완료를 뜻하지 않는다.**

같은 구조를 공통 이상 탐지 프레임워크에 넣은 작업은 [포팅 명세](../../method10/markdown/MuSc_common_framework_porting_spec.md)와 [공식 실행 대비 비교](../../method10/markdown/MuSc_paper_vs_local_mvtec_all.md)로 연결한다. 공통 framework 실행과 이 문서의 공식 구현 제거 실험을 같은 prediction run으로 취급하지 않는다.

## 8. 참고문헌

[1] Li, Xurui, et al. “MuSc: Zero-Shot Industrial Anomaly Classification and Segmentation with Mutual Scoring of the Unlabeled Images.” *The Twelfth International Conference on Learning Representations*, 2024. [출판 원문](https://proceedings.iclr.cc/paper_files/paper/2024/file/096b1019463f34eb241e87cfce8dfe16-Paper-Conference.pdf).

[2] Radford, Alec, et al. “Learning Transferable Visual Models From Natural Language Supervision.” *Proceedings of the 38th International Conference on Machine Learning*, vol.139, 2021, pp.8748–8763. 원문 [1]의 backbone 참고문헌이며 이번 작업에서 CLIP 사전학습 자체를 수행하지 않았다.

[3] Roth, Karsten, et al. “Towards Total Recall in Industrial Anomaly Detection.” *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2022, pp.14318–14328. [조사·원문 연결](CVPR22_PatchCore_Towards_Total_Recall_in_Industrial_Anomaly_Detection.md).

[4] Jeong, Jongheon, et al. “WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation.” *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2023, pp.19606–19616. [조사·원문 연결](CVPR23_WinCLIP_Zero_Few_Shot_Anomaly_Classification_and_Segmentation.md).
