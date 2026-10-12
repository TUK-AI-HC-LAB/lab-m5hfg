# MuSc 재현에 필요한 전체 작업 재감사 — 2026-10-10

기존 목록의 누락·중복과 표 집계의 미연결을 바로잡았다. 기준은 원문 Table 1–18과 실제 저장 CSV/검증 JSON이다. 연결된 원문 PDF는 부록 포함 **26쪽**, Table 18개, Figure 16개다. 이 문서의 완료는 내 PC에서 해당 결과를 확보했다는 뜻이며 논문의 수치·설정 동등성 인증이 아니다.

## 이번에 바로잡은 내용

- WinCLIP은 MVTec 15개·VisA 12개 모두 0/4-shot **7지표 전부** 있다. F1-max/AP 부족이라는 앞선 설명은 틀렸다. Table 1·12·13 성능을 기존 CSV에 연결했다. 4-shot 단일 seed를 여러 seed의 평균±표준편차로 표현하지 않는다.
- MuSc MVTec RsCIN 적용 전 점수도 저장돼 있다. 모델을 다시 실행하지 않고 전/후 AUROC·F1-max·AP를 계산했으며 후처리 후 값은 기존 CSV와 1e-12 이내 일치했다. Table 5를 채웠다.
- DSR·STPM은 Table 11에 필요하다. SPADE의 기존 BTAD 실행은 MVTec RsCIN 비교를 대체하지 않는다.
- PaDiM은 기존 공통 프레임워크 MVTec 결과가 있다. 필요한 새 조건은 **BTAD full-shot**이다.
- PatchCore도 과거 MVTec full-shot AUROC 결과 CSV가 있다. 새 실행은 다른 데이터/shot과 RsCIN·원시결과 보존을 포함한다. 4조건 모두 완료했고 105개 category run 검증을 통과했다. 중복 실행하지 않는다.
- 원문에서 '-'로 미보고한 지표는 원문과 같은 표를 채우기 위한 필수가 아니다. 자체 추가 지표는 별도로 표시한다. WinCLIP 0-shot에 반복 표준편차를 새로 요구하지 않는다.

## 중복을 제외한 비교군 상태

비교군 **17개 방법, 34개 dataset/shot 조건**이다. RsCIN 전후는 같은 원시 점수의 후처리이므로 별도 학습 조건으로 중복 세지 않았다. 반복 seed 수도 조건 수에 더하지 않았다. MuSc 자체 제거 실험·확장·benchmark는 이 34개와 별도다.

| 상태 | 고유 조건 수 | 의미 |
|---|---:|---|
| 기존 결과 있음 | 33 | 지표 CSV/검증 근거 보유; WinCLIP4 분산·RegAD4속도 등의 추가 작업은 남음 |
| 현재 PatchCore pipeline | 0 | 4조건 모두 완료·105개 category run 검증 통과 |
| 새 조건의 결과 필요 | 0 | 중단한 IGD를 제외한 비교군 조건 결과 확보 |
| 사용자 요청으로 중단 | 1 | IGD MVTecfull; 자동 재개하지 않음 |

## 방법별 필요한 일

| 방법 | 보유 결과 | 남은 필수 작업 | 표 |
|---|---|---|---|
| WinCLIP | MVTec·VisA0/4 각7지표; 네조건 batch1/32 속도·메모리 검증 | 4-shot 반복 정확도 평균/분산; 구현·측정범위 대조 | 1,12,13 |
| APRIL-GAN | MVTec·VisA·BTAD0/4, RsCIN, 두backbone속도 | 새 조건 실행 불필요; 설정 차이 검토 | 1,11,12,13,14 |
| ACR | MVTec0 7지표 | 새 실행 불필요; 설정 차이 검토 | 1 |
| RegAD | MVTec2/4/8와RsCIN | MVTec32, BTAD4, MVTec4속도/메모리 | 1,2,11,13,14 |
| GraphCore | MVTec4/8 | 새 실행 불필요; backbone/feature층 차이 유지 | 1,2 |
| PatchCore | MVTecfull/4·VisA4·BTAD4·RsCIN 검증 완료 | 새 실행 불필요 | 1,2,11,14 |
| CutPaste | MVTecfull 자체학습/평가 완료 | 새 실행 불필요 | 2 |
| NSA | MVTecfull 자체학습/평가 완료 | 새 실행 불필요 | 2 |
| IGD | 2026-10-11 새 실행 중간 모델 보존·전체 결과 없음 | 사용자 실행 취소·자동 재개 금지 | 2 |
| DRAEM | MVTecfull 공개가중치 자체평가/RsCIN | 비교표 수치 확보 완료; 내PC 재학습 완료와 구분 | 11 |
| STPM | MVTec15 공개student 평가·RsCIN 검증 완료 | 새 실행 불필요·재학습과 구분 | 11 |
| DSR | MVTec15 공개모델 평가·RsCIN 검증 완료 | 새 실행 불필요·재학습과 구분 | 11 |
| SPADE | BTADfull3제품; MVTec 검증 완료 | 추가 실행 불필요 | 11,14 |
| VT-ADL | BTADfull3제품 | 새 실행 불필요; 코드/설정차이 유지 | 14 |
| P-SVDD | BTADfull3제품 | 표용 수치 있음; 03제품50% 결과 진단은 별도이며 수치를 바꾸지 않음 | 14 |
| PaDiM | 기존 MVTec; BTAD 검증 완료 | 새 실행 불필요 | 14 |
| PyramidFlow | BTADfull3제품 | 새 실행 불필요 | 14 |

## MuSc 자체 작업

| 실험 | 실제 필요한 결과 | 중복 재사용 |
|---|---|---|
| 기본0-shot | VisA12개7지표·BTAD3개image/pixel AUROC | VisA한번으로Table1/5/7(s1)/8(기본)/12/18 공유. MVTec기본은 이미 있음 |
| LNAMD | 두dataset r={1},{3},{5},{1,3},{3,5},{1,3,5},{1,3,5,7},{1,3,5,7,9},{1,3,5,7,9,13} | Table3/9 중복r조합은 같은조건 결과 공유 |
| MSM 전략 | min/max/mean 및 하위30%의 min/max/mean | 기본30%mean 재사용; min계열의 동일성은 검증 후만 통합 |
| MSM interval | 0/2/4/6/8/10%–30% | 0–30 기본 재사용 |
| test pool 분할 | 두dataset s1/2/3 AC/AS·실제subset크기 | s1 기본 재사용; 같은분할 rawmap으로Table13 AUPRO계산 가능 |
| backbone | DINO B16/B8; DINOv2 B14/L14; CLIP B32/B16/B16+240/L14/L14-336; Swin B4/L4, 두dataset | 11backbone중기본MVTec1개 있음. B16+ MVTec s1은 속도표와 공유 |
| MuSc·MuSc+ 확장 | 각각MVTec1/2/4/8/16/32/full 7조건 | 0-shot 두표공유. 두방법의reference사용방식이 달라 서로대체 불가 |
| benchmark | MuSc 두CLIPbackbone×s1/2/3, time/ms·maxGPU·AC/AUPRO | Table6의Large3조건은Table13과공유. 속도benchmark와전체성능평가는 pool범위가다름 |

GPU는 이 PC의 RTX 5080을 사용한다. 모델로딩·metric·파일저장을 포함한 기존 wall time은 논문의 추론시간 칸에 넣지 않는다. 소수점 정밀도는 원본 결과에 맞춰 표시한다.

## Table 1–18 대조

| Table | 보유/연결한 결과 | 남은 일 |
|---|---|---|
| 1 | WinCLIP 4행 7지표·APRIL-GAN 4행·ACR·RegAD4·GraphCore4·MuSc MVTec 있음 · PatchCore 해당 조건 검증 완료 | MuSc VisA 기본; WinCLIP4 반복 분산 보완 |
| 2 | CutPaste·NSA·GraphCore8·MuSc MVTec 있음; 구 PatchCore full AUROC CSV도 있음 · PatchCore 해당 조건 검증 완료 · RegAD확장 검증 완료 | IGD 중단 |
| 3 | MuSc MVTec r={1,3,5} 기본 있음 | 양 dataset의 6개 r조합 중 기본 중복 제외 나머지 |
| 4 | MuSc MVTec 30%+mean 기본 있음 | 양 dataset 6개 전략 중 기본 중복 제외 나머지; min과30%+min은 수학적/코드 동등성 검증 후만 재사용 |
| 5 | MuSc MVTec RsCIN 전후 모두 기존 점수로 재계산·검증 완료 | VisA 기본 점수와 RsCIN 전후 |
| 6 | 기존 실행 wall time은 loading/metric 포함이므로 직접 대입 불가 | 최대200장 pool 조건, s=1/2/3, 추론시간과최대메모리 |
| 7 | MVTec s=1 성능/크기 있음 | MVTec s=2/3; VisA s=1/2/3; 부분집합 예측 합쳐 지표 계산 |
| 8 | ViT-L-14-336 MVTec 기본 있음 | 다른 backbone10개×MVTec/VisA + 기본 backbone VisA; B16+ MVTec는 Table13과 공유 |
| 9 | MVTec r={1,3,5} 기본 있음 | {1},{1,3}은Table3과공유; 추가큰r 3조합×2dataset; VisA기본공유 |
| 10 | MVTec 0%–30% 기본 있음 | 하한2/4/6/8/10%×2dataset; VisA기본공유 |
| 11 | APRIL-GAN MVTec/VisA0/4·DRAEM·RegAD2/4/8 RsCIN 전후 있음; STPM 공개학생 모델 자체 RsCIN 전후 검증 완료; DSR 공개 모델 자체 RsCIN 전후 검증 완료 · PatchCore 해당 조건 검증 완료 · SPADE MVTec RsCIN검증 완료 | DSR 신규 실행 불필요; STPM 완료 |
| 12 | WinCLIP5category+평균+MVTec전체 연결 완료; APRIL-GAN5category 있음 | APRIL-GAN5category평균은 이번 연결수정으로 채움; MuSc VisA2category 필요 |
| 13 | APRIL-GAN 두 backbone benchmark 있음; WinCLIP성능 연결; 네조건 자체 benchmark 검증됨; RegAD4성능 있음 | RegAD4속도/메모리; WinCLIP 타방법과 측정범위 대조; MuSc 두backbone×s1/2/3 속도/메모리 및AC/AUPRO |
| 14 | VT-ADL·P-SVDD·SPADE·PyramidFlow·APRIL-GAN0/4 BTAD 있음 · PatchCore 해당 조건 검증 완료 · PaDiM 검증 완료 · RegAD확장 검증 완료 | MuSc0 BTAD 없음 |
| 15 | MuSc MVTec0 기본 있음 | MuSc 1/2/4/8/16/32/full 정상reference추가 7조건 |
| 16 | 0-shot(MuSc)는Table15기본과공유 | MuSc+ 1/2/4/8/16/32/full reference전용score 7조건 |
| 17 | MuSc MVTec15category 7지표 모두 있음 | 신규실행 없음; 원문/공식코드 설정차이 명시 |
| 18 | MuSc VisA category 지표 없음 | VisA12category 7지표; Table1/5/7/8/12 기본결과와공유 |

## 그림까지 재현할 때 필요한 자료

- Figure1/2: 방법 비교/구조 설명 그림; 신규 성능측정이 아니라 구조를 정확히 다시 그리는 작업.
- Figure3/9: r별 이상지도. LNAMD 실행에서 선택이미지 rawmap을 함께 보존한다.
- Figure4/10: MSM 평균 전 **다른 이미지별 최소거리 분포**와 정상/이상patch 라벨. 최종 AUROC나 최종map만으로는 복구할 수 없다.
- Figure5: RsCIN전후 정상/이상 image점수 히스토그램과 이웃예시. MVTec 점수는 보유, 이웃예시는 class feature/거리 필요.
- Figure6: MuSc/APRIL-GAN0/4 같은이미지 비교지도. 기존metric CSV만으로 생성할 수 없다.
- Figure7: MSM선택비율10/30/50/70/90/100%의 두dataset AC/AS. 30%기본만 재사용.
- Figure8: RsCIN single-window k=2…9, full-window 및w/o/full RsCIN의 두dataset F1-max. feature와원시image점수가 있으면 GPU추론 재실행 없이 후처리 가능.
- Figure11–16: VisA12·MVTec15 전체category의 정성적 예시. dataset이미지/GT/예측map 및 선정manifest 필요.

확인한 `/home/test/musc_results`에는 npy/npz 원시map이 발견되지 않았다(최대깊이3 확인). 보존CSV의 image점수는 재사용 가능하지만, 정성지도와중간patch거리분포는 별도 확인/재생성이 필요하다. 이번 감사는 전체 디스크의 모든 과거파일을 조사했다는 주장이 아니다.

## 처리 순서

1. **기존자료 연결·재계산:** WinCLIP 표 연결과MuSc MVTec RsCIN전후 재계산을 이번에 완료했다. 기존결과를 새실행 목록에 넣지 않는다.
2. **PatchCore 완료:** 4조건/105개 category run 검증 완료 및 표 반영. IGD는 보류.
3. **비교군 빈 조건:** STPM/DSR/SPADE MVTec, PaDiM BTAD, RegAD32/BTAD4. 각method의 공개코드/가중치/학습조건 확정 후 실행한다. 이번 감사 요청만으로 새학습을 시작하지 않았다.
4. **MuSc 기본빈칸:** VisA·BTAD. 원시feature·지도·전후image점수·manifest를 보존해서 중복추론을 줄인다.
5. **제거실험·확장·backbone:** 기본feature를 재사용할 수 있는 조건부터 실행하고 필요한중간거리도 함께 저장한다.
6. **반복·benchmark·그림:** WinCLIP4 반복수/표준편차정의 먼저 확정; 같은PC에서 측정범위 통일; 그림7/8 sweep 및정성결과 생성.

완료라고 표시해도 공개checkpoint 평가(DRAEM/RegAD 등), 비공식구현(WinCLIP/SPADE 등), backbone차이(GraphCore), precision/배치변경 및원문RsCIN창과공식기본값 차이는 보고서에 유지한다. 논문 수치를 내PC 측정값으로 쓰지 않는다. 공식학습까지 재현하는 것은 표용자체평가 수치확보와 별도 작업이다.

## 근거

- [수정한 전체 비교표](MuSc_all_reproduction_tables.md)
- [WinCLIP 기존 보고서](../../method7/markdown/CVPR23_WinCLIP_reproduction_report.md)
- [34개 조건·근거 SHA-256 CSV](../source/results/remaining_work_audit_20261010/comparator_conditions.csv)
- [18개 표 요구사항 CSV](../source/results/remaining_work_audit_20261010/table_requirements.csv)
- [감사 provenance](../source/results/remaining_work_audit_20261010/provenance.json)
- [MuSc RsCIN 기존점수 재계산](../source/results/musc_rscin_existing_20261010/mean_metrics.csv)
- [원문 PDF](../paper/ICLR24_MuSc_Zero_Shot_Industrial_Anomaly_Classification_and_Segmentation_with_Mutual_Scoring_of_the_Unlabeled_Images.pdf)


SPADE MVTec 최신 상태: **완료·검증 통과**. [실행 보고서](../../method20/markdown/SPADE_mvtec_rscin_execution.md). 비교군 상태는 [조건 CSV](../source/results/remaining_work_audit_20261010/comparator_conditions.csv)와 감사 provenance가 최신 기준이다.


PaDiM BTAD 최신 상태: **완료·검증 통과**. [실행 보고서](../../method23/markdown/PaDiM_btad_execution.md).


RegAD MVTec32·BTAD4 최신 상태: **완료·검증 통과**. 추가local학습없음. [조건·실행 보고서](../../method16/markdown/RegAD_mvtec32_btad4_execution.md).


STPM MVTec 최신 상태: **완료·검증 통과**. 저자 공개학생 모델 자체 평가이며 추가 학습은 수행하지 않았다. [실행·RsCIN 보고서](../../method24/markdown/STPM_mvtec_rscin_execution.md).


WinCLIP 네조건 자체 속도·메모리 측정 완료. 모든category 선택입력 반복, batch1/32 분리. [측정 정의·결과](../../method7/markdown/WinCLIP_speed_memory_20261011.md).


DSR MVTec 최신 상태: **완료·검증 통과**. 저자 공개 모델 자체 평가이며 추가 학습 없음. [실행·RsCIN 보고서](../../method25/markdown/DSR_mvtec_rscin_execution.md).


IGD 최신 상태(2026-10-11): **사용자 요청으로 실행 취소**. 학습 프로세스 종료 확인, 새 bottle local recovery·optimizer·로그 보존, 전체 평가 결과 없음. 자동 재개하지 않는다. [실행 기록](../../method14/markdown/IGD_mvtec_execution.md).
