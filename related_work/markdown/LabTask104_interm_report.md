# LabTask #104 진행 현황 및 LabTask #105 PDF 완성 계획

- 목표: MuSc 논문의 Experiments 부분을 작성하고, Method까지 구현된 LaTeX 원고와 통합해 PDF를 완성한다.

## 1. 현재까지 완료된 작업

### LaTeX 원고

- MuSc 논문의 Method 부분까지 LaTeX 구현 완료

### Experiments 작성을 위한 로컬 빌드 결과

다음 15개 기법에서 내 PC로 측정한 결과값을 확보했다. 일부 기법은 특정 데이터셋·shot 조건의 결과만 확보한 상태이며, MuSc 비교표에 필요한 모든 실험 조건을 완료했다는 의미는 아니다. 남은 조건은 2절에 정리한다.

| 번호 | 기법 | 상태 |
|---|---|---|
| 1 | MuSc | 결과값 확보 완료 |
| 2 | WinCLIP | 결과값 확보 완료 |
| 3 | APRIL-GAN | 결과값 확보 완료 |
| 4 | ACR | 결과값 확보 완료 |
| 5 | RegAD | 결과값 확보 완료 |
| 6 | PatchCore | MuSc 비교용 추가 실험 완료: MVTec full/4-shot, VisA·BTAD 4-shot, RsCIN 비교 |
| 7 | GraphCore | 결과값 확보 완료 |
| 8 | NSA | 결과값 확보 완료 |
| 9 | VT-ADL | 결과값 확보 완료 |
| 10 | P-SVDD | 결과값 확보 완료 |
| 11 | SPADE | 결과값 확보 완료 |
| 12 | PaDiM | 결과값 확보 완료 |
| 13 | DRAEM | 결과값 확보 완료 |
| 14 | CutPaste | 결과값 확보 완료 |
| 15 | PyramidFlow | 결과값 확보 완료 |

## 2. 남은 작업

### 새 결과 확보 필요

| 방법 | 실험 조건 | 필요한 결과 |
|---|---|---|
| STPM | MVTec full-shot | 학습·평가 결과, RsCIN 적용 전·후 지표 |
| DSR | MVTec full-shot | 학습·평가 결과, RsCIN 적용 전·후 Image AUROC, F1-max, AP |
| SPADE | MVTec full-shot | 평가 결과, RsCIN 적용 전·후 지표. BTAD는 완료 |
| PaDiM | BTAD full-shot | 해당 조건의 평가 결과 |
| RegAD | MVTec 32-shot·BTAD 4-shot | 해당 조건의 학습·평가 결과 |
| IGD | MVTec 전체 학습·평가 | 신규 결과 확보 필요 |

### 기존 결과 보완 필요

| 방법 | 보완 내용 |
|---|---|
| WinCLIP | MVTec·VisA 4-shot 반복 실험의 평균·표준편차, 속도 측정 |
| RegAD | 속도 측정 |

## 3. 10월 12일까지의 작업 계획

| 기간 | 주요 작업 | 목표 산출물 |
|---|---|---|
| 10월 8일–12일 | 남은 기법들을 실행하고 결과를 확보한다. 결과가 전부 확보되면 LaTeX 최종본을 PDF로 만들어 제출한다. | 남은 기법들의 결과값, 전체 결과 확보 후 제출용 최종 PDF |
| 10월 10일–11일 | QueCo를 공통 프레임워크에 포팅한 후, QueCo 코드를 분석한다. | 공통 프레임워크에 포팅한 QueCo 코드 및 코드 분석 내용 |
