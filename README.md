# TUK AI-HC Lab 학생 Repository 가이드

이 가이드는 학생 연구 repository를 어떻게 구성하고, 주간 미팅 brief를 어떻게 작성하고, 피드백을 어떻게 다음 연구로 연결할지 정의합니다. 이 Guide는 학생의 현재 숙련도와 관계없이 지도 교수와 협업하며 연구 과정을 논문으로 발전시킬 수 있도록 돕는 작성·검토 harness를 만들어 가는 중입니다. 즉, 학생 숙련도에 맞춰 계속해서 업데이트할 예정입니다.

목표는 검토 가능한 연구 흔적을 남기는 것입니다. 지금 어떤 문제를 보고 있는지, 현재 원인 가설이 무엇인지, 어떤 기존 방법 후보를 비교했는지, 왜 한 방향을 선택했는지, 어떤 실험으로 검증했는지, 그 결과가 가설을 어떻게 바꾸는지를 남겨야 합니다.

검토 요청은 연구 협업의 기본 에티켓을 갖춘 상태에서만 가능합니다. 현 GitHub 기반 검토는 기존 PPT 기반 검토보다 코드, 결과, claim, evidence까지 훨씬 세밀한 맥락을 확인할 수 있다는 장점이 있습니다. 학생이 이 GitHub 형식에 맞춰 정리하더라도, 지도교수가 여러 업무로 바쁜 와중에 매주 시간을 내어 검토하고 지도하는 것은 상당한 지도 시간을 배정하는 일입니다. 그 시간은 학생이 임의로 만든 자료의 맥락을 대신 추적하고 정리해 주기 위한 시간이 아닙니다. 따라서 Github 절차를 우회한 채, 학생이 임의로 구성한 자료에 대한 검토 요청은 수용하지 않습니다.

조직의 정해진 시스템을 성실히 따르는 태도는, 연구실과 회사 모두에서 중요한 역량입니다. 이는 단순한 순응이 아니라, 서로의 일을 검토하고 신뢰할 수 있도록 약속된 협업 절차를 지키는 태도입니다. GitHub 절차는 형식적인 제출 규칙이 아니라, 연구자로서 자신의 판단과 근거를 검토 가능한 형태로 남기는 기본 협업 방식입니다. 우리 연구실에서는 이 GitHub 절차를 잘 따르는 것에서부터 그 습관을 만들어 가겠습니다.

---

## 1. 핵심 원칙

### 1.1 AI를 사용하되, 판단을 외주화하지 않는다

Claude, ChatGPT, Cursor, Codex, Gemini CLI 같은 AI 도구를 논문 읽기, 코드 작성, 디버깅, 실험 설계, 노트 정리에 적극적으로 사용하세요.

하지만 AI가 쓴 원문을 그대로 weekly brief로 제출하지 마세요. push하기 전에 반드시 본인이 확인하고, 고치고, 본인의 이해와 판단이 드러나게 다시 써야 합니다.

문장이 유창해도 본인의 가설, 근거 경로, 선택 이유, 결과 해석이 보이지 않으면 검토할 수 없습니다.

### 1.2 연구자는 작은 진행을 계속 남긴다

연구는 미팅 직전에 한 번에 정리하는 과제가 아닙니다. 의미 있는 진행이 생길 때마다 작게 commit하고 push하세요. 한 줄짜리 결과, 실패한 실행, 바뀐 가설, 새로 읽은 논문 요약도 다음 판단에 필요하면 연구 흔적입니다.

이 습관의 목적은 감시가 아니라 연구자 정체성 형성입니다. 매번 작은 판단과 근거를 남기는 사람은 연구를 하고 있는 것이고, 미팅 직전에 AI로 문서를 몰아 쓰는 사람은 연구 과정을 보이지 않게 만든 것입니다.

이전 미팅이 끝나면 다음 주 `meetings/YYYY-Www_brief.md` skeleton을 즉시 만들고 push하세요. 그 뒤에는 연구가 진행될 때마다 brief를 업데이트합니다.

지도교수는 필요하다고 판단하면 언제든 최신 push를 기준으로 피드백할 수 있습니다. 피드백은 지도교수 또는 AI가 읽은 시점에 repository에 보이는 내용을 기준으로 합니다.

### 1.3 push 부담을 줄이는 도구를 사용한다

작게 자주 push하는 부담은 도구로 줄여야 합니다. Claude Code, Codex, Gemini CLI 같은 도구에 아래 자동화를 붙이면 좋습니다.

- 한 명령으로 변경 분석, commit message 생성, push까지 실행하는 skill 또는 script.
- 하루 끝, 작업 종료 시점, 또는 정해진 시간에 자동 commit/push하는 cron 또는 scheduler.
- 실험 실행 후 result path와 command log를 brief에 자동으로 붙이는 helper script.

자동화는 연구를 대신하는 것이 아니라 연구 흔적을 덜 잊게 만드는 장치입니다.

### 1.4 모든 판단에는 근거 경로가 필요하다

파일을 많이 올리는 것이 목표가 아닙니다. 판단과 근거가 연결되어야 합니다.

나쁜 예:

```text
baseline을 돌렸다.
```

좋은 예:

```text
baseline 실행 결과, class A에서 AUROC가 낮게 나왔다.
- code: `method1/source/run_baseline.sh`
- command log: `method1/source/logs/baseline_20260504.log`
- raw result: `method1/source/results/baseline_20260504.csv`
- analysis: `method1/markdown/baseline_analysis.md`
```

숫자, 표, plot, 논문 claim, 실패 사례는 모두 다시 확인 가능한 경로가 있어야 합니다.

### 1.5 지도교수에게 판단을 외주화하지 않는다

지도교수의 역할은 학생이 충분하고 모호함 없이 전달한 맥락을 통해 잠정 판단이 옳은지, 근거가 충분한지, 다음 실험 방향이 맞는지 코멘트하는 것입니다.

따라서 brief에는 최소한 다음 흐름이 보여야 합니다.

1. 현재 문제의 원인 가설이 무엇인가.
2. 그 가설을 해결할 수 있는 기존 방법 후보는 무엇인가.
3. 그 후보들 중 무엇을 선택했고, 왜 다른 후보보다 나은가.
4. 선택한 방향으로 어떤 실험을 했는가.
5. 결과가 가설을 지지했는가, 약화했는가, 반박했는가.
6. 그래서 다음 판단은 무엇인가.

다수의 가설과 검증이 쌓이기도 전에 맥락 없이 "무엇을 하면 좋을까요?", "publishable한가요?", "이 방향이 맞나요?"라고 묻는 것은 검토 가능한 질문이 아닙니다.

---

## 2. Guide 용어

이 문서에서 사용하는 핵심 용어는 아래 뜻으로 고정합니다.

- *artifact*: 연구 중 생긴 구체적인 파일 또는 출력물입니다. code, script, csv, plot, table, log, paper note, meeting brief가 모두 artifact가 될 수 있습니다. 예: `method1/source/run_baseline.sh`, `method1/source/results/baseline.csv`, `method1/markdown/paper_summary.md`.
- *evidence*: 어떤 주장이나 판단을 지지하거나 반박하는 데 사용된 artifact입니다. 파일 종류가 아니라 역할입니다. 같은 csv도 단순 저장물이면 artifact이고, 특정 가설 검증에 연결되면 evidence입니다. 예: "class A에서 AUROC가 낮다"는 판단을 확인하기 위해 연결한 `method1/source/results/auroc_by_class.csv`.
- *claim*: evidence가 지지하는 문장입니다. 방법을 사용했다는 사실이 아니라, 그 방법이나 분석이 보여주는 결론입니다. 예: "method X를 사용했다"는 방법 설명이고, "method X가 condition Y에서 AUROC를 3.2p 개선했다"는 claim입니다.
- *limitation*: 기존 방법, 실험 설정, 데이터, metric, 해석에서 발견한 구체적 한계입니다. 단순한 느낌이 아니라 evidence나 관련 연구와 연결되어야 합니다. 예: "baseline이 per-class evaluation에서는 높지만 cross-class score comparison에서는 category별 score scale이 달라진다."
- *argument*: 문제, 원인 가설, 기존 방법의 한계, 선택한 해결 방향, evidence가 서로 어떻게 연결되는지 설명하는 논리입니다. 예: "cross-class anomaly detection에서 category별 score scale 차이가 관찰되었고, 이 차이가 baseline의 limitation이라면 calibration 실험으로 AUROC variance가 줄어야 한다."

---

## 3. Repository 구조

학생 repository는 아래 구조를 기본으로 합니다.

```text
your-repo/
├── README.md
├── .gitignore
├── meetings/
│   ├── 2026-W18_brief.md
│   └── ...
├── related_work/
│   ├── paper/
│   └── markdown/
├── method1/
│   ├── source/
│   ├── markdown/
│   └── paper/
├── method2/
│   ├── source/
│   ├── markdown/
│   └── paper/
└── ...
```

| Path                         | 넣을 것                                          | 넣지 말 것                             |
| ---------------------------- | --------------------------------------------- | ---------------------------------- |
| `README.md`                  | 프로젝트 개요, 현재 연구 방향, method 목록, 주요 brief 링크     | 매주 상세 보고서                          |
| `meetings/YYYY-Www_brief.md` | 주간 검토 문서, 이번 주 연구 논리, 피드백 반영, evidence map    | 발표 대본, 파일 목록만 있는 보고                |
| `methodN/source/`            | 재현 가능한 코드, 실행 script, config, result csv/json | 대용량 dataset, weight, cache, 임시 log |
| `methodN/markdown/`          | primary paper 요약, 실험 보고, 분석 노트, 결과 해석         | PDF 원문                             |
| `methodN/paper/`             | 해당 method의 primary paper PDF 1편               | summary note, 관련 연구 PDF 묶음         |
| `related_work/paper/`        | 비교나 배경으로 쓰는 참고 논문 PDF                         | primary method paper               |
| `related_work/markdown/`     | 관련 연구 요약, 비교표, 후보 방법 정리                       | primary method 실험 보고               |

`methodN/`은 자기완결적인 학습 또는 실험 단위입니다. 보통 논문 1편 또는 하나의 실험 방향을 한 method 폴더로 둡니다.

기존 논문 method 설명은 `Related Work`나 `methodN/markdown/`에 둡니다. weekly brief의 `Method` 섹션은 본인의 제안 방법, 변형, 분석 절차, protocol change, 선택한 실험 절차를 쓰는 곳입니다.

### 3.1 PDF 파일명 규칙

논문 PDF 파일명은 아래 형식이 필수입니다.

```text
VenueYY_Abbrev_Full_Title.pdf
```

규칙:

- ASCII, underscore, no spaces.
- 학회나 저널에 출판된 버전이 있으면 arXiv보다 proceedings 또는 official version을 저장합니다.
- arXiv PDF는 출판 버전을 찾을 수 없을 때만 사용합니다.

예:

```text
CVPR22_RD_Anomaly_Detection_via_Reverse_Distillation.pdf
CCS24_PLeak_Prompt_Leaking_Attacks.pdf
KDD25_MGTBench2_Generalization_and_Adaptation_Ability.pdf
```

---

## 4. Weekly Brief의 목적

weekly brief는 일기, 파일 묶음, 발표 대본, 지도교수에게 넘기는 질문 목록이 아닙니다.

weekly brief는 압축된 연구 논리입니다. 읽는 사람이 brief만 보고 이번 주의 현재 원인 가설, 해결 후보, 선택한 방향, 실험, 결과, 다음 판단을 이해할 수 있어야 합니다.

미팅에서 발표할 내용도 이 brief를 기반으로 해야 합니다. brief에 없는 별도 slide story를 만들지 마세요. 미팅에서 보여줄 내용은 이미 brief에 있고, 그 근거 artifact 경로가 연결되어 있어야 합니다.

---

## 5. 필수 Weekly Brief 템플릿

`meetings/YYYY-Www_brief.md`에는 아래 구조를 사용합니다. 섹션을 삭제하지 마세요. 채울 수 없는 섹션이 있다면 그 아래에 왜 채울 수 없는지 적으세요.

````markdown
# 2026-W18 미팅 Brief - <짧은 연구 제목>

## 0. 지난주 피드백 반영과 이번 주 변화

### 지난주 상태
- 지난주에는 프로젝트가 어디까지 와 있었는가:
- 지난주 핵심 한계 또는 미결정 사항:

### 피드백 반영
| 지난주 피드백 | 처리 상태 | 근거 경로 | 메모 |
|---|---|---|---|
| `((...))` | 완료 / 부분 / 미완료 | `path` |  |
| `<<용어>>` | 정의 추가 / 미완료 | `path` |  |

### 이번 주 변화
- 새로 얻은 결과:
- 새로 확인한 실패:
- 바뀐 원인 가설 또는 판단:

### 현재 연구 단계
- 현재 phase: Phase 1 / Phase 2 / Phase 3 / Phase 4 / Phase 5
- 이 phase라고 판단한 이유:
- 다음 phase로 가기 위해 필요한 evidence:

### Evidence Map
| 판단 / claim / 질문 | 짧은 요약 | 근거 경로 | Raw data / Plot | 이 근거가 결정하는 것 |
|---|---|---|---|---|
|  |  | `path` | `path` |  |

## 1. 서론

### 1.1 연구 배경과 중요성
- 이 연구 문제가 속한 큰 영역은 무엇인가:
- 왜 이 문제가 실제로 중요한가:
- 최근 어떤 흐름이나 접근이 이 문제에서 중요해졌는가:

### 1.2 기존 접근의 큰 계열과 한계
- 기존 접근 A:
- 접근 A가 유용한 이유:
- 접근 A의 한계:
- 기존 접근 B, 필요하면:
- 접근 B가 유용한 이유:
- 접근 B의 한계:

### 1.3 우리가 집중하는 접근과 남은 문제
- 이번 연구가 집중하는 접근:
- 이 접근이 기존 대안보다 나은 이유:
- 그러나 아직 해결되지 않은 challenge 1:
- challenge 1이 실제 문제인 evidence:
- challenge 2:
- challenge 2가 실제 문제인 evidence:

### 1.4 원인 가설
- 현재 관찰한 limitation 또는 failure:
- 원인 가설 H1:
- 왜 H1이 그럴듯한가:
- H1이 맞다면 관찰되어야 하는 현상:

### 1.5 이번 주 선택한 방향
- 이번 주 해결 후보:
- 다른 후보보다 이것을 선택한 이유:
- 이 선택이 H1을 어떻게 검증하는가:

### 1.6 이번 주 기여 또는 검토 초점
- 이번 주 brief가 보여주려는 핵심 판단:
- 이 판단을 지지하는 evidence 위치:

## 2. 사전 지식과 문제 정의

### 배경
- 이번 주 결과를 이해하기 위해 필요한 배경:
- 이전 결과 또는 논문 맥락:

### 기술 용어
- *기술 용어* [1]: 이 brief에서의 뜻. 작은 예시: ...
- *metric 이름* [2]: 무엇을 측정하는지, 단위, 값이 높거나 낮을 때의 의미. 작은 숫자 예시: ...

### 표기법
| 표기법 | 설명 |
|---|---|
| `D_s` | source dataset |
| `f_\theta` | model, detector, or scoring function |

### 정의
정의 1. <local operational definition>.
단위, 범위, formula, decision rule, 왜 필요한지 적는다.

### 문제 정의
문제 1. <문제 이름>.
<입력과 조건>이 주어졌을 때, 우리의 과제는 <해야 할 일>이다. 구체적으로 우리는 <제약 조건> 아래에서 <출력>을 예측 / 추정 / 결정한다. 목표는 <성공 기준>을 만족하는 것이다.

## 3. 관련 연구

### 3.1 현재 문제를 푸는 큰 연구 흐름
- 큰 연구 흐름 1:
  - 이 흐름이 풀려는 문제:
  - 대표 논문과 핵심 아이디어:
  - 이 흐름의 한계:
  - 현재 내 문제와의 연결:
- 큰 연구 흐름 2:
  - 이 흐름이 풀려는 문제:
  - 대표 논문과 핵심 아이디어:
  - 이 흐름의 한계:
  - 현재 내 문제와의 연결:

### 3.2 현재 원인 가설에 대응하는 관련 연구
- 원인 가설 H1과 관련된 prior work:
  - H1을 지지하거나 반박하는 근거:
  - 이 prior work가 제안하는 해결 방향:
  - 내 문제에 그대로 적용하기 어려운 이유:
- 원인 가설 H2, 필요하면:
  - H2를 지지하거나 반박하는 근거:
  - 이 prior work가 제안하는 해결 방향:
  - 내 문제에 그대로 적용하기 어려운 이유:

### 3.3 이번 주 해결 후보 요약
- 후보 A:
  - 관련 연구 근거:
  - 왜 가능성이 있는가:
  - 왜 실패할 수 있는가:
  - 이번 주 판단: 선택 / 제외 / 보류
- 후보 B:
  - 관련 연구 근거:
  - 왜 가능성이 있는가:
  - 왜 실패할 수 있는가:
  - 이번 주 판단: 선택 / 제외 / 보류

## 4. 선택한 방법 또는 절차
- 이번 주 선택한 후보:
- 다른 후보보다 이것을 선택한 이유:
- 기존 연구와의 차이:
- 구현 또는 protocol 변경:
- 이 선택이 현재 원인 가설을 어떻게 검증하는가:

## 5. 실험

### 비교 기준
- 비교 기준:

### 평가 지표
- metric과 해석 방법:

### 데이터셋
- dataset, sample unit, split, protocol:

### 실험 1: <짧은 이름>

#### 질문
- 이 실험이 확인하는 질문:

#### 가설
- H1:
- H2, 필요하면:

#### 설정
- Dataset:
- Baseline:
- Method / condition:
- Metric:
- 근거 경로:
  - code: `path`
  - command/log: `path`
  - result: `path`

#### 기대 결과
- 가설이 맞다면 어떤 결과가 나와야 하는가:

#### 실제 결과
| 지표 | 기대 | 실제 | 해석 | Raw Path |
|---|---|---|---|---|
|  |  |  |  | `path` |

#### 해석
- 결과가 가설을 지지 / 약화 / 반박 / 아직 판단 불가 중 어디에 해당하는가:

## 6. 논의
- 지금 주장할 수 있는 것:
- 아직 주장할 수 없는 것:
- 이번 결과가 원인 가설에 대해 알려준 것:
- 남은 불확실성:
- 다음 검증:

## 7. 참고문헌
[1] Author Last Name, First Name, et al. "Paper Title." Conference or Journal, Year.
[2] Author Last Name, First Name, et al. "Paper Title." Conference or Journal, Year.

## 8. 이번 주 주요 질문 1개
- 질문:
- 내가 현재 판단한 답:
- 그 판단의 근거:
- 왜 이 판단이 틀렸을 수 있는가:
- 교수님이 판단해주면 다음 주 무엇이 바뀌는가:
````

---

## 6. 연구 진행 단계

현재 phase에 따라 그 주 brief에서 어느 부분이 성숙해야 하는지 결정합니다. 별도 phase 보고서를 만들지 말고, 일반 brief 섹션 안에서 phase가 드러나게 하세요.

| Phase | Name | 목적 | 최소 산출물 |
|---|---|---|---|
| Phase 1 | Problem Grounding | 문제, 용어, 평가 단위, metric, dataset/protocol, anchor literature를 고정한다. | 문제 정의, 용어, notation, anchor paper metadata, input/output/unit/constraint/success criterion. |
| Phase 2 | Evidence Reproduction | 기존 claim을 만드는 evidence를 재현하거나 재구성한다. | code path, command/log path, raw result path, original claim과의 비교. |
| Phase 3 | Limitation Identification | 기존 방법의 구체적 한계, mismatch, instability, cost, failure를 찾는다. | limitation analysis, diagnostic table/figure, failure case, related-work comparison. |
| Phase 4 | Hypothesis-Driven Extension | 확인한 limitation을 겨냥한 focused change, ablation, metric, protocol, analysis를 검증한다. | 질문, 가설, 설정, 기대 결과, 실제 결과, 해석, raw path. |
| Phase 5 | Claim Consolidation | 누적 evidence를 논문 수준 contribution으로 정리한다. | paper-level claim, evidence map, main table/figure 후보, supporting diagnostics, scope limit. |

### 6.1 Phase별 성숙해야 하는 brief 섹션

| Phase | 성숙해야 하는 섹션 | 지도교수가 확인할 수 있어야 하는 것 |
|---|---|---|
| Phase 1. Problem Grounding | `1. 서론`, `2. 사전 지식과 문제 정의`, `3. 관련 연구`, `7. 참고문헌` | 학생이 문제 설정, 용어, 평가 단위, metric, dataset/protocol, anchor literature를 이해했는가. |
| Phase 2. Evidence Reproduction | `0. Evidence Map`, `3. 관련 연구`, `5. 실험` | 기존 claim과 재현 가능한 evidence component가 code/log/raw path로 연결되는가. |
| Phase 3. Limitation Identification | `1. 서론`, `3. 관련 연구`, `5. 실험`, `6. 논의` | limitation이 직관이 아니라 evidence와 prior work에 근거하는가. |
| Phase 4. Hypothesis-Driven Extension | `4. 선택한 방법 또는 절차`, `5. 실험`, `6. 논의`, `8. 이번 주 주요 질문 1개` | 초점이 분명한 가설과 그것을 검증하는 변화, ablation, metric, protocol, analysis가 있는가. |
| Phase 5. Claim Consolidation | `0. Evidence Map`, `5. 실험`, `6. 논의`, `7. 참고문헌` | 논문 수준 claim을 table/figure 후보, ablation, diagnostics, limitation으로 지지할 수 있는가. |

---

## 7. 작성 규칙

### 7.1 한 주의 검토 초점을 하나로 좁힌다

모든 phase에 완성된 claim이 있어야 하는 것은 아닙니다. Phase 1에서는 문제 정의나 용어 정리가 핵심일 수 있고, Phase 2에서는 재현 evidence가 핵심일 수 있습니다.

하지만 한 주의 brief에는 검토 초점이 하나 있어야 합니다. 여러 이야기를 나열하지 말고, 이번 주에 다음 연구 결정을 바꾸는 가장 중요한 판단 하나를 중심으로 쓰세요.

### 7.2 Argument first, artifact second

파일 목록으로 시작하지 마세요. 먼저 문제, 원인 가설, 후보 방법, 선택 이유, 실험, 결과, 판단을 설명하세요. 그 다음 artifact를 evidence로 연결하세요.

### 7.3 Advisor task list 금지

brief에 "다음에 무엇을 할까요?", "이거 publishable한가요?", "이 방향이 맞나요?" 같은 open-ended question을 나열하지 마세요.

먼저 원인분석, 가설, 기존 방법 후보, 선택 이유, 실험과 검증 결과를 통해 본인의 잠정 판단을 보여야 합니다. 지도교수는 그 판단의 옳고 그름, 근거의 충분성, 다음 실험 우선순위에 코멘트합니다.

### 7.4 주요 질문은 맨 마지막에 1개만 둔다

질문은 brief 맨 마지막 `## 8. 이번 주 주요 질문 1개`에 하나만 씁니다.

질문은 반드시 아래 조건을 만족해야 합니다.

1. 한 문장으로 명확하게 쓸 수 있다.
2. 현재 본인의 잠정 판단과 근거가 있다.
3. 교수님 판단에 따라 다음 주 실험이나 연구 방향이 실제로 바뀐다.

작은 구현 질문, 용어 검색, 코드 오류, 단순 정리 여부는 미팅 질문이 아닙니다. 학생이 직접 처리하고, 필요하면 body의 caveat로 한 줄만 남기세요.

### 7.5 근거 없는 숫자 금지

모든 결과 숫자에는 raw path가 필요합니다. csv, json, notebook, log, command output, figure-data path가 없는 숫자는 review할 수 없습니다.

### 7.6 계획만 있는 side document 금지

paper table, related-work table, experiment report로 직접 재사용되는 문서가 아니라면 별도 planning note를 만들지 마세요. 중요한 정보라면 brief에 요약하고 artifact path를 연결하세요.

### 7.7 이전 피드백은 다음 brief에서 보여야 한다

매주 brief의 `0. 지난주 피드백 반영과 이번 주 변화`에는 지난주 피드백을 어떻게 처리했는지 남겨야 합니다. 처리하지 못했다면 미완료라고 쓰고 이유를 적으세요.

### 7.8 Citation rule

weekly brief 본문에서 논문을 언급할 때는 `[1]`, `[2]`처럼 bracket number를 사용하세요. 번호는 처음 등장한 순서로 붙입니다.

`## 7. 참고문헌`에는 Google Scholar의 MLA 형식을 복사해 넣는 것을 기본으로 합니다. 논문 reference를 URL만으로 대체하지 마세요.

`## 8. 이번 주 주요 질문 1개`가 맨 마지막 섹션이므로 참고문헌은 그 바로 앞에 둡니다.

### 7.9 Presentation rule

주간 미팅 발표는 `meetings/YYYY-Www_brief.md`를 기반으로 해야 합니다. 별도 slide-like story를 만들지 마세요. 미팅에서 보여줄 내용은 brief에 들어 있고 evidence path가 연결되어 있어야 합니다.

### 7.10 중복 없이 컴팩트하게 쓴다

weekly brief는 길게 쓰는 문서가 아니라, 검토 가능한 연구 논리를 압축하는 문서입니다. 같은 내용을 여러 섹션에서 반복하지 마세요. 본문에는 문제, 원인 가설, 관련 연구, 선택 이유, 실험, 결과 해석, 다음 판단에 직접 필요한 내용만 남깁니다.

raw log, 긴 command output, 실패 기록, 세부 파일 목록은 brief 본문에 풀어 쓰지 마세요. 필요한 artifact는 repository에 남기되, brief에는 판단에 필요한 요약과 evidence path만 연결합니다. 본문 문장이나 표가 현재 판단, evidence path, 다음 결정에 기여하지 않으면 줄이거나 삭제하세요.

---

## 8. 사전 지식과 문제 정의 작성법

`Prerequisites`에 해당하는 섹션은 reader가 method, experiments, discussion을 읽기 전에 알아야 할 용어와 설정을 모두 정리해야 합니다.

### 8.1 기술 용어

표준 용어가 있으면 새 이름을 만들지 말고 논문에서 쓰는 용어를 사용하세요.

형식:

```markdown
- *기술 용어* [1]: 이 brief에서의 뜻. 작은 예시: ...
- *metric 이름* [2]: 무엇을 측정하는지, 단위, 값이 높거나 낮을 때의 의미. 작은 숫자 예시: ...
```

인용한 논문에서 쓰지 않는 용어라면 표준 용어처럼 쓰지 말고 `정의`에서 local operational definition으로 먼저 정의하세요.

### 8.2 표기법

수식, table, plot, 문제 정의에서 쓰는 모든 기호를 정의합니다.

```markdown
| Notation | Description |
|---|---|
| `D_s` | source dataset |
| `D_t` | target dataset |
| `f_\theta` | model, detector, or scoring function |
```

표기법 표는 짧게 유지합니다. 예시가 필요하면 `정의`나 `문제 정의` 본문에 둡니다.

### 8.3 정의

논문에서 이미 표준적으로 쓰는 용어가 아니라, 이번 연구에서 새로 정의한 local term, metric, decision rule, operational definition은 여기서 먼저 정의합니다.

형식:

```markdown
정의 1. <local 또는 operational term>.
이 용어를 명확히 정의한다. 필요하면 단위, 범위, formula, decision rule을 적는다. 이 정의가 이번 주 결과 해석에 왜 필요한지도 설명한다.
```

### 8.4 문제 정의 형식

문제를 입력, 출력, 단위, 제약, 성공 기준이 보이게 써야 합니다.

권장 형식:

```markdown
## 문제 정의

먼저 중요한 정의를 소개한다. 그 다음 이 연구가 다루는 문제를 문제 1로 정의한다.

정의 1. <핵심 object A>.
<핵심 object A>가 무엇인지 정의한다. 필요하면 notation을 함께 도입한다.

정의 2. <핵심 object B>.
입력에서 추출하거나 비교하는 단위가 무엇인지 정의한다.

정의 3. <보조 정보 또는 제약 조건>.
문제를 풀 때 사용하는 정보와 제약 조건을 정의한다.

문제 1. <문제 이름>.
<입력과 조건>이 주어졌을 때, 우리의 과제는 <해야 할 일>이다. 구체적으로 우리는 <제약 조건> 아래에서 <필요한 출력>을 결정 / 예측 / 추정한다. 목표는 <성공 기준 또는 최적화 목표>를 만족하는 것이다.
```

반드시 포함할 항목:

| 항목 | 확인 질문 |
|---|---|
| Input | 무엇이 주어지는가. Dataset, document, model output, image set, existing knowledge base 등. |
| Output | 무엇을 예측, 연결, 복원, 탐지, 개선해야 하는가. |
| Unit | sample, triple, patch, category 하나가 정확히 무엇인가. |
| Constraint | training data가 있는가. zero-shot, few-shot, full-shot인가. runtime, memory, black-box 제약은 무엇인가. |
| Success criterion | 어떤 metric이나 조건이면 문제가 해결되었다고 볼 수 있는가. |

좋은 problem definition은 이후 실험 가설과 직접 연결됩니다. 병목 분석을 했다면 "어디가 느린가"에서 끝내지 말고 "왜 병목이 생기는가"에 대한 원인 가설 2-3개를 세우고, 관련 논문과 검증 실험으로 연결하세요.

---

## 9. 실험 보고와 표/그림 작성법

### 9.1 실험은 네 부분을 포함한다

각 실험은 결과 숫자만 적지 말고 아래 네 부분을 포함해야 합니다.

1. 확인하려는 질문.
2. 가설.
3. 가설이 맞다면 기대되는 결과.
4. 실제 결과가 기대와 어떻게 달랐는지.

가설은 결과를 보기 전에 문서에 먼저 있어야 합니다. 자신의 데이터가 자신의 가설을 반박하는 것은 좋은 연구입니다. 중요한 것은 그 반박이 다음 판단을 어떻게 바꾸는지 명시하는 것입니다.

### 9.2 Plot 작성 규칙

plot은 누구나 같은 결론에 도달할 수 있게 작성해야 합니다.

- x-axis와 y-axis가 무엇을 뜻하는지, 단위가 무엇인지 적습니다.
- line, point, color, marker가 각각 무엇을 뜻하는지 legend나 caption에 적습니다.
- 점 하나가 sample, category, seed average 중 무엇인지 적습니다.
- figure만 올리지 말고, figure를 그린 raw table을 함께 commit하고 링크합니다.

권장:

```markdown
Figure 1은 category별 AUROC 변동을 보여준다.
- figure: `method1/source/figures/auroc_by_category.png`
- raw table: `method1/source/results/auroc_by_category.csv`
- one point: one category average over 3 seeds
```

### 9.3 Table 작성 규칙

table은 각 row와 column의 의미, 값의 단위, bold나 color의 의미를 설명해야 합니다.

- `%`, `ms`, `count`, `AUROC`처럼 단위를 명시합니다.
- `Delta`, `Mean`, `+/-` 같은 symbol은 한 번 정의합니다.
- bold cell이 best인지, selected condition인지, statistically meaningful difference인지 설명합니다.

body text, figure, raw table이 한곳에서 같은 결론을 가리켜야 합니다. plot 축이 없거나, raw table 없이 이미지만 있거나, column 설명 없는 table만 있으면 미팅 시간이 결과 해석이 아니라 복원에 쓰입니다.

---

## 10. 논문 조사 노트

매주 최소 1편의 관련 논문을 읽고 요약하세요. 그 논문이 main method가 되면 `methodN/paper/`와 `methodN/markdown/`에 두고, background 또는 comparison이면 `related_work/paper/`와 `related_work/markdown/`에 둡니다.

논문을 요약하거나 조사할 때는 summary 앞에 아래 metadata를 넣으세요. paper file이나 official page에서 확인할 수 없으면 생략하지 말고 `none` 또는 `unverified`라고 적습니다.

```markdown
## Paper Metadata

| Item | Content |
|---|---|
| Title |  |
| Authors |  |
| Conference / Journal |  |
| Year |  |
| Paper link |  |
| GitHub / Official code | URL, `none`, or `unverified` |
| Reason for investigation |  |
```

논문 조사는 검색 결과를 많이 모으는 일이 아닙니다. 가능하면 아래 우선순위를 따르세요.

1. premier international conference 또는 journal 논문.
   - conference 예: NeurIPS, ICML, ICLR, CVPR, ICCV, ECCV, WACV, ACL, EMNLP, NAACL, KDD, WWW, SIGIR, CHI, UIST, USENIX Security, IEEE S&P, ACM CCS, NDSS.
   - journal 예: Nature, Science, Nature Machine Intelligence, TPAMI, IJCV, JMLR, TACL, TOCHI, ESWA.
2. official GitHub code가 있는 논문.
3. 실험 protocol, dataset, metric이 재현 가능한 논문.
4. 현재 풀고 있는 문제의 큰 연구 흐름을 설명하거나, 현재 원인 가설 또는 후보 방법과 직접 연결되는 논문.

`Conference / Journal`, `Year`, `GitHub / Official code`, `Reason for investigation`이 없는 paper note는 미팅에서 다시 확인해야 하므로 불완전한 조사입니다.

---

## 11. 재현성 규칙

`source/` 안에서 실험을 실행할 때는 아래 규칙을 따릅니다.

1. `run_baseline.sh` 같은 script로 실행하세요. 터미널에 직접 친 command만으로 남기지 마세요.
2. 결과 숫자는 csv, json, 또는 검토 가능한 artifact로 저장하세요.
3. 보고서에 세 줄 source를 남기세요.

```markdown
- commit: `abc1234`
- sh: `method1/source/run_baseline.sh`
- result: `method1/source/results/baseline_20260504.csv`
```

4. 모든 figure에는 그 figure를 그린 raw table을 commit하고 링크하세요.

재현 불가능한 결과는 결과가 없는 것보다 더 위험합니다. 실제 결과인지 우연한 사고인지 구분할 수 없기 때문입니다.

external repo를 fork하거나 clone해서 사용했다면, 학생 repository에는 최소한 실행 방법, 사용 commit, 수정한 파일, 결과를 재현하는 script를 남겨야 합니다. weight나 dataset을 올릴 수 없다면 다운로드 방법이나 생성 방법을 적으세요.

---

## 12. `.gitignore` 규칙

검토에 필요한 source code, script, config, requirements, result csv/json은 commit하세요.

model weight, 대용량 dataset, log, checkpoint, cache, archive, virtual environment, secret은 commit하지 마세요.

권장 제외 항목:

```gitignore
*.pth
*.pt
*.ckpt
*.safetensors
*.bin
data/
datasets/
*.h5
*.npy
*.npz
*.zip
*.tar
*.tar.gz
*.7z
*.log
logs/
runs/
wandb/
outputs/
checkpoints/
__pycache__/
*.py[cod]
.ipynb_checkpoints/
.venv/
venv/
env/
node_modules/
.DS_Store
Thumbs.db
*.env
.env.*
```

result csv가 너무 커지면 Git LFS 도입을 상의하세요.

---

## 13. 피드백 방식

지도교수 피드백은 `Guide/feedback/<student-repo-id>/YYYY-Www.md`에 저장됩니다.

피드백 파일은 학생의 weekly brief를 복사한 뒤, 필요한 위치에 교수 피드백을 직접 추가하는 방식으로 작성됩니다. 기존의 별도 `What Went Well / Needs Improvement / Next Week Priorities` 양식은 사용하지 않습니다.

### 13.1 피드백 표기법

피드백에는 두 가지 표기를 사용합니다.

- `<<용어>>`: 사전 정의되어야 하는데 모호하게 사용된 용어, metric, notation, local definition을 표시합니다.
- `((피드백))`: 해당 문장, 판단, 실험, 근거, 다음 방향에 대한 교수 피드백을 표시합니다.

`<<>>` 예:

```markdown
본 실험에서는 <<retention>>이 낮아지는 문제가 관찰되었다.
```

`(())` 예:

```markdown
후보 A를 이번 주 방향으로 선택했다.
((후보 B와 C를 제외한 이유가 부족함. 다음 brief의 `3. 관련 연구`와 `4. 선택한 방법 또는 절차`에서 각 후보의 장단점과 선택 이유를 비교해서 설명할 것.))
```

### 13.2 학생이 해야 할 일

`<<>>`로 표시된 항목은 다음 brief에서 반드시 처리해야 합니다.

- 표준 용어 또는 metric이면 `기술 용어`에 citation과 함께 정의합니다.
- 학생이 만든 local term, metric, decision rule이면 `Definitions`에 formula, unit/range, decision rule, 필요한 이유를 정의합니다.
- 수식 기호이면 `Notation` 표에 추가합니다.

`(())`로 표시된 피드백은 다음 연구 진행에 반영해야 합니다. 다음 brief의 `0. 지난주 피드백 반영과 이번 주 변화`에서 각 피드백의 처리 상태와 evidence path를 보여야 합니다.

피드백은 문장 교정 서비스가 아닙니다. 교수의 주석은 학생이 다음 연구 판단과 실험 설계를 고치기 위한 입력입니다.

### 13.3 피드백 자동화와 한계

AI는 학생 brief를 복사하고, 정의되지 않은 용어 후보를 찾고, raw path 누락을 표시하고, 반복되는 구조 문제를 초벌로 잡는 데 사용할 수 있습니다.

하지만 최종 피드백은 지도교수의 판단입니다. 모든 문장을 line-by-line으로 교정하지 않습니다. 핵심은 학생이 다음 주에 어떤 가설을 세우고, 어떤 후보를 비교하고, 어떤 실험으로 검증해야 하는지 명확하게 만드는 것입니다.

---

## 14. FAQ

**Q. 기존 `WEEKLY_LOG.md`는 계속 써야 하나요?**

아니요. 주차별 reasoning, 피드백 반영, evidence map은 모두 `meetings/YYYY-Www_brief.md`에 둡니다. 기존 `WEEKLY_LOG.md`가 있어도 새 주차에서 별도 보고서처럼 갱신하지 마세요.

**Q. method 폴더는 언제 새로 만드나요?**

하나의 논문, 하나의 실험 방향, 하나의 자기완결적 학습 단위가 생겼을 때 만듭니다. 단순 아이디어 메모나 planning만으로 method 폴더를 만들지 마세요.

**Q. summary note를 `paper/` 폴더에 같이 넣어도 되나요?**

안 됩니다. `methodN/paper/`에는 primary paper PDF만 둡니다. 요약, 분석, 실험 해석은 `methodN/markdown/`에 둡니다. 관련 연구 PDF는 `related_work/paper/`에 둡니다.

**Q. 어떤 날은 거의 진전이 없는데도 push해야 하나요?**

의미 있는 진행이 있다면 작게 push하세요. 실패한 실행, 바뀐 가설, 읽은 논문에서 제외 판단을 한 것도 다음 연구 판단에 필요하면 진행입니다.

**Q. 미팅 직전에 한 번에 push해도 되나요?**

안 됩니다. weekly brief는 다른 사람이 미리 읽고 미팅에 들어오기 위한 자료입니다. 미팅 직전 한꺼번에 올라온 문서는 연구 과정을 보이지 않게 만들고, 지도교수나 AI가 사전 검토할 시간을 없앱니다.

**Q. 질문이 여러 개 있으면 어떻게 하나요?**

brief 맨 마지막에는 주요 질문 1개만 둡니다. 나머지는 코드 수정, 용어 정리, caveat, 또는 다음 실험 후보로 본문에 흡수하세요. 미팅 질문은 다음 주 연구 방향을 실제로 바꾸는 것만 남깁니다.
