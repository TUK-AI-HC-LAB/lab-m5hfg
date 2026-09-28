# W40 기존 method 코드 기반 한계와 개선안 비교

> GitHub 코드 근거: [PatchCore Trainer](../../method9/source/common_framework_validation_patch/framework_snapshot/trainer/trainer_patchcore.py), [RD Trainer](../../method9/source/common_framework_validation_patch/framework_snapshot/trainer/trainer_rd.py), [공통 Dataset](../../method9/source/common_framework_validation_patch/framework_snapshot/datasets/base.py), [전체 snapshot](../../method9/source/common_framework_validation_patch/framework_snapshot/).

## 1. 목적

공통 프레임워크에서 실행한 기존 method의 코드가 무엇을 정상 기준으로 쓰는지 확인하고, 그 구조에서 생기는 limitation과 기존 논문의 개선 방향을 연결한다.

이번 W40에서는 코드와 재현 evidence가 있는 PatchCore와 RD4AD에 집중한다. 서로 다른 limitation을 한 모델에 한 번에 섞어 해결하려 하지 않는다.

```text
코드의 정상 기준
  → 구조적 limitation
  → 기존 논문의 대응 방법
  → 실제 failure/evidence 확인
```

## 2. 한눈에 보는 비교

| method | 코드에서 만드는 기준 | 코드 기반 limitation | 기존 개선 방향 | 현재 다음 행동 |
|---|---|---|---|---|
| PatchCore | query patch와 normal memory bank의 최근접 거리 | memory bank 저장과 최근접 검색 비용 | RD4AD의 memory-bank-free reverse distillation | coreset 비율별 실제 memory·검색 시간 측정 |
| RD4AD | teacher feature와 decoder 복원 feature 차이 | normal feature compactness·anomaly signal 억제를 직접 강제하지 않음 | RD++의 compactness loss + pseudo anomaly reconstruction | transistor map으로 원인 먼저 진단 |

## 3. PatchCore: normal memory bank의 비용

### 3.1 코드에서 확인한 기준

PatchCore는 정상 train image의 patch feature를 모아 memory bank를 만들고, coreset으로 일부를 남긴다. test patch는 이 memory에서 가장 가까운 feature와의 거리를 anomaly score로 사용한다.

```text
normal train image
  → patch embedding
  → featuresampler.run(): coreset
  → nearest-neighbour memory bank

test patch
  → nearest normal memory distance
  → image score / anomaly map
```

근거 코드: `method1/source/patchcore_annotated/src/patchcore/patchcore.py`의 `fit()`, `_fill_memory_bank()`, `_predict()`.

### 3.2 limitation P1

> normal patch를 많이 보관할수록 기준은 촘촘해질 수 있지만, 저장 공간과 최근접 검색 비용이 커진다.

이는 구현 오류가 아니라 memory-based nearest-neighbour 방식의 구조적 비용이다. 현재 재현에서 coreset을 10%에서 1%로 줄이면 MVTec AD 평균 Image/Pixel AUROC가 약 `0.14%p` 낮아졌다. memory를 줄이는 선택에는 성능 trade-off가 있음을 보여 준다.

근거: `method1/source/results/PatchCore_MVTecAD_IM224_WR50_coreset_comparison.csv`.

### 3.3 기존 개선 방향: RD4AD

RD4AD [1]는 normal memory bank를 저장·검색하는 대신, teacher feature와 reverse decoder feature의 차이를 anomaly score로 사용한다.

```text
PatchCore: test patch ↔ 저장된 normal memory
RD4AD:     teacher feature ↔ decoder reconstruction feature
```

RD4AD는 PatchCore의 memory lookup을 없애는 **대체 계열**이다. 이 문서는 RD4AD가 항상 더 빠르거나 더 정확하다고 주장하지 않는다.

### 3.4 다음 검증

현재 결과에는 coreset 비율별 AUROC만 있음. 아래를 같은 GPU·batch·category 조건으로 추가해야 memory–성능 trade-off를 말할 수 있음.

1. coreset 비율별 memory bank patch 수
2. GPU peak memory
3. 이미지당 평균 추론 시간
4. Image AUROC, Pixel AUROC

## 4. RD4AD: normal feature 분리의 부족

### 4.1 기준과 현재 관찰

RD4AD는 fixed teacher feature를 one-class bottleneck과 reverse decoder로 복원하고, teacher–decoder feature 차이로 anomaly map을 만든다.

현재 재현에서 `transistor`의 Pixel AUROC는 `92.30%`, AU-PRO는 `78.30%`로 전체 평균보다 낮았다.

근거: `meetings/2026-W34_brief.md`, `method3/source/results/RD4AD_MVTecAD_WR50_paper_protocol_results.csv`.

### 4.2 limitation R1

> RD4AD의 원래 bottleneck과 distillation loss만으로는 normal feature를 충분히 compact하게 만들거나 anomaly signal을 제거하도록 직접 강제하지 않는다.

이 limitation은 RD++ [2]가 RD4AD를 분석하며 제시했다. 다만 현재 `transistor`의 낮은 위치 지표가 정말 이 limitation 때문인지는 확정하지 않았다. RD4AD 원 논문이 언급한 annotation–prediction 위치 불일치도 다른 원인 후보이기 때문이다.

### 4.3 기존 개선 방향: RD++

| RD4AD의 부족 | RD++의 대응 |
|---|---|
| compact normal representation을 직접 강제하지 않음 | self-supervised optimal transport와 contrast loss |
| anomaly signal 억제 신호가 약함 | simplex noise pseudo anomaly와 reconstruction loss |

### 4.4 다음 검증

`transistor`의 동일 query에 대해 아래를 한 panel로 비교한다.

```text
원본 image + ground-truth mask + RD4AD anomaly map
```

- map의 높은 반응이 실제 결함 위치에 있는가?
- 정답 mask에 포함됐지만 모델이 반응하지 않은 위치가 annotation 해석 차이인가?
- 실제 오탐·누락이라면 RD++의 feature compactness/pseudo anomaly 방향을 적용 후보로 검토함.

## 5. 현재 W40 결론

PatchCore와 RD4AD는 모두 one-class anomaly detection이지만 정상 기준과 limitation이 다르다.

- PatchCore: normal memory를 직접 비교하므로 memory·검색 비용을 측정해야 함.
- RD4AD: memory bank는 없지만 feature 분리가 충분한지 location failure를 먼저 진단해야 함.

따라서 W40에서 현재 주장할 수 있는 것은 **공통 framework로 방법별 code path를 구분했고, 각 방법마다 다른 limitation과 다음 evidence를 정했다**는 점까지다.

## 6. Evidence Map

| 판단 | 근거 경로 | 말할 수 있는 것 | 아직 부족한 것 |
|---|---|---|---|
| PatchCore는 normal memory와 NN search를 사용 | `method1/source/patchcore_annotated/src/patchcore/patchcore.py` | memory/search 비용 구조가 있음 | 실제 latency·GPU memory 값 |
| PatchCore coreset은 trade-off를 보임 | `method1/source/results/PatchCore_MVTecAD_IM224_WR50_coreset_comparison.csv` | 10%→1%에서 평균 지표가 약 0.14%p 하락 | 모든 category에서 같은 trade-off인지 |
| RD4AD의 feature limitation 후보 | `related_work/markdown/CVPR23_RDpp_Revisiting_Reverse_Distillation_for_Anomaly_Detection.md` | RD++가 compactness/anomaly suppression 부족을 지적 | 현재 transistor failure의 직접 원인 |

## 참고문헌

[1] Deng, Hanqiu, and Xingyu Li. "Anomaly Detection via Reverse Distillation From One-Class Embedding." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2022, pp. 9737-9746.

[2] Tien, Tran Dinh, et al. "Revisiting Reverse Distillation for Anomaly Detection." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2023, pp. 24511-24520. [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Tien_Revisiting_Reverse_Distillation_for_Anomaly_Detection_CVPR_2023_paper.pdf).
