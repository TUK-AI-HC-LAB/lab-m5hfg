# W40 공통 프레임워크 전체 method 실행 결과 (MVTec AD `bottle`)

## 결론

공통 프레임워크에 등록된 12개 방법 중 **11개는 `bottle` 클래스에서 끝까지 실행되어 결과 CSV를 만들었다.** `GLASS`만 실행에 필요한 외부 DTD 텍스처 이미지가 준비되지 않아 학습 시작 단계에서 멈췄다. 이는 모델 코드의 결과가 아니라 데이터 의존성 미충족이다.

이 문서는 "각 방법이 현재 프레임워크에서 한 번 실제로 끝까지 도는가"를 확인한 실행 기록이다. 한 클래스, 한 seed 결과이므로 방법 간 성능 우열이나 논문 재현 성능을 주장하는 표는 아니다.

## 1. 무엇을 같은 조건으로 실행했나

| 항목 | 설정 |
|---|---|
| 데이터셋 | MVTec AD `bottle` |
| 학습 데이터 | 정상 이미지 209장 |
| 시험 데이터 | 정상 20장 + 이상 63장 = 83장 |
| seed | `0` |
| 공통 실행 환경 | WSL의 `patchcore-gpu` 가상환경, GPU 사용 |
| 설정 기준 | 각 방법의 `configs/<method>.yaml` 기본값을 사용하고, 공통으로 `dataset=mvtec`, `category=bottle`, `num_workers=1`만 지정 |
| 결과 폴더 | `/home/test/shared_framework_all_methods_w40_20260926` |

`image AUROC`은 이미지 한 장이 정상/이상인지 맞히는 능력이고, `pixel AUROC`은 이미지 안의 각 픽셀이 이상 위치인지 맞히는 능력이다. `saliency f1`은 예측한 이상 영역을 임계값으로 이진화했을 때 정답 마스크와 얼마나 겹치는지를 나타낸다. 값은 모두 높을수록 좋다.

## 2. 실행 결과

| Method | Image AUROC | Pixel AUROC | Saliency F1 | 실행 상태 | 대략 실행 시간 |
|---|---:|---:|---:|---|---:|
| PatchCore | 1.0000 | 0.9878 | 0.6735 | 완료 | 26초 |
| PaDiM | 1.0000 | 0.9872 | 0.7154 | 완료 | 1분 43초 |
| WinCLIP | 0.9992 | 0.9585 | 0.7052 | 완료 | 2분 13초 |
| COAD | 1.0000 | 0.9925 | 0.6903 | 완료 | 1분 33초 |
| SimpleNet | 0.9944 | 0.8776 | 0.5308 | 완료 | 9분 46초 |
| RD | 1.0000 | 0.9882 | 0.7533 | 완료 | 59분 02초 |
| RD-Orig | 1.0000 | 0.9885 | 0.7595 | 완료 | 2분 30초 |
| PromptAD | 1.0000 | 0.9902 | 0.8024 | 완료 | 7분 26초 |
| Dinomaly | 1.0000 | 0.9934 | 0.6053 | 완료 | 49분 04초 |
| UniAD | 1.0000 | 0.9848 | 0.6863 | 완료 | 43분 03초 |
| AnomalyCLIP | 0.8881 | 0.9038 | - | 완료 | 23초 |
| GLASS | - | - | - | **미완료: DTD 데이터 없음** | 시작 직후 중단 |

### 이 표를 읽을 때의 주의점

- `bottle` 하나만 평가했으므로, `1.0000`이 다른 클래스나 전체 MVTec에서도 유지된다는 뜻은 아니다.
- seed가 하나뿐이다. 결과의 안정성은 여러 seed 평균과 표준편차로 확인해야 한다.
- AnomalyCLIP은 ViSA에서 학습된 공식 prompt checkpoint를 불러와 평가했다. 정상 `bottle` 이미지로 해당 방법을 처음부터 학습한 다른 방법들과 출발 조건이 같지 않다. 따라서 이 행을 성능 순위 비교에 쓰면 안 된다.
- 각 방법은 자체 이미지 크기, backbone, epoch, 샘플링 같은 기본 설정이 다르다. 여기서 말하는 "같은 조건"은 같은 데이터 클래스·seed·실행 환경이지, 알고리즘 내부 하이퍼파라미터까지 동일하다는 뜻이 아니다.

## 3. GLASS만 왜 끝나지 않았나

GLASS는 **Generalized Latent Anomaly Synthesis and Segmentation** 방법이다. 정상 MVTec 이미지에 별도 텍스처 이미지(보통 DTD, Describable Textures Dataset)를 섞어 가짜 이상 이미지를 만든 뒤 학습한다. 따라서 MVTec `bottle` 폴더만으로는 학습을 시작할 수 없다.

현재 실행에서 `trainer/GLASS_lib/mvtec.py`가 텍스처 파일 목록을 읽었지만 빈 목록이었다. 그 뒤 `np.random.choice(self.anomaly_source_paths)`가 빈 목록에서 하나를 고르려 하며 다음 오류로 멈췄다.

```text
ValueError: 'a' cannot be empty unless no samples are taken
```

프레임워크 기본 경로는 `/datasets/dtd/images`이다. DTD 이미지가 있는 실제 경로를 준비한 뒤 `anomaly_source_path`에 그 `images` 폴더를 지정하면 GLASS를 같은 프로토콜로 재실행할 수 있다. 이 데이터는 현재 PC/WSL에서 찾지 못했으므로, 임의 다운로드나 다른 이미지 폴더 대체는 하지 않았다.

## 4. 실행 중 발견한 환경 의존성

처음에는 일부 방법이 아래 패키지 부재로 멈췄다. 필요한 패키지를 현재 WSL 환경에 설치한 뒤, 실패한 방법만 같은 조건으로 다시 실행했다.

| 영향 방법 | 부족했던 패키지 | 재실행 결과 |
|---|---|---|
| WinCLIP, COAD, PromptAD | `open_clip_torch` | 완료 |
| RD, RD-Orig | `geomloss`, 이후 `numba` | 완료 |
| GLASS | `albumentations` | 패키지 문제는 해결, DTD 데이터 부재로 미완료 |
| Dinomaly | `colorama` | 완료 |
| PromptAD | `seaborn` | 완료 |

이는 프레임워크의 모든 optional method 의존성을 한 가상환경에 처음부터 설치하지 않았기 때문에 생긴 환경 준비 문제다. 결과 표에는 패키지 설치 후 실제 학습/평가까지 완료된 재실행 결과를 기록했다.

## 5. 재현 근거 (Evidence Map)

### 실행 스크립트와 상태 파일

| 역할 | 경로 |
|---|---|
| 최초 12개 순차 실행 스크립트 | `C:\\Users\\test\\Downloads\\dinomaly_share_codebase\\dinomaly_share_codebase\\run_all_methods_bottle_wsl.sh` |
| 의존성 설치 후 1차 재실행 스크립트 | `C:\\Users\\test\\Downloads\\dinomaly_share_codebase\\dinomaly_share_codebase\\rerun_failed_methods_bottle_wsl.sh` |
| RD/RD-Orig/PromptAD 2차 재실행 스크립트 | `C:\\Users\\test\\Downloads\\dinomaly_share_codebase\\dinomaly_share_codebase\\rerun_remaining_methods_bottle_wsl.sh` |
| 최초 상태와 로그 | `/home/test/shared_framework_all_methods_w40_20260926/status.tsv`, `/home/test/shared_framework_all_methods_w40_20260926/logs/` |
| 1차 재실행 상태와 로그 | `/home/test/shared_framework_all_methods_w40_20260926/retry_after_dependencies/status.tsv`, `/home/test/shared_framework_all_methods_w40_20260926/retry_after_dependencies/logs/` |
| 2차 재실행 상태와 로그 | `/home/test/shared_framework_all_methods_w40_20260926/retry_after_dependencies_round2/status.tsv`, `/home/test/shared_framework_all_methods_w40_20260926/retry_after_dependencies_round2/logs/` |

### 결과 CSV 위치

| Method | Raw CSV |
|---|---|
| PatchCore | `/home/test/shared_framework_all_methods_w40_20260926/patchcore/patchcore__layer2_layer3_/results_patchcore.csv` |
| PaDiM | `/home/test/shared_framework_all_methods_w40_20260926/padim/padim__layer2_layer3_/results_padim.csv` |
| WinCLIP | `/home/test/shared_framework_all_methods_w40_20260926/retry_after_dependencies/winclip/winclip__layer2_layer3_/results_winclip.csv` |
| COAD | `/home/test/shared_framework_all_methods_w40_20260926/retry_after_dependencies/coad/coad__2_3_5_6_7_8_9_/results_coad.csv` |
| SimpleNet | `/home/test/shared_framework_all_methods_w40_20260926/simple/simple__layer2_layer3_/results_simple.csv` |
| RD | `/home/test/shared_framework_all_methods_w40_20260926/retry_after_dependencies_round2/rd/rd__layer2_layer3_/results_rd.csv` |
| RD-Orig | `/home/test/shared_framework_all_methods_w40_20260926/retry_after_dependencies_round2/rd_orig/rd_orig__layer2_layer3_/results_rd_orig.csv` |
| PromptAD | `/home/test/shared_framework_all_methods_w40_20260926/retry_after_dependencies_round2/promptad/promptad__layer2_layer3_/results_promptad.csv` |
| Dinomaly | `/home/test/shared_framework_all_methods_w40_20260926/retry_after_dependencies/dinomaly/dinomaly__layer2_layer3_/results_dinomaly.csv` |
| UniAD | `/home/test/shared_framework_all_methods_w40_20260926/uniad/uniad__layer2_layer3_/results_uniad.csv` |
| AnomalyCLIP | `/home/test/shared_framework_all_methods_w40_20260926/anomalyclip/anomalyclip__layer24_/results_anomalyclip.csv` |

## 6. 현재 판단과 다음 1개 작업

현재 판단은 "공통 실행 흐름은 GLASS를 제외한 11개 등록 방법을 `bottle`에서 실제 결과 CSV까지 생성할 수 있다"이다. 아직 주장할 수 없는 것은 "12개 방법의 성능 비교가 공정하다" 또는 "전체 MVTec 재현이 끝났다"는 결론이다.

다음 작업은 DTD의 `images` 폴더를 준비하고 GLASS만 다시 실행하는 것이다. 그 결과가 생기면 이 문서의 GLASS 행과 상태를 갱신하면 된다.
