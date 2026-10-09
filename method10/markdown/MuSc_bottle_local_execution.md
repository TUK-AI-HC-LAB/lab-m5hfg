# MuSc 논문 조건 기반 RTX 5080 첫 실행 결과

후속 완료: MVTec AD 15개 category 전체 실행 결과와 논문 비교는 [전체 실행 보고서](MuSc_paper_vs_local_mvtec_all.md)에 정리했다. 아래는 첫 bottle 실행 당시의 기록이다.

## 질문과 가설

질문: 현재 RTX 5080 / WSL 환경에서 공식 MuSc의 논문 설정을 유지한 채 MVTec AD category 하나의 전체 test pool을 실행하고 검토 가능한 실제 결과를 얻을 수 있는가?

실행 전 가설: 최신 GPU에 맞는 현재 PyTorch 환경으로 pretrained ViT, LNAMD, MSM, RsCIN을 실행할 수 있으며, 16GB VRAM에서 bottle 전체 pool을 분할 없이 평가할 수 있다. 기대 결과는 OOM 없이 완료, 유한한 score/map, 7개 지표와 이미지별 근거 저장이다. 논문 숫자를 맞추는 것을 성공 조건으로 두지 않았다.

## 설정과 원문과의 차이

- 원본 논문: MuSc, ICLR 2024. `../paper/ICLR24_MuSc_Zero_Shot_Industrial_Anomaly_Classification_and_Segmentation_with_Mutual_Scoring_of_the_Unlabeled_Images.pdf`.
- 공식 code: https://github.com/xrli-U/MuSc
- commit: `b76b93da8bd3096a99964a96ae29d46f197a0651`.
- checkout 원본: `/home/test/MuSc`. scoring/loader/metric 소스 수정 없이 실행 래퍼로 저장 기능을 추가했다.
- GPU: RTX 5080. Python 3.12.13, PyTorch 2.11.0+cu128, CUDA 12.8, WSL2.
- 논문 유지: OpenAI CLIP ViT-L/14-336, 입력 518×518, 0-based layers `[5,11,17,23]`, aggregation `[1,3,5]`, IA `[0,0.3]`, 전체 category pool 1개, batch size 4.
- 공식 RsCIN MVTec setting `[1,2,3]` 사용. class token 유사도에서 자기 이미지가 포함되고 k=1이 원점수 항 역할을 하는 공식 구현을 유지했다.
- seed 42를 명시하고 NumPy/PyTorch seed 고정, TF32 비활성화. 공식 CUDA autocast를 그대로 사용했다. 논문 당시 dependency pin 대신 RTX 5080을 지원하는 현재 환경을 유지했고, 없던 `openpyxl==3.1.2`만 추가 설치했다.
- 원문 RTX 3090과 하드웨어 및 소프트웨어 버전이 다르며, 이후 수정된 공식 code commit을 사용한다. 비트 단위 동일 결과를 주장하지 않는다.
- data 원본: `/home/test/data/mvtec/bottle`. test 정상 20장, 이상 63장, 총 83장; 이상 마스크 누락 0. 정상 train은 사용하지 않았다.
- loader와 official metric의 resize/mask 처리도 유지했다. 평가는 원본 해상도가 아닌 official 518×518 grid이며, mask는 공식 변환 후 int32로 변환된다. 공통 framework의 nearest-mask 평가와 섞지 않는다.

## 실제 결과

아래 단위는 %이며 2026-10-02 이 PC에서 측정했다. 원문 숫자와 섞지 않았다.

| 지표 | 실제 값 (%) |
|---|---:|
| Image AUROC | 99.9206349206 |
| Image F1-max | 99.2125984252 |
| Image AP | 99.9751984127 |
| Pixel AUROC | 98.4773434913 |
| Pixel F1-max | 79.1672909682 |
| Pixel AP | 83.0415730452 |
| AUPRO | 96.1027154284 |

Raw CSV: `../source/results/bottle_paper_20261002/metrics.csv`. CSV는 0–1 단위이고 이 표만 100을 곱했다. official AUPRO는 200개 threshold를 스캔하고 FPR<0.3 구간을 선택한 뒤 선택된 FPR 범위를 min–max 정규화한다(`utils/metrics.py`). 다른 AUPRO 구현과 수치가 같다고 가정하지 않는다.

원시 map shape는 `[83,1,518,518]`, score shape는 `[83]`; 모두 NaN/Inf 없음. 공식 heatmap은 각 이미지별 min–max 정규화이므로 색상 강도를 이미지 사이의 절대 점수로 비교하면 안 된다.

## 시간과 메모리

- 모델 로딩·원시 결과 저장·metric·시각화를 포함한 전체 wall time: 75.1078초.
- PyTorch peak allocated: 6,756,543,488 bytes (약 6.29GiB).
- PyTorch peak reserved: 7,610,564,608 bytes (약 7.09GiB).
- official 로그의 MuSc 계산 시간: 376.6247ms/image. 이는 upstream 측정 경계의 값이며 warm-up/반복/CUDA synchronization을 갖춘 정식 속도 benchmark가 아니다. 논문 효율 비교표에는 별도의 측정 실험이 필요하다.
- peak는 모델 로딩 후 reset한 PyTorch allocator 통계다. NVIDIA 전체 VRAM 사용량이나 모든 시스템 프로세스의 합을 의미하지 않는다.

## 재현 근거

- commit: `b76b93da8bd3096a99964a96ae29d46f197a0651`
- sh: `method10/source/run_baseline.sh`
- result: `method10/source/results/bottle_paper_20261002/metrics.csv`
- 실행 래퍼: `../source/run_musc_local.py`
- 환경: `../source/results/bottle_paper_20261002/environment.json`
- 설정: `../source/results/bottle_paper_20261002/config.json`
- 이미지별 score와 이미지 SHA256: `../source/results/bottle_paper_20261002/image_scores.csv`
- 대용량 원시 결과: `/home/test/musc_results/bottle_paper_20261002/raw_predictions.npz` (image_labels, image_scores, masks, anomaly_maps; 약 160MB, repo에 복사하지 않음).
- command log: `/home/test/musc_results/bottle_paper_20261002/run.log`
- 공식 결과·heatmaps: `/home/test/musc_results/bottle_paper_20261002/official/`
- preview 생성 script와 입력 목록: `../source/make_preview.py`, `../source/results/bottle_paper_20261002/preview_manifest.json`

실행: WSL에서 `bash /mnt/c/Users/test/Desktop/Codex/lab-m5hfg/method10/source/run_baseline.sh`. 재실행 시 output 경로를 새 run ID로 바꿔 기존 근거를 보존해야 한다.

## 해석과 다음 검증

현재 가설을 지지한다: full bottle test pool에서 논문 backbone/해상도/pool을 유지해 OOM 없이 완료됐고 실제 평가 결과를 확보했다. 아직 MVTec 15 category 평균, VisA 평균, 다른 방법 대비 개선, 정식 속도 benchmark, 반복 실행 안정성은 주장할 수 없다.

다음 검증은 같은 설정으로 MVTec 전체 category를 실행하고 category마다 별도 raw output을 남기는 것이다. 현재 래퍼는 category 하나의 출력 파일을 저장하므로 `--category ALL`로 바로 실행하면 덮어쓰게 된다. 전체 실행은 category별 별도 output 디렉터리를 사용하는 runner가 필요하다.

## 참고문헌

[1] Li, Xurui, et al. "MuSc: Zero-Shot Industrial Anomaly Classification and Segmentation with Mutual Scoring of the Unlabeled Images." International Conference on Learning Representations, 2024.
