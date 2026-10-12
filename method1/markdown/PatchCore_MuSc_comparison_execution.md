# PatchCore — MuSc 비교표 조건 실행

## Paper Metadata

|Item|Content|
|---|---|
|Title|Towards Total Recall in Industrial Anomaly Detection|
|Authors|Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Schölkopf, Thomas Brox, Peter Gehler|
|Conference / Journal|CVPR|
|Year|2022|
|Paper link|https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html|
|GitHub / Official code|https://github.com/amazon-science/patchcore-inspection|
|Reason for investigation|MuSc Table1/2/11/14의 PatchCore 행을 이 PC의 직접 측정값으로 채움|

2026-10-10 사용자 요청으로 실행했다. 이전 method1에는 MVTec15개 category의 CSV31개만 있고 저장된 `.faiss`·`patchcore_params.pkl`을 찾지 못했다. 이번에는 공식 code를 WSL에서 고정 commit으로 실행하고 모델 params, GPU에서 생성한 인덱스의 CPU 저장본, coreset memory bank를 함께 보존한다.

## 실행 범위와 조건

|실험|Category|Support|Seed|
|---|---|---|---|
|MVTec full-shot|15|정상 train 전체|42|
|MVTec4-shot|15|category별 정상4장|42,43,44|
|VisA4-shot|12|공식1cls split 정상4장|42,43,44|
|BTAD4-shot|3|제품별 정상4장|42,43,44|

총105 category runs. image AUROC/F1-max/AP, pixel AUROC/F1-max/AP, AUPRO를 모두 계산한다. dataset의 기존 meta.json을 읽고 원본 image·mask SHA와 정상 support/test 분리를 확인한다. support는 sorted metadata 순서에서 Python Random(seed).sample로4장 선택하고 전체경로·SHA를 CSV로 보존한다. 성능은 category별 비가중 평균 후3개의 seed 평균·population std(ddof0)를 구한다. MVTec full-shot은 단일 seed이므로 ±를 붙이지 않는다.

- 공식 commit: `fcaa92f124fb1ad74a7acf56726decd4b27cbcad`; tracked upstream source는 수정하지 않음.
- 공개 README 기본 baseline: ImageNetV1 WideResNet50, layer2/layer3, resize(shorter)256·center crop224, bilinear RGB·ImageNet normalization. mask nearest 같은 crop 후 >0.
- pretrain/target embedding1024, patchsize3, stride1, nearest neighbor1, native approximate greedy coreset10%, projection128·native random anchors.
- GPU FP32 특징 추출·coreset 및 native FAISS GPU exact flat squaredL2 검색. TF32·AMP 미사용, cuDNNbenchmark·worker4/pinned memory 사용.
- native 최대 patch score의 image score, native bilinear upsample·Gaussian sigma4 anomaly map을 사용. 평가224crop. 공통 framework 결과를 섞지 않음.
- MVTec full-shot은 공식 RsCIN 적용 전후 classification3지표도 계산. 공식 MuSc ViT-L14-336/518 projected normalized class token을 사용하며 이미지 path/label/SHA로 정렬한다. 기존 feature cache가 없어 필요한 category를 공식 MuSc로 새로 추출하고 보존한다. RsCIN 공개 기본 창[1,2,3] 사용(논문 본문의[2,3]과 차이 명시).

## 조건 차이·검증 한계

MuSc 원문에서 PatchCore의 세부 resize/coreset/ensemble 및 few-shot support sampling을 확정하지 못했다. 이 실행은 공식 README의 단일 WR50/224 baseline을 명시한 로컬 결과이며 원문의 PatchCore 최고 ensemble 수치 재현이나 bitwise 동일성을 주장하지 않는다. 저자 당시 Python/torch/FAISS 환경과 다르고, seed42·3회 반복은 로컬 설정이다. 원문의 수치를 결과 셀에 쓰지 않는다.

AUPRO는 MuSc의200threshold/FPR<0.3/관측FPR normalization 정의를 사용하되 connected regions와 score sorting을 캐시해서 동일 threshold의 count를 구한다. 실제 변환 이미지의 축소 subset에서 native cal_pro_score와1e-12 이내로 대조한다. reported metric에는224 전체 map을 사용한다. 정상 label은 원본 metadata에서 유지하고 cropped mask에서 추론하지 않는다(BTAD02의 원래 빈 mask를 그대로 유지).

각 model의64 actual query를 전체 coreset 대상 FAISS CPU flat과 비교하고4 query는 전체 memory FP64 brute-force와 비교한다(rtol1e-4,atol0.003). 모든 input/mask SHA와 저장된 raw/model SHA를 보존한다. raw prediction을 다시 읽어7지표를 재계산한다. 모든 test forward를 이중 실행하거나 전체 coreset selection을 독립 구현으로 재실행하는 검증은 아니다. 완료 category의 raw hash가 맞으면 재개 시 건너뛰며 저장된 index/params가 있으면 memory bank 생성 없이 평가를 이어간다.

## 초기 실행에서 확인한 문제

진행 표시를 비활성화하는 초기 wrapper가 native tqdm context-manager를 깨뜨려 fit 전에 실패했다. tqdm 자체를 유지하고 disable=True만 설정하도록 수정했다. 첫 bottle memory/index 생성 및 거리 비교 후 기존 RsCIN feature cache가 없어서 중단된 것도 기록했다. 이미 저장된 bottle index/params를 재사용하고 category별 공식 MuSc feature 추출 fallback을 추가하여 다시 실행했다. 이전 실패 log는 삭제하지 않는다.

## 실행·산출물

- sh: [run_patchcore_musc.sh](../source/run_patchcore_musc.sh)
- 실행: [run_patchcore_musc.py](../source/run_patchcore_musc.py), [시작](../source/start_patchcore_musc.py)
- 공식 RsCIN 특징: [extract_patchcore_rscin.py](../source/extract_patchcore_rscin.py)
- 환경: [runtime_patchcore_musc.txt](../source/runtime_patchcore_musc.txt)
- result: [결과 폴더](../source/results/musc_comparison_20261010/)
- Raw/model: `/home/test/patchcore_results/musc_comparison_20261010`
- Log: `/home/test/patchcore_results/run_musc_20261010.log`
- 주요 artifacts: category별 support/test/imagescores.csv, metrics.json, verification.json, distance_verification.json, raw_manifest.json. 집계 category_metrics/run_means/aggregates/rscin_mean_metrics.csv.
- 메모리 뱅크: `memory_bank.npy`, `nnscorer_search_index.faiss`, `patchcore_params.pkl`. 예측: `predictions.npz`에 original labels·image score·map·변환mask·RsCIN score.

최종105개 run 검증을 통과한 뒤에만 [MuSc 비교표](../../method10/markdown/MuSc_all_reproduction_tables.md)에 숫자를 반영한다. 2026-10-10 모든 105개 category run 검증을 통과하여 실행 완료했다. MVTec full/4-shot, VisA 4-shot, BTAD 4-shot과 MVTec RsCIN 전후 결과를 비교표에 반영했다. [최종 검증](../source/results/musc_comparison_20261010/verification.json) · [집계 CSV](../source/results/musc_comparison_20261010/aggregates.csv). 원본 method1 paper를 재사용하고 dataset/model/env는 연구 원본 밖에 유지한다. Obsidian 사본을 만들지 않고 연결 목록만 갱신한다. 반복 진행률/ETA polling은 추가하지 않았다.
