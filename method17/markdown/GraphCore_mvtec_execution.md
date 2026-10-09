# GraphCore MVTec AD 자체 실행

## Paper Metadata

| Item | Content |
|---|---|
| Title | Pushing the Limits of Few-Shot Anomaly Detection in Industry Vision: GraphCore |
| Authors | Guoyang Xie, Jinbao Wang, Jiaqi Liu, Feng Zheng, Yaochu Jin |
| Conference / Journal | ICLR |
| Year | 2023 |
| Paper link | https://openreview.net/forum?id=xzmqxHdZAwO |
| GitHub / Official code | 저자 벤치마크 https://github.com/M-3LAB/open-iad ; 별도 ICLR GraphCore 전용 저장소는 확인하지 못함 |
| Reason for investigation | MuSc Table1의 MVTec4-shot 7지표와 Table2의8-shot Image/Pixel AUROC를 이 PC에서 직접 측정 |

## 문제와 가설

카테고리별 정상 학습 이미지4장 또는8장을 입력으로 memory bank를 구성하고 해당 카테고리의 전체 test 이미지를 평가한다. 전체는15개 카테고리1,725장이다. Image AUROC/F1-max/AP, Pixel AUROC/F1-max/AP, AUPRO를 저장한다. 가설은 ImageNet 사전학습 graph backbone의 특징이 소량의 정상 support로도 이상을 구분한다는 것이다. AUROC와 support별 변동으로 이 가설을 평가하며 결과를 보기 전에 기록한다.

## 실행 구현과 논문 대조

저자의 홈페이지가 연결하는 `M-3LAB/open-iad`에서 `arch/graphcore.py`, `models/graphcore/`를 사용한다. 이것은 저자 벤치마크의 실제 GraphCore 코드이며 ICLR 실험 당시 코드·MuSc 저자 실험 코드와 동일하다고 주장하지 않는다.

| 항목 | 실행 선택 및 근거 |
|---|---|
| Backbone | 공개 `pvig_ti_224_gelu` ImageNet 가중치. 논문Table21의 pyramid, 채널48/96/240/384에 맞춤. 저장소 기본 `vig_ti_224_gelu`에서 명시 변경 |
| Stage 반복 수 | 공개 weight는[2,2,6,2]. 논문Table21은[2,2,2,2]로 쓰여 있어 완전 일치하지 않음. pretrained weight를 임의로 잘라 쓰지 않음 |
| 특징 추출 층 | Pyramid stage2·stage3의 마지막 FFN, backbone0-based[4,11],96×28×28와240×14×14. 논문이 feature tap을 명시하지 않아 재현자의 선택이며 수치 동등성 미확인 |
| Coreset | 논문Table2에 명시된1%=0.01. benchmark 기본0.001에서 변경. 원본 SparseRandomProjection(eps0.9)+KCenterGreedy를 호출 |
| Normal support | category별 local seed42+category index,10개 random.sample draws. 4-shot은 동일8-shot draw 앞4장. 원논문의 sample IDs·반복횟수와 동일하다고 주장하지 않음 |
| 데이터 증강 | 없음. GraphCore 논문2.5와 일치 |
| 이미지·mask | native normal transform 그대로. 이미지224×224 bilinear→CenterCrop→ImageNet정규화. mask는 짧은변224 bilinear→CenterCrop→>=0.5 |
| 점수 | 원본 benchmark prediction: FAISS squared L2 nearest distance, 이미지는patch최대, pixel은cv2 resize+Gaussian sigma4+category minmax. k9를 조회하나 실제점수는첫번째거리만 사용. 논문 nearest-neighbor 논의와 차이 가능 |
| 사전학습 | 공개ImageNet weight 사용, strict state_dict load. MVTec gradient 학습 없음, category별 memory bank 구성 |
| Precision | FP32, matmul·cuDNN TF32끄기, deterministic algorithms, cuDNN benchmark끄기. AMP 미사용 |
| 캐시 | native batch1/eval 특징을 category별1번 CUDA추출 후 CPU에 보존. 동일한 특징을 support10round에서 재사용 |

## 코드 변경과 검증 범위

외부 git checkout은 수정하지 않는다. 원본 GraphCore class를 AST로 읽어 native train/prediction 메서드를 사용하고, 전체 benchmark의 optimizer/logger 초기화만 wrapper로 대체한다. CPU fold buffer·NumPy 변환 전에 graph feature를 CPU로 옮긴다. GPU graph backbone의 FP32 값은 형변환하지 않는다. 캐시는 입력별 특징을 native 메서드에 다시 전달하며 이미지 tensor 자체를 대체한다. 이 절차는 속도 최적화이고 모델 재학습이 아니다.

NumPy2에서 삭제된 `np.float`는 positional initialization에만 `float64`로 호환한다. 원래 요구timm0.6.11은 Python3.12 dataclass import오류로 쓸 수 없어1.0.28을 사용한다. 사용 API는DropPath·register_model·ImageNet상수이며 모델의graph 연산과 weight는저장소 코드 그대로다.

최종 verifier는 raw와bankSHA, 이미지·maskSHA와라벨, 전체300round image3지표, category/shot별round0 pixel3지표를 재계산한다. AUPRO는APRIL-GAN200threshold/FPR<0.3 근사값이며 sourceSHA와값범위를 확인하고 독립 재계산하지 않는다. raw anomaly map은대용량이라WSL 원본에 저장하며Obsidian으로복사하지 않는다. 결과표에는검증passed 이후만반영한다.

## 실행 근거

- commit: `05044fedab142ce4bfd71bb618b3200c0e43f198`
- 준비: [prepare_graphcore.sh](../source/prepare_graphcore.sh)
- sh: [run_graphcore.sh](../source/run_graphcore.sh)
- result: [결과 폴더](../source/results/mvtec_pvig_fp32_20261008/)
- report 원본: `lab-m5hfg/method17/markdown/GraphCore_mvtec_execution.md`
- raw: `/home/test/graphcore_results/mvtec_pvig_fp32_20261008`
- log: `/home/test/graphcore_results/run_20261008.log`

논문PDF는OpenReview403으로출판version을받지못해arXiv가제공하는ICLR2023표기version을보존했다. arXivURL:https://arxiv.org/pdf/2301.12082 . 이실행의논문조건과구현차이를감안해MuSc와직접수치동등성을주장하지않는다.

## 초기 실행 복구

`mvtec_pvig_20261008`의 초기 실행은 cuDNN TF32·benchmark가 켜져 있었다. bottle6장의 원본 GPU 추론과 저장 캐시 대조에서 normalized map 최대차이0.14 이상이 관찰되어 중단했다. 반복 실행에서도 특징·점수가 달라졌다. TF32/알고리즘 선택이 원인인지 아직 확정하지 않으며, 정확성 검사를 통과하지 못한 초기 결과는 MuSc 표에 사용하지 않는다. 초기 원시 결과는 같은 이름의 WSL 폴더에 보존한다. 새 `mvtec_pvig_fp32_20261008`는 TF32를 모두 끄고 deterministic algorithms와 CUBLAS_WORKSPACE_CONFIG=:4096:8로 실행한다. 동일6장 대조가 통과해야 최종 집계가 가능하다.

새 FP32 실행에서 bottle6장 대조가 통과했다. 두 추출층 특징·이미지 점수·정규화 anomaly map의 최대 절대 차이가 모두0이었다. 이는 해당6장의 캐시 경로를 지지하는 증거이며 전체 테스트 이미지의 bitwise 일치를 증명하지 않는다. [보존된 대조 JSON](../source/results/mvtec_pvig_fp32_20261008/native_cache_comparison.json). 전체300round 평가 후 같은 대조와 최종 raw 검증을 다시 수행한다.

## 자체 측정 결과

단위는 %. ±는 10개 support round별 15-category 비가중 평균의 표준편차(ddof0)이다.

| Shot | image_auroc | image_f1_max | image_ap | pixel_auroc | pixel_f1_max | pixel_ap | aupro |
|---|---|---|---|---|---|---|---|
| 4 | 80.9780 ± 1.0801 | 89.4086 ± 0.3108 | 89.7765 ± 0.5897 | 88.7568 ± 0.4668 | 36.5777 ± 0.6149 | 27.6922 ± 0.5830 | 75.3666 ± 0.4896 |
| 8 | 82.0203 ± 0.4270 | 89.5216 ± 0.2203 | 90.2780 ± 0.3389 | 89.8711 ± 0.3309 | 38.0023 ± 0.3789 | 29.0837 ± 0.2900 | 76.9229 ± 0.7215 |

[카테고리 CSV](../source/results/mvtec_pvig_fp32_20261008/category_metrics.csv) · [검증 JSON](../source/results/mvtec_pvig_fp32_20261008/verification.json)
