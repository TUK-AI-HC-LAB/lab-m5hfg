# Figure 7·8 — RTX 5080 자체 측정값

MuSc ICLR2024 Figure7/8의 구성에 맞춰 검증된 로컬 sweep CSV로 작성했다. 논문 보고 수치를 그림에 사용하지 않았다. 원본 모양은 Figure7의 AC/AS 그룹 막대와 Figure8의 single-window 곡선 및 두 기준선을 따른다. 가독성을 위해 폰트·색·간격은 조정했다.

| Figure | PNG | PDF | SVG | Plot data |
|---|---|---|---|---|
| 7 | [PNG](figure7_local.png) | [PDF](figure7_local.pdf) | [SVG](figure7_local.svg) | [CSV](figure7_data.csv) |
| 8 | [PNG](figure8_local.png) | [PDF](figure8_local.pdf) | [SVG](figure8_local.svg) | [CSV](figure8_data.csv) |

## 그림 설명

**Figure 7.** MSM minimum-value interval 비율에 따른 MVTec AD와 VisA의 이상 탐지 성능. x축은 하위10·30·50·70·90·100% 평균 구간이다. AC는 Image AUROC, AS는 Pixel AUROC이며 y축 단위는 %. 각 막대는 MVTec15 또는 VisA12 category의 비가중 평균이다. 막대 숫자는 소수1자리 표시값이며 원시 정밀도는 CSV에 보존한다. 논문과 같이 비교를 위해 y축을 절단했고 패널별 범위가 다르다. 두 panel의 막대 높이를 직접 비교하지 않는다.

**Figure 8.** RsCIN single-window 크기 k=2…9 및 full-window(∞)에 따른 Image F1-max(%). 각 점은 category별 비가중 평균이다. 파란 점선은 RsCIN 미적용, 붉은 점선은 공식 multi-window RsCIN이다. MVTec multi-window k={1,2,3}, VisA k={1,8,9}. ∞는 해당 category의 모든 이미지를 사용하는 하나의 window이며 multi-window 기준선과 다르다. 패널별 y축 범위가 다르다.

고정 조건: RTX5080, CLIP ViT-L-14-336/OpenAI, 입력518, seed42, r={1,3,5}, native FP16 autocast·FP64 score aggregation·FP32 RsCIN. MVTec1,725장·VisA2,162장. 단일 seed이므로 오차막대를 추가하지 않았다. backbone 및 공식 module의 재현 차이는 실행 보고서에 기록했다.

- commit: `b76b93da8bd3096a99964a96ae29d46f197a0651`
- sh: [plot_musc_figures7_8.sh](../../plot_musc_figures7_8.sh)
- source: [MSM 평균 CSV](../../results/sweeps_seed42_20261012/msm_interval_ratio_mean_metrics.csv), [RsCIN 평균 CSV](../../results/sweeps_seed42_20261012/rscin_single_window_mean_metrics.csv)
- verification: [sweep 검증](../../results/sweeps_seed42_20261012/verification.json), [figure_manifest.json](figure_manifest.json)
- report: [실험 조건·한계](../../../markdown/MuSc_sweeps_execution.md)

Figure7에서 MVTec의 최고 AC는30%, 최고 AS는50%이며 VisA의 최고 AC는30%, 최고 AS는10%다. Figure8에서는 두 dataset 모두 native multi-window 기준선의 F1-max가 이번 single-window 점들보다 높다. 이 해석은 단일 seed의 관측이며 통계적 우월성 주장으로 확대하지 않는다.
