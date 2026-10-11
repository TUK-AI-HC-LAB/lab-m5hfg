# Method23 — PaDiM 로컬 측정 지표

이번 업로드는 실제 측정 지표와 검증 상태 기록만 포함한다. 코드·논문·가중치·환경 경로·예측 배열은 포함하지 않는다. 논문 공개값을 로컬 측정값으로 대체하지 않았다.

BTAD 3개 product, test 741장의 단일 조건 실행이다. 결과 폴더의 식별 조건은 WR50, d=550, seed=1024다. 평균은 product별 비가중 평균이고 원시 값은 0–1 단위다.

- [product별 지표](source/results/btad_wr50_d550_seed1024_20261010/category_metrics.csv)
- [평균 지표](source/results/btad_wr50_d550_seed1024_20261010/mean_metrics.csv)
- [검증 상태](source/results/btad_wr50_d550_seed1024_20261010/verification.json)

Image AUROC 94.79%, Pixel AUROC 97.60%. 이 수치는 해당 실행 조건의 측정값이며 다른 조건의 방법과 우열을 주장하지 않는다.
