# Method24 — STPM 로컬 측정 지표

이번 업로드는 실제 측정 지표와 검증 상태 기록만 포함한다. 코드·논문·가중치·환경 경로·예측 배열은 포함하지 않는다. 새 학습 결과가 아니라 공개 가중치 평가와 RsCIN 적용 결과다.

MVTec AD 15개 category, test 1,725장. 원시 값은 0–1 단위이며 평균은 category별 비가중 평균이다.

- [category별 지표](source/results/mvtec_public_20261011/rscin_category_metrics.csv)
- [평균 지표](source/results/mvtec_public_20261011/rscin_mean_metrics.csv)
- [검증 상태](source/results/mvtec_public_20261011/verification.json)

Image AUROC는 RsCIN 미적용 95.55%, 적용 96.23%다. RsCIN 전후 행은 같은 입력의 후처리 조건 비교이며 독립 재학습 seed 반복이 아니다. 두 행의 표준편차를 seed 변동으로 해석하지 않는다.
