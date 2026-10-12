# STPM MVTec 공개 모델 자체 평가와 RsCIN 비교

## Paper Metadata

| Item | Content |
|---|---|
| Title | Student-Teacher Feature Pyramid Matching for Anomaly Detection |
| Authors | Guodong Wang, Shumin Han, Errui Ding, Di Huang |
| Conference / Journal | BMVC |
| Year | 2021 |
| Paper link | https://www.bmva-archive.org.uk/bmvc/2021/assets/papers/1273.pdf |
| GitHub / Official code | https://github.com/gdwang08/STFPM |
| Reason for investigation | MuSc Table11 STPM MVTec full-shot classification 및 RsCIN 전후 비교 값을 이 PC에서 측정 |

## 질문·가설

질문: 같은 STPM 기본 이미지 점수에서 RsCIN 적용이 Image AUROC/F1-max/AP를 어떻게 바꾸는가. 가설: CLS 유사도가 높은 이미지 간의 점수 smoothing이 단독 점수의 불안정성을 줄일 수 있으나, 비슷한 정상·이상 이미지가 섞이면 일부 category는 악화될 수 있다. 적용 전후 동일 image/label을 사용해 검증한다.

## 실행과 재현 범위

- commit: `2598a5e35fd02f2f9dcfd0f3e8249adc22320e59`
- sh: `method24/source/run_stpm_mvtec_rscin.sh`
- result: `method24/source/results/mvtec_public_20261011/`
- raw/log: `/home/test/stpm_results/`
- 저자 공개 학생 가중치: README의 Google Drive `16Ra76UhwY8EZg2SAaJCdFFZfaJbpGhdq`, `/home/test/stfpm_assets/STFPM/snapshots/{category}/best.pth.tar`. source·teacher·student·dataset·CLS feature SHA 기록.

MVTec 15개 category/1,725장 test를 평가한다. ImageNet ResNet18 teacher와 범주별 공개 student를 strict load한다. 추가 학습은 수행하지 않으며, 공개 모델의 실제 학습 epoch·로그는 독립적으로 검증할 수 없다. 논문은 100epoch, SGD lr0.4/batch32/정상train80%-validation20%를 설명하고 README 예시에는200epoch가 있다. 이 실행은 제공된 best checkpoint 평가이다.

공식 Resize256×256 + ToTensor + ImageNet 정규화, feature layer1/2/3, batch1, FP32 경로를 유지한다. AMP/TF32는 꺼두고 cuDNN benchmark 및 데이터 로딩 worker4를 사용한다. 공식 test 함수는 채널별 L2 정규화 특징의 제곱차를 각각64×64로 bilinear 보간하고 세 map을 곱한 뒤 cv2로256×256 보간한다. 최종 image score는256map 최대값이다. 논문의 image-size 보간 후 곱셈 설명과 공식 코드의64map 곱셈 후 보간 차이를 명시하며 공식 코드를 기준으로 한다. 현대 torchvision Resize의 antialias 기본값은 환경과 함께 기록한다.

RsCIN에는 기존 ViT-L14-336 / 입력518 / projected L2-normalized CLS를 사용하되 이미지 경로·라벨·SHA를 대조한다. 공식 Mobile_RsCIN은 MVTec k=[2,3]를 지정하고 MMO에서1을 추가하므로 실제 평균은[2,3,1]이다. 비교군에 동일한 module을 사용한다.

MuSc Table11에 필요한 image AUROC/F1-max/AP만 계산한다. 시간 소모가 큰 pixel PRO는 이 요청에 포함되지 않는다. raw64map·label·기본 score·RsCIN score를 보존해 image score와 metric을 재계산하고, 각 category 첫 실제 이미지에서 NumPy 특징 정규화/곱셈을 대조한다. RsCIN은 저장 score에서 exact 재계산 및 별도 NumPy smoothing 계산을 검증한다. 15개 category의 비가중 평균만 검증 완료 후 표에 반영한다. 논문 수치를 자체 결과로 복사하지 않는다.

## 자체 측정 결과

15개 category 비가중 평균, 단위 %.

| RsCIN | Image AUROC | F1-max | AP |
|---|---:|---:|---:|
| w/o | 95.5488 | 95.2830 | 98.1578 |
| w | 96.2275 | 95.6064 | 98.3426 |

15 category/1,725장 공개 학생 모델 자체 평가와 RsCIN 검증 완료. 추가 gradient 학습은 수행하지 않았다.

AUROC 평균은 +0.6787%p, F1-max는 +0.3234%p, AP는 +0.1848%p다. 평균 개선이 모든 category의 개선을 뜻하지 않으며 capsule·hazelnut·pill·wood의 AUROC는 감소했다. 상세 값은 `rscin_category_metrics.csv`에 보존했다.

첫 실행에서는 별도 NumPy RsCIN 검증 계산이 저장 CLS의 FP16 similarity에서 바로 행 정규화를 수행해 실패했다. 공식 MMO는 similarity를 FP32로 변환한 뒤 계산한다. 검증 계산도 이 순서를 적용해 수정했고, 공식 추론 및 공식 RsCIN 알고리즘은 변경하지 않았다. 수정 후 모든 category의 독립 계산과 저장 결과 재계산이 통과했다. 공개 모델 다운로드 출처·파일 SHA는 `method24/source/asset_provenance.json`, 실패와 재실행 로그는 `/home/test/stpm_results/run_mvtec_20261011.log`에 남겼다.
