# INP-Former zero-shot 재현 설정

## Protocol

논문 [1] 및 공식 구현체의 zero-shot protocol을 따른다. VisA `1cls` 분할에서 multi-class 모델을 학습하고, source weight를 MVTec AD 15개 범주에 적용한다.

## Official settings

- source / target: VisA / MVTec AD
- encoder: `dinov2reg_vit_base_14`
- resize / crop: 448 / 392
- INP 수: 6
- epoch / batch size: 200 / 16
- official implementation revision: `17d265381d9b323a2ef6e05aab0665a85edebe84`

## Local runtime

저자 환경은 Python 3.8.12, PyTorch 2.0.0+cu118, RTX 4090 24GB다. 실행은 RTX 5080 16GB, Python 3.12, PyTorch 2.11.0+cu128에서 수행했다. batch size는 논문과 같은 16을 유지했다.

## Local compatibility changes

- `adeval`이 최신 NumPy에서 제거된 `np.trapz`를 호출해, 동일한 사다리꼴 적분 함수인 `np.trapezoid`로 교체했다. 이는 AUROC/AU-PRO 평가 호환성 수정이다.
- 원본 학습 코드는 마지막 VisA 평가가 끝난 뒤에만 `model.pth`를 저장한다. 평가 오류로 학습 가중치가 사라지지 않도록, 최종 epoch의 평가 **전**에도 같은 state dict를 저장하게 했다.
- zero-shot 이미지 저장 함수가 Windows 구분자(`\\`)로만 image path를 분리한다. WSL에서 동작하도록 `os.path.normpath(...).split(os.sep)`로 교체했다.

위 변경은 모델 구조, loss, 데이터 split, epoch, batch size를 바꾸지 않는다. 다만 저자 환경과 동일한 runtime은 아니므로 bitwise-identical 재현은 주장하지 않는다.

## Result

VisA `1cls`에서 200 epoch 학습 후 MVTec AD 15개 범주 zero-shot 평가를 완료했다. 평균 Image AUROC는 79.00%, Pixel AUROC는 87.25%, Pixel AU-PRO는 74.61%다.

## Artifacts

- script: [`run_inpformer_visa_to_mvtec_zero_shot_wsl.sh`](../source/run_inpformer_visa_to_mvtec_zero_shot_wsl.sh)
- raw metric: [`INP_Former_VisA_1cls_to_MVTec_zero_shot_20260826.csv`](../source/results/INP_Former_VisA_1cls_to_MVTec_zero_shot_20260826.csv)
- external code: `/home/test/Project/INP-Former`
- VisA preprocessing: official `spot-diff` repository의 `1cls` split

## Reference

[1] Luo, Wei, et al. "Exploring Intrinsic Normal Prototypes within a Single Image for Universal Anomaly Detection." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2025.
