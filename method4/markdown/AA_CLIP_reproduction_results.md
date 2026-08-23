# AA-CLIP zero-shot MVTec AD 재현 결과

## 실행 상태

VisA full-shot으로 text adapter 5 epoch와 image adapter 20 epoch을 처음부터 학습한 뒤, final epoch 20 checkpoint를 MVTec AD 15개 범주에서 평가했다.

- official revision: `53db195f230442aa118c246876c94ba1c76139cc`
- 학습: CLIP backbone 명시 동결, TF32 on, BF16 AMP off
- 평가: final `image_adapter.pth`, batch size 16
- 실행 script: [`run_aaclip_visa_to_mvtec_tf32_fp32_frozen_wsl.sh`](../source/run_aaclip_visa_to_mvtec_tf32_fp32_frozen_wsl.sh)
- raw result: [`AA_CLIP_VisA_fullshot_to_MVTec_seed111_tf32_fp32_frozen.csv`](../source/results/AA_CLIP_VisA_fullshot_to_MVTec_seed111_tf32_fp32_frozen.csv)

## 비교 기준

논문 [1] Table 1-2의 VisA full-shot → MVTec AD 결과는 Pixel AUROC 91.9%, Image AUROC 90.5%다.

| 지표 | 논문 보고값 | 이번 재현 | 차이 | Raw path |
|---|---:|---:|---:|---|
| Pixel AUROC | 91.90% | 92.19% | +0.29%p | [`CSV`](../source/results/AA_CLIP_VisA_fullshot_to_MVTec_seed111_tf32_fp32_frozen.csv) |
| Image AUROC | 90.50% | 90.22% | -0.28%p | [`CSV`](../source/results/AA_CLIP_VisA_fullshot_to_MVTec_seed111_tf32_fp32_frozen.csv) |

## 해석

- 두 핵심 지표가 논문 값과 0.3%p 이내로 근접해 이 protocol의 재현에 성공했다.
- 초기 실행은 backbone parameter를 명시적으로 동결하지 않아 학습이 비정상적으로 느리고 성능도 낮았다. 최종 실행에서는 `requires_grad=False`로 두 CLIP backbone을 동결했고 adapter만 업데이트했다.
- 현재 환경은 Python 3.12/PyTorch 2.11/CUDA 12.8이므로 저자 환경과 bitwise-identical한 실행은 아니다.

## 참고문헌

[1] Ma, Wenxin, et al. "AA-CLIP: Enhancing Zero-Shot Anomaly Detection via Anomaly-Aware CLIP." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2025, pp. 4744-4754.
