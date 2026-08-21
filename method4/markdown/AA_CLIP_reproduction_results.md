# AA-CLIP zero-shot MVTec AD 재현 결과

## 실행 상태

VisA full-shot 학습은 `2026-08-21`에 시작했으며, text adapter epoch 0~2 완료 checkpoint를 남기고 epoch 3 도중 중지함. 이후 TF32+BF16 AMP로 재개할 예정이며 raw 결과는 [`AA_CLIP_VisA_fullshot_to_MVTec_seed111_tf32_amp_resume.csv`](../source/results/AA_CLIP_VisA_fullshot_to_MVTec_seed111_tf32_amp_resume.csv)에 저장함.

## 비교 기준

논문 [1] Table 1-2에서 VisA full-shot으로 학습하고 MVTec AD를 평가한 결과는 Pixel AUROC 91.9%, Image AUROC 90.5%임.

| 지표 | 논문 보고값 | 이번 재현 | 차이 | Raw path |
|---|---:|---:|---:|---|
| Pixel AUROC | 91.9% | 재개 대기 | - | `../source/results/AA_CLIP_VisA_fullshot_to_MVTec_seed111_tf32_amp_resume.csv` |
| Image AUROC | 90.5% | 재개 대기 | - | `../source/results/AA_CLIP_VisA_fullshot_to_MVTec_seed111_tf32_amp_resume.csv` |

## 현재 해석

- official revision, MVTec metadata, ViT-L/14-336 GPU forward, VisA archive SHA-256 및 12개 class metadata를 확인함.
- 아직 VisA adapter 학습과 MVTec evaluation 결과가 없으므로, 논문 결과 재현 여부는 판단 불가임. TF32+BF16 AMP를 켜 재개하면, strict FP32 공식-code run과는 구분해 기록해야 함.

## 참고문헌

[1] Ma, Wenxin, et al. "AA-CLIP: Enhancing Zero-Shot Anomaly Detection via Anomaly-Aware CLIP." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2025, pp. 4744-4754.
