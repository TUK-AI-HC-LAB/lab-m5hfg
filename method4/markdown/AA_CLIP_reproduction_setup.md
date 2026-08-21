# AA-CLIP zero-shot MVTec AD 재현 설정

## Paper Metadata

| Item | Content |
|---|---|
| Title | *AA-CLIP: Enhancing Zero-Shot Anomaly Detection via Anomaly-Aware CLIP* |
| Authors | Wenxin Ma, Xu Zhang, Qingsong Yao, Fenghe Tang, Chenxu Wu, Yingtai Li, Rui Yan, Zihang Jiang, S. Kevin Zhou |
| Conference / Journal | CVPR 2025 |
| Year | 2025 |
| Paper link | https://openaccess.thecvf.com/content/CVPR2025/papers/Ma_AA-CLIP_Enhancing_Zero-Shot_Anomaly_Detection_via_Anomaly-Aware_CLIP_CVPR_2025_paper.pdf |
| GitHub / Official code | https://github.com/Mwxinnn/AA-CLIP |
| Reason for investigation | 정상 데이터만으로 범주별 학습하는 PatchCore·RD4AD와 달리, source anomaly dataset에서 CLIP adapter를 학습한 뒤 보지 못한 MVTec 범주에 zero-shot으로 적용하는 방법을 비교하기 위함. |

## 재현 목표와 protocol

AA-CLIP의 MVTec AD 결과는 MVTec으로 adapter를 학습하는 설정이 아니라, **VisA를 source dataset으로 학습하고 MVTec AD를 target dataset으로 평가**하는 zero-shot protocol임 [1]. 따라서 MVTec만으로 학습·평가하면 논문 zero-shot claim을 재현하는 것이 아님.

| 항목 | 설정 |
|---|---|
| 공식 구현체 | `Mwxinnn/AA-CLIP` |
| 고정 revision | `53db195f230442aa118c246876c94ba1c76139cc` |
| source / target | VisA full-shot / MVTec AD |
| backbone | OpenCLIP ViT-L/14, `ViT-L-14-336px.pt` |
| input size | 518 × 518 |
| stage 1 | text adapter 5 epoch, Adam learning rate `1e-5` |
| stage 2 | image adapter 20 epoch, Adam learning rate `5e-4` |
| adapter hyperparameters | `λ=0.1`, `K_T=3`, `K_I=6`, `γ=0.1` |
| feature layers | visual encoder layer 6, 12, 18, 24 |
| seed | 111 |
| training batch size | text adapter 16, image adapter 2 (official `train.py` default) |
| evaluation batch size | 8; RTX 5080 16 GB에 맞춘 throughput-only 변경 (official `test.py` default는 32) |

## 환경과 data path

- 실행 환경: WSL2 Ubuntu, Python 3.12, PyTorch 2.11.0+cu128, NVIDIA GeForce RTX 5080 (16 GB).
- 데이터: MVTec AD는 `/home/test/data/mvtec`, VisA는 `/home/test/data/VisA_20220922`를 사용함. dataset과 checkpoint는 대용량이므로 GitHub에 올리지 않음.
- VisA archive SHA-256: `2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362`; 12개 class의 `image_anno.csv`를 확인함.
- 공식 코드가 Python 3.10·PyTorch 2.3.1을 권장하지만 RTX 5080 GPU 호환을 위해 기존 CUDA 12.8 PyTorch를 유지함. `einops`, `ftfy`, `kornia`, `ipdb`, `openai-clip`만 추가 설치함.
- OpenAI ViT-L/14-336 checkpoint의 SHA-256: `3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02`.

## 로컬 수정 사항

| 파일 | 수정 | 이유 |
|---|---|---|
| `dataset/constants.py` | `BASE_PATH`를 `/home/test`로 변경 | 공식 코드의 저자 로컬 경로(`/data/wenxinma`)를 현재 WSL dataset 경로에 연결 |
| `test.py` | `--results_csv` option 추가 | 범주별 metric을 검토 가능한 CSV로 저장 |
| `train.py`, `test.py` | opt-in `--tf32`, `--amp` flags 추가 | 중단된 checkpoint를 Tensor Core 가속(TF32)과 BF16 AMP로 재개할 수 있게 함 |

모델 구조, adapter hyperparameter, epoch, source/target protocol은 바꾸지 않음. 단, TF32+AMP 재개 run은 strict FP32 공식-code run과 구분해 기록함.

## 실행과 결과 경로

- 실행 script: [`run_aaclip_visa_to_mvtec_wsl.sh`](../source/run_aaclip_visa_to_mvtec_wsl.sh)
- checkpoint: `/home/test/aaclip_checkpoints/visa_fullshot_seed111` (GitHub 제외)
- raw result: [`AA_CLIP_VisA_fullshot_to_MVTec_seed111.csv`](../source/results/AA_CLIP_VisA_fullshot_to_MVTec_seed111.csv)

## 현재 상태

- 완료: official clone revision 고정, MVTec metadata load, GPU ViT-L/14-336 forward, checkpoint 및 VisA archive hash 검증, VisA 12개 class metadata 확인.
- 중지: text adapter epoch 0~2 완료 checkpoint를 보존한 뒤 epoch 3 진행 중 사용자가 중지함. GPU/OS 오류나 OOM은 확인되지 않음.
- 재개 준비: TF32와 BF16 AMP를 켜는 수정 코드 및 [`run_aaclip_visa_to_mvtec_tf32_amp_resume_wsl.sh`](../source/run_aaclip_visa_to_mvtec_tf32_amp_resume_wsl.sh)를 준비함. 이 실행은 앞부분 FP32, 재개 이후 TF32+BF16 AMP인 혼합 정밀도 run임.
- 결과가 생성되기 전에는 논문 수치 재현 여부를 주장하지 않음.

## 참고문헌

[1] Ma, Wenxin, et al. "AA-CLIP: Enhancing Zero-Shot Anomaly Detection via Anomaly-Aware CLIP." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2025, pp. 4744-4754.
