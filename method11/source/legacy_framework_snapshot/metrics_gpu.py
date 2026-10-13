import torch
import numpy as np
import torch.nn.functional as F
import torchmetrics
import pandas as pd
from sklearn import metrics
from skimage import measure
import torch
import numpy as np

def normalize_segmentations(segmentations, len_scores, device="cuda"):
    # segmentations를 Torch Tensor로 변환 (GPU로 이동)
    # 역할: `normalize_segmentations`에 해당하는 작업을 수행.
    # 매개변수: segmentations, len_scores, device.
    # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
    # 상세 흐름: 주요 호출은 `torch.tensor`, `.view`, `torch.zeros_like`, `range`, `norm_segmentations_t.cpu.numpy`입니다.
    # 제어 흐름: 반복문 1개입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
    segmentations_t = torch.tensor(segmentations, dtype=torch.float32, device=device)
    
    # min, max 계산 (원래 코드와 동일한 로직)
    # 각 샘플별로 flatten 후 min/max 추출
    min_scores = segmentations_t.view(segmentations_t.shape[0], -1).min(dim=1)[0].view(-1, 1, 1, 1)
    max_scores = segmentations_t.view(segmentations_t.shape[0], -1).max(dim=1)[0].view(-1, 1, 1, 1)
    
    # 원본 코드: norm_segmentations = np.zeros_like(segmentations)
    norm_segmentations_t = torch.zeros_like(segmentations_t)
    
    # 모든 샘플의 min_score, max_score에 대해 누적
    # 원본 코드에서는 for min_score, max_score in zip(min_scores, max_scores) 후 전체 segmentations에 대해 연산
    # 여기서도 동일하게 구현
    for i in range(len(segmentations)):
        # (segmentations - min_score) / max(max_score - min_score, 1e-2)
        denom = torch.clamp(max_scores[i] - min_scores[i], min=1e-2)
        norm_segmentations_t += (segmentations_t - min_scores[i]) / denom

    # 마지막에 scores의 길이로 나눔
    norm_segmentations_t = norm_segmentations_t / len_scores

    # CPU로 이동 후 NumPy로 변환
    return norm_segmentations_t.cpu().numpy()

def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights, anomaly_ground_truth_labels
):
    # 역할: 이름에 해당하는 수치 또는 지표를 계산.
    # 매개변수: anomaly_prediction_weights, anomaly_ground_truth_labels.
    # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
    # 역할: `compute_pro`에 해당하는 작업을 수행합니다.
    # 매개변수: masks, amaps, num_th.
    # 반환값: 구현에서 계산한 결과 또는 None입니다.
    # 상세 흐름: 주요 호출은 `torch.device`, `torch.tensor`, `torch.tensor.int`, `torchmetrics.functional.auroc`, `auroc.cpu.item`입니다.
    # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
    """
    이미지 단위의 AUROC, FPR, TPR를 계산합니다.

    Args:
        anomaly_prediction_weights: [torch.Tensor 또는 리스트] [N] 이미지별 이상 확률 값.
        anomaly_ground_truth_labels: [torch.Tensor 또는 리스트] [N] 이진 레이블 - 이상이면 1, 아니면 0.
    # 역할: 이름에 해당하는 수치 또는 지표를 계산.
    # 매개변수: anomaly_segmentations, ground_truth_masks.
    # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
    # 상세 흐름: 주요 호출은 `torch.device`, `isinstance`, `anomaly_segmentations.view`, `ground_truth_masks.view.int`, `torchmetrics.functional.roc`입니다.
    # 제어 흐름: 조건 분기 3개입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 2개입니다.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 입력 데이터를 텐서로 변환하고 GPU로 이동
    anomaly_prediction_weights = torch.tensor(anomaly_prediction_weights, device=device)
    anomaly_ground_truth_labels = torch.tensor(anomaly_ground_truth_labels, device=device).int()

    # AUROC 계산
    auroc = torchmetrics.functional.auroc(
        anomaly_prediction_weights, anomaly_ground_truth_labels, task='binary'
    )

    ## FPR, TPR, 임계값 계산
    #fpr, tpr, thresholds = torchmetrics.functional.roc(
    #    anomaly_prediction_weights, anomaly_ground_truth_labels, task='binary'
    #)

    ## 결과를 CPU로 이동하여 numpy 배열로 변환
    #fpr = fpr.cpu().numpy()
    #tpr = tpr.cpu().numpy()
    #thresholds = thresholds.cpu().numpy()
    auroc = auroc.cpu().item()

    return {"auroc": auroc}#, "fpr": fpr, "tpr": tpr, "threshold": thresholds}

def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
    # 역할: 이름에 해당하는 수치 또는 지표를 계산.
    # 매개변수: masks, amaps, num_th.
    # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
    """
    픽셀 단위의 AUROC, FPR, TPR를 계산합니다.

    Args:
        anomaly_segmentations: [리스트 또는 torch.Tensor] [NxHxW] 생성된 세그멘테이션 마스크.
        ground_truth_masks: [리스트 또는 torch.Tensor] [NxHxW] 실제 세그멘테이션 마스크.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 입력 데이터를 텐서로 변환하고 GPU로 이동
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = torch.stack(anomaly_segmentations).to(device)
    else:
        anomaly_segmentations = torch.tensor(anomaly_segmentations, device=device)

    if isinstance(ground_truth_masks, list):
        ground_truth_masks = torch.stack(ground_truth_masks).to(device)
    else:
        ground_truth_masks = torch.tensor(ground_truth_masks, device=device)

    if ground_truth_masks.unique().numel() == 1:
        return {
            "auroc": -1,
            "fpr": -1,
            "tpr": -1,
            "optimal_threshold": -1,
            "optimal_fpr": -1,
            "optimal_fnr": -1,
        }

    # 텐서 평탄화
    flat_anomaly_segmentations = anomaly_segmentations.view(-1)
    flat_ground_truth_masks = ground_truth_masks.view(-1).int()

    # ROC 커브 계산
    fpr, tpr, thresholds = torchmetrics.functional.roc(
        flat_anomaly_segmentations, flat_ground_truth_masks, task='binary'
    )
    auroc = torchmetrics.functional.auroc(
        flat_anomaly_segmentations, flat_ground_truth_masks, task='binary'
    )

    # Precision-Recall 커브 계산
    precision, recall, pr_thresholds = torchmetrics.functional.precision_recall_curve(
        flat_anomaly_segmentations, flat_ground_truth_masks, task='binary'
    )

    # F1 스코어 계산
    F1_scores = 2 * precision * recall / (precision + recall + 1e-8)

    # 최적 임계값 찾기
    optimal_idx = torch.argmax(F1_scores)
    optimal_threshold = pr_thresholds[optimal_idx]

    # 최적 임계값으로 예측값 계산
    predictions = (flat_anomaly_segmentations >= optimal_threshold).int()

    # 최적 FPR 및 FNR 계산
    tp = ((predictions == 1) & (flat_ground_truth_masks == 1)).sum().float()
    tn = ((predictions == 0) & (flat_ground_truth_masks == 0)).sum().float()
    fp = ((predictions == 1) & (flat_ground_truth_masks == 0)).sum().float()
    fn = ((predictions == 0) & (flat_ground_truth_masks == 1)).sum().float()

    fpr_optim = fp / (fp + tn + 1e-8)
    fnr_optim = fn / (fn + tp + 1e-8)

    # 결과를 CPU로 이동하여 numpy 배열로 변환
    fpr = fpr.cpu().numpy()
    tpr = tpr.cpu().numpy()
    thresholds = thresholds.cpu().numpy()
    auroc = auroc.cpu().item()
    optimal_threshold = optimal_threshold.cpu().item()
    fpr_optim = fpr_optim.cpu().item()
    fnr_optim = fnr_optim.cpu().item()

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
    }

def compute_pro(masks, amaps, num_th=200):
    # 역할: `compute_pro`에 해당하는 작업을 수행합니다.
    # 매개변수: masks, amaps, num_th.
    # 반환값: 구현에서 계산한 결과 또는 None입니다.
    # 상세 흐름: 주요 호출은 `torch.device`, `torch.tensor`, `amaps.min.item`, `amaps.max.item`, `torch.ones`입니다.
    # 제어 흐름: 반복문 3개입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
    """
    SimpleNet의 Per-region Overlap (PRO) AUC를 계산하는 함수.

    Args:
        masks: [torch.Tensor] 실제 마스크, 크기 [N, H, W]
        amaps: [torch.Tensor] 이상 맵, 크기 [N, H, W]
        num_th: [int] 평가할 임계값의 수
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 입력 데이터를 텐서로 변환하고 GPU로 이동
    masks = torch.tensor(masks, device=device)
    amaps = torch.tensor(amaps, device=device)

    df = []

    min_th = amaps.min().item()
    max_th = amaps.max().item()
    delta = (max_th - min_th) / num_th

    # 팽창을 위한 구조 요소 생성
    k = torch.ones((1, 1, 5, 5), device=device)

    for th in torch.arange(min_th, max_th, delta, device=device):
        binary_amaps = (amaps > th).float()

        # 팽창 연산
        binary_amaps = F.conv2d(
            binary_amaps.unsqueeze(1), k, padding=2
        ).clamp(max=1).squeeze(1)

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            # mask에 연결 요소 레이블링 수행
            mask_np = mask.cpu().numpy().astype(int)
            binary_amap_np = binary_amap.cpu().numpy().astype(int)

            labeled_mask = measure.label(mask_np)
            regions = measure.regionprops(labeled_mask)
            for region in regions:
                coords = region.coords
                axes0_ids = coords[:, 0]
                axes1_ids = coords[:, 1]
                tp_pixels = binary_amap_np[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        # FPR 계산
        inverse_masks = 1 - masks
        fp_pixels = (inverse_masks * binary_amaps).sum().item()
        fpr = fp_pixels / inverse_masks.sum().item()

        df.append({"pro": np.mean(pros), "fpr": fpr, "threshold": th.item()})

    df = pd.DataFrame(df)

    # FPR을 0 ~ 0.3 범위로 정규화
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = metrics.auc(df["fpr"], df["pro"])
    return pro_auc

def rescale(x):
    # 역할: `rescale`에 해당하는 작업을 수행.
    # 매개변수: x.
    # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
    # 상세 흐름: 주요 호출은 `torch.tensor`, `x.min`, `x.max`입니다.
    # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
    x = torch.tensor(x)
    return (x - x.min()) / (x.max() - x.min())
# 한국어 코드 안내: 이 파일은 이상 탐지 성능 지표 계산을 담당합니다.
# 원본 실행 로직은 변경하지 않고, 초보자용 설명 주석만 추가했습니다.
