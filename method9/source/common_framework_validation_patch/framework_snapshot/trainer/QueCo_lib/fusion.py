"""
융합: 레이어/브랜치 로짓을 합쳐 최종 이상 맵과 이미지 점수를 만든다.

로짓의 형태는 [B, H, W, 2] (정상, 이상)이다.
"""
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------
# 로짓 변환 / 정규화
# ----------------------------------------------------------------------


FUSION_MODES = ("visual", "text", "fixed", "variance", "entropy")


def prob_to_logits(P, eps=1e-6):
    """이상 확률 [B, H, W] -> 로짓 [B, H, W, 2]"""
    P = P.clamp(eps, 1 - eps)
    Pa, Pn = P, 1.0 - P
    l0 = torch.log(Pn / (1 - Pn + eps))
    l1 = torch.log(Pa / (1 - Pa + eps))
    return torch.stack([l0, l1], dim=-1)


def pixel_entropy(logits):
    """픽셀별 엔트로피. [B, H, W, 2] -> [B, H, W]"""
    probs = F.softmax(logits, dim=-1)
    return -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)


def normalize_logits(logits, T=1.0, mask=None):
    """
    로짓을 표준화한 뒤 온도 T로 나눈다.

    mask: [B, H, W] 또는 [B, 1, H, W]. 주어지면 mask > 0.5 인 픽셀의 통계를 쓰고,
          그런 픽셀이 10개 미만이면 이미지 전체 통계를 쓴다.
    """
    T = max(float(T), 1e-6)

    if mask is not None and mask.ndim == 4:
        mask = mask[:, 0, :, :]

    valid = (mask > 0.5) if mask is not None else None
    if valid is None or valid.sum() < 10:
        mu = logits.mean(dim=(1, 2, 3), keepdim=True)
        std = logits.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    else:
        pixels = logits[valid.unsqueeze(-1).expand_as(logits)]
        mu = pixels.mean()
        std = pixels.std().clamp_min(1e-6)

    return (logits - mu) / std / T


# ----------------------------------------------------------------------
# 레이어 / 브랜치 융합
# ----------------------------------------------------------------------


def fuse_layers(logits_list, T_layer=1.0, mask=None):
    """레이어별 로짓을 정규화해 평균한다. (레이어가 1개면 mask 없이 정규화)"""
    if len(logits_list) == 1:
        return normalize_logits(logits_list[0], T_layer)
    stacked = torch.stack(
        [normalize_logits(lg, T_layer, mask=mask) for lg in logits_list], dim=0
    )
    return stacked.mean(dim=0)


@torch.no_grad()
def fuse_branches(visual_logits, text_logits, mode="entropy", fixed_visual_weight=0.5, T_branch=1.0):
    """
    시각 브랜치와 텍스트 브랜치를 합친다.

    mode:
        visual   시각 브랜치만
        text     텍스트 브랜치만
        fixed    고정 가중치 (fixed_visual_weight)
        variance 이상 확률의 공간 분산이 클수록 큰 가중치
        entropy  평균 엔트로피가 낮을수록 큰 가중치 (기본)

    return: fused_logits, stats(dict), visual_norm, text_norm
    """
    visual_norm = normalize_logits(visual_logits, T=T_branch)
    text_norm = normalize_logits(text_logits, T=T_branch)

    visual_prob_raw = F.softmax(visual_logits, dim=-1)[..., 1]
    text_prob_raw = F.softmax(text_logits, dim=-1)[..., 1]

    entropy_visual = pixel_entropy(visual_logits).mean()
    entropy_text = pixel_entropy(text_logits).mean()
    variance_visual = visual_prob_raw.var(dim=(1, 2), unbiased=False).mean()
    variance_text = text_prob_raw.var(dim=(1, 2), unbiased=False).mean()

    if mode == "visual":
        weights = visual_logits.new_tensor([1.0, 0.0])
        fused = visual_norm
    elif mode == "text":
        weights = visual_logits.new_tensor([0.0, 1.0])
        fused = text_norm
    else:
        if mode == "fixed":
            w_visual = float(fixed_visual_weight)
            if not 0.0 <= w_visual <= 1.0:
                raise ValueError(f"fixed_visual_weight must be in [0, 1], but received {w_visual}.")
            weights = visual_logits.new_tensor([w_visual, 1.0 - w_visual])
        elif mode == "variance":
            weights = torch.softmax(torch.stack([variance_visual, variance_text]), dim=0)
        elif mode == "entropy":
            weights = torch.softmax(torch.stack([-entropy_visual, -entropy_text]), dim=0)
        else:
            raise ValueError(
                f"Unsupported fusion mode: {mode}. Choose from {', '.join(FUSION_MODES)}."
            )
        fused = weights[0] * visual_norm + weights[1] * text_norm

    stats = {
        "entropy_visual": float(entropy_visual.item()),
        "entropy_text": float(entropy_text.item()),
        "variance_visual": float(variance_visual.item()),
        "variance_text": float(variance_text.item()),
        "weight_visual": float(weights[0].item()),
        "weight_text": float(weights[1].item()),
    }
    return fused, stats, visual_norm, text_norm


def apply_objectness_gate(logits, objectness, gamma=4.0, lam=0.5):
    """
    배경에서 이상 로짓을 낮춘다.

    objectness: [B, H, W], 값이 클수록 배경으로 취급한다.
    이미지 평균보다 배경 성향이 강한 픽셀일수록 이상 로짓을 더 많이 깎고,
    정상 로짓은 그 절반만큼 올린다.
    """
    bg = objectness.clamp(0, 1)
    suppress = F.softplus(gamma * (bg - bg.mean(dim=(1, 2), keepdim=True)))

    out = logits.clone()
    out[..., 1] = out[..., 1] - lam * suppress
    out[..., 0] = out[..., 0] + 0.5 * lam * suppress
    return out


# ----------------------------------------------------------------------
# 최종 맵 / 이미지 점수
# ----------------------------------------------------------------------


@torch.no_grad()
def logits_to_anomaly_map(logits):
    """
    최종 로짓 [1, H, W, 2] -> 이상 맵 [H, W].

    이상 확률과 그 3x3 평균의 기하평균을 써서 고립된 점 잡음을 줄인다.
    """
    prob = F.softmax(logits, dim=-1)[..., 1][0]
    x = prob.unsqueeze(0).unsqueeze(0)
    local_mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    return torch.sqrt(x * local_mean).squeeze()


@torch.no_grad()
def topk_image_score(anomaly_map, topk_ratio, gamma=5.0):
    """상위 topk_ratio 픽셀의 softmax 가중 평균. anomaly_map: [H, W] -> 스칼라 텐서"""
    H, W = anomaly_map.shape
    k = max(1, int(round(float(topk_ratio) * H * W)))
    vals, _ = torch.topk(anomaly_map.view(-1), k=k)
    weights = torch.softmax(gamma * vals, dim=0)
    return (vals * weights).sum()
