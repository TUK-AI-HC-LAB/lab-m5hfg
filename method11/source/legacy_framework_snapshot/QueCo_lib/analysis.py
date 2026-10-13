"""
분석 도구: 시각화 저장과 브랜치 융합 진단. 추론 결과에 대한 분석을 수행하고, 히트맵과 오버레이를 저장하며, 엔트로피 기반 브랜치 융합의 성능을 진단한다.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from loguru import logger
from PIL import Image

from .utils import parse_class_and_subclass, parse_mvtec_cls_type


# ----------------------------------------------------------------------
# 시각화: 히트맵 / 오버레이
# ----------------------------------------------------------------------


def _resize_map(m, width, height):
    """[h, w] (0~1) 맵을 이미지 크기에 맞춘다."""
    if m.shape == (height, width):
        return m
    img = Image.fromarray((m * 255).astype(np.uint8)).resize((width, height), resample=Image.BILINEAR)
    return np.array(img) / 255.0


def _save_heatmap(m, path, cmap, add_colorbar=False, pad_inches=0.0, **imshow_kwargs):
    plt.figure(figsize=(5, 5))
    plt.imshow(m, cmap=cmap, interpolation="bilinear", **imshow_kwargs)
    plt.axis("off")
    if add_colorbar:
        plt.colorbar(fraction=0.046, pad=0.04)
    plt.savefig(path, bbox_inches="tight", pad_inches=pad_inches)
    plt.close()


def _save_overlay(image, m, path, cmap, alpha):
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.imshow(m, cmap=cmap, alpha=float(alpha), interpolation="bilinear")
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", pad_inches=0.0)
    plt.close()


@torch.no_grad()
def save_objectness_map(obj_hw, image_path, save_dir="./obj_vis", alpha=0.45, add_colorbar=False, tag="obj"):
    """
    objectness 맵 [H, W] (0~1)을 <save_dir>/<class>/<subclass>/ 아래에 저장한다.
    원본 이미지가 있으면 오버레이도 함께 저장한다.
    """
    if isinstance(image_path, (list, tuple)):
        image_path = image_path[0] if len(image_path) > 0 else None

    if image_path is not None:
        cls_name, sub_class = parse_class_and_subclass(image_path)
        img_name = os.path.splitext(os.path.basename(image_path))[0]
    else:
        cls_name, sub_class = "unknown_class", "unknown_subclass"
        img_name = "unknown"

    out_dir = os.path.join(save_dir, cls_name, sub_class)
    os.makedirs(out_dir, exist_ok=True)

    m = obj_hw.detach().to("cpu").clamp(0, 1).numpy()
    _save_heatmap(m, os.path.join(out_dir, f"{img_name}_{tag}.png"), cmap="jet_r",
                  add_colorbar=add_colorbar, vmin=0, vmax=1)

    if image_path is not None and os.path.exists(image_path):
        base = Image.open(image_path).convert("RGB")
        W, H = base.size
        _save_overlay(base, _resize_map(m, W, H),
                      os.path.join(out_dir, f"{img_name}_{tag}_overlay.png"), cmap="jet_r", alpha=alpha)


def save_overlay_map(anomaly_maps, image_paths, save_dir="./final_vis_png",
                     overlay=True, alpha=0.5, add_colorbar=False):
    """
    최종 이상 맵 [B, H, W] (numpy)을 <save_dir>/<class>/<subclass>/ 아래에 저장한다.
    """
    os.makedirs(save_dir, exist_ok=True)
    for amap, img_path in zip(anomaly_maps, image_paths):
        cls_name, sub_class = parse_class_and_subclass(img_path)
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        out_dir = os.path.join(save_dir, cls_name, sub_class)
        os.makedirs(out_dir, exist_ok=True)

        amap = np.clip(amap, 0, 1)
        _save_heatmap(amap, os.path.join(out_dir, f"{img_name}_final_map.png"), cmap="jet",
                      add_colorbar=add_colorbar, pad_inches=0.1)

        if overlay:
            image = Image.open(img_path).convert("RGB")
            W, H = image.size
            _save_overlay(image, _resize_map(amap, W, H),
                          os.path.join(out_dir, f"{img_name}_overlay.png"), cmap="jet", alpha=alpha)


@torch.no_grad()
def save_fusion_failure_case(image_path, gt_mask, visual_map, text_map, fused_map, save_dir, stats):
    """엔트로피 융합이 잘못 고른 사례를 한 장으로 저장: Input | GT | Visual | Text | Fused"""
    if isinstance(image_path, (list, tuple)):
        image_path = image_path[0]
    if image_path is None or not os.path.exists(str(image_path)):
        return

    os.makedirs(save_dir, exist_ok=True)
    image = Image.open(str(image_path)).convert("RGB")

    def to_np(t):
        return t.detach().cpu().squeeze().numpy()

    cls, _ = parse_mvtec_cls_type(image_path)
    base_name = os.path.splitext(os.path.basename(str(image_path)))[0]
    filename = f"{cls}_{base_name}" if cls else base_name

    panels = [
        (to_np(gt_mask), "gray", "GT"),
        (to_np(visual_map), "jet",
         f"Visual\nH={stats['entropy_visual']:.3f}, BCE={stats['bce_visual']:.3f}"),
        (to_np(text_map), "jet",
         f"Text\nH={stats['entropy_text']:.3f}, BCE={stats['bce_text']:.3f}"),
        (to_np(fused_map), "jet",
         f"Fused\nwv={stats['weight_visual']:.3f}, wt={stats['weight_text']:.3f}"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].imshow(image)
    axes[0].set_title("Input")
    for ax, (data, cmap, title) in zip(axes[1:], panels):
        ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title)
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{filename}_fusion_failure.png"), bbox_inches="tight", dpi=180)
    plt.close(fig)


# ----------------------------------------------------------------------
# 브랜치 융합 진단
# 엔트로피가 낮은 브랜치가 실제로도 오차(BCE)가 낮은 브랜치인지 GT로 확인한다.
# ----------------------------------------------------------------------


class FusionDiagnostics:
    def __init__(self, args, dataset_name, fusion_mode, fixed_visual_weight):
        self.dataset_name = dataset_name
        self.fusion_mode = fusion_mode                       # 기록용 (소문자)
        self.fixed_visual_weight = fixed_visual_weight

        self.file_tag = str(getattr(args, "fusion_mode", "entropy"))   # 파일 이름용
        self.save_dir = getattr(args, "fusion_diagnostic_dir", "./fusion_diagnostics")
        self.save_failures = bool(getattr(args, "save_fusion_failures", False))
        self.failure_threshold = float(getattr(args, "fusion_failure_threshold", 0.05))
        self.failure_dir = getattr(args, "fusion_failure_dir", "./fusion_failure_cases")

        self.records = []

    @torch.no_grad()
    def update(self, image_path, gt_mask, visual_norm, text_norm, fused_logits, stats):
        """이미지 한 장의 진단을 기록한다. stats에는 bce_visual / bce_text가 추가된다."""
        visual_prob = F.softmax(visual_norm, dim=-1)[..., 1][0]
        text_prob = F.softmax(text_norm, dim=-1)[..., 1][0]
        fused_prob = F.softmax(fused_logits, dim=-1)[..., 1][0]     # objectness 게이트 적용 전

        gt = gt_mask.to(visual_prob.device).float().squeeze()
        if gt.shape != visual_prob.shape:
            gt = F.interpolate(gt[None, None], size=visual_prob.shape[-2:], mode="nearest")[0, 0]
        gt = (gt > 0).float()

        bce_visual = F.binary_cross_entropy(visual_prob.clamp(1e-6, 1 - 1e-6), gt).item()
        bce_text = F.binary_cross_entropy(text_prob.clamp(1e-6, 1 - 1e-6), gt).item()
        stats["bce_visual"] = float(bce_visual)
        stats["bce_text"] = float(bce_text)

        preferred = "visual" if stats["entropy_visual"] < stats["entropy_text"] else "text"
        lower_error = "visual" if bce_visual < bce_text else "text"
        misselected = preferred != lower_error

        # 잘못 고른 브랜치에 실린 가중치 x 오차 차이
        if preferred == "text":
            severity = stats["weight_text"] * (bce_text - bce_visual)
        else:
            severity = stats["weight_visual"] * (bce_visual - bce_text)

        self.records.append({
            "dataset": self.dataset_name,
            "image_path": str(image_path),
            "fusion_mode": self.fusion_mode,
            "fixed_visual_weight": self.fixed_visual_weight,
            **stats,
            "entropy_preferred_branch": preferred,
            "lower_error_branch": lower_error,
            "entropy_misselection": int(misselected),
            "misselection_severity": float(severity),
        })

        if self.save_failures and misselected and severity > self.failure_threshold:
            save_fusion_failure_case(
                image_path, gt, visual_prob, text_prob, fused_prob, self.failure_dir, stats
            )

    def save(self):
        """이미지별 CSV와 요약 CSV(오선택률, 엔트로피-BCE 스피어만 상관)를 저장한다."""
        if len(self.records) == 0:
            return

        df = pd.DataFrame(self.records)
        os.makedirs(self.save_dir, exist_ok=True)
        prefix = f"{str(self.dataset_name).replace('/', '_')}_{self.file_tag}"
        df.to_csv(os.path.join(self.save_dir, f"{prefix}_diagnostics.csv"),
                  index=False, encoding="utf-8-sig")

        misselection_rate = float(df["entropy_misselection"].mean())
        logger.info(f"[Fusion diagnostic] Entropy mis-selection rate: {misselection_rate:.4f}")

        try:
            from scipy.stats import spearmanr
            rho_visual, p_visual = spearmanr(df["entropy_visual"], df["bce_visual"])
            rho_text, p_text = spearmanr(df["entropy_text"], df["bce_text"])
            summary = pd.DataFrame([{
                "dataset": self.dataset_name,
                "fusion_mode": self.file_tag,
                "num_images": len(df),
                "entropy_misselection_rate": misselection_rate,
                "spearman_visual_entropy_bce": float(rho_visual),
                "spearman_visual_p": float(p_visual),
                "spearman_text_entropy_bce": float(rho_text),
                "spearman_text_p": float(p_text),
            }])
            summary.to_csv(os.path.join(self.save_dir, f"{prefix}_summary.csv"),
                           index=False, encoding="utf-8-sig")
            logger.info(f"[Fusion diagnostic] Visual rho={rho_visual:.4f}, Text rho={rho_text:.4f}")
        except Exception as e:
            logger.warning(f"Spearman analysis failed: {e}")
