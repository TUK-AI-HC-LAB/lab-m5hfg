"""공용 도구: 시드 고정, GT 마스크 정규화, 이미지 경로 해석."""
import random

import numpy as np
import torch


def set_seed(seed=42):
    """재현성을 위해 모든 난수 시드를 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


@torch.no_grad()
def normalize_mask(mask) -> torch.Tensor:
    """
    GT 마스크를 [1, H, W] 이진 마스크로 통일한다.

    입력: [B,C,H,W] / [C,H,W] / [B,H,W] / [H,W]
    출력: [1,H,W], 0이 아닌 픽셀은 모두 이상(1)
    """
    mask = torch.as_tensor(mask).float()

    if mask.dim() == 4:                                   # [B,C,H,W]
        B, C, _, _ = mask.shape
        mask = mask.amax(dim=1) if C > 1 else mask[:, 0, ...]
        if B > 1:
            mask = mask[0:1, ...]
    elif mask.dim() == 3:                                 # [C,H,W] 또는 [B,H,W]
        if mask.shape[0] > 1:
            mask = mask.amax(dim=0).unsqueeze(0)
    elif mask.dim() == 2:                                 # [H,W]
        mask = mask.unsqueeze(0)
    else:
        raise ValueError(f"[normalize_mask] Unexpected mask dim: {mask.shape}")

    return (mask > 0).float()


# ----------------------------------------------------------------------
# 이미지 경로에서 클래스 / 결함 유형 추출
# ----------------------------------------------------------------------


_SPLIT_DIRS = {"train", "test", "val", "validation"}


def _first_path(img_path, empty):
    """리스트/튜플로 들어온 경로는 첫 원소를 쓴다."""
    if isinstance(img_path, (list, tuple)):
        return img_path[0] if len(img_path) > 0 else empty
    return img_path


def parse_mvtec_cls_type(img_path):
    """
    .../<class>/test/<type>/xxx.png -> (class, type)

    못 찾으면 (None, None). 클래스별 분기(transistor 등)에 쓴다.
    """
    path = str(_first_path(img_path, "")).replace("\\", "/")
    parts = [p for p in path.split("/") if len(p) > 0]

    for i, token in enumerate(parts):
        if token in ("test", "train") and i > 0:
            typ = parts[i + 1] if (i + 1) < len(parts) else None
            return parts[i - 1], typ
    return None, None


def parse_class_and_subclass(img_path):
    """
    저장 폴더 이름용 (class, subclass).

    못 찾으면 ("unknown_class", "unknown_subclass")를 돌려준다.
    """
    img_path = _first_path(img_path, None)
    if img_path is None:
        return "unknown_class", "unknown_subclass"

    parts = str(img_path).replace("\\", "/").split("/")
    for i, token in enumerate(parts):
        if token in _SPLIT_DIRS and i > 0 and i + 1 < len(parts):
            return parts[i - 1], parts[i + 1]
    if len(parts) >= 3:
        return parts[-3], parts[-2]
    return "unknown_class", "unknown_subclass"
