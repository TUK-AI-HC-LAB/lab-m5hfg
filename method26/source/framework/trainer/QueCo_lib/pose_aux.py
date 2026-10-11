"""
자세/배치 보조 맵 (선택 기능, metal_nut / transistor 전용).

패치 단위 비교로는 잡기 어려운 "뒤집힘·밀림" 이상을, 고정 텍스트 쌍에 대한
이미지 단위 CLIP 점수로 보완한다. 클래스 이름과 결함 유형을 미리 아는 것을
전제로 하므로 본 실험에서는 끄고(enable_pose_aux=False) 보조 실험에서만 쓴다.
"""
import torch
import torch.nn.functional as F

from .prompt_ensemble import tokenize

from .utils import parse_mvtec_cls_type

TEMPERATURE = 0.07

# 클래스별 (정상 문장, 이상 문장). args의 "<key>_norm_prompt" 등으로 바꿀 수 있다.
DEFAULT_PROMPTS = {
    "metal_nut": ("a photo of a metal nut.", "a photo of an inverted metal nut."),
    "transistor": ("a photo of a transistor.", "a photo of a transistor shifted to the right."),
}
_ARG_KEYS = {
    "metal_nut": ("metalnut_norm_prompt", "metalnut_flip_prompt"),
    "transistor": ("transistor_norm_prompt", "transistor_mis_prompt"),
}


@torch.no_grad()
def encode_text(model, prompt_learner, text, device):
    """
    문장 하나를 정규화된 텍스트 임베딩 [D]로 만든다.

    design_details가 들어간 모델은 encode_text()가 동작하지 않아 encode_text_learn()을 쓴다.
    """
    tokens = tokenize(text).to(device)                                   # [1,77]
    cast_dtype = model.transformer.get_cast_dtype()
    prompts = model.token_embedding(tokens).type(cast_dtype)             # [1,77,dim]

    # deep prompt를 넘겨야 transformer([x, deep, 0]) 경로를 탄다
    deep = prompt_learner.compound_prompts_text
    feat = model.encode_text_learn(prompts, tokens, deep).float()        # [1,D]
    return F.normalize(feat, dim=-1)[0]


class PoseAux:
    """
    사용 순서: 생성 -> calibrate(bank) -> anomaly_map(item, cls, H, W)
    """

    CLASSES = tuple(DEFAULT_PROMPTS)

    def __init__(self, model, prompt_learner, args, device):
        self.device = device
        self.enabled = bool(getattr(args, "enable_pose_aux", True))

        # 사전에 고정한 상수 (테스트셋으로 튜닝하지 않음)
        self.beta = float(getattr(args, "pose_aux_beta", 4.0))
        self.delta = float(getattr(args, "pose_aux_delta", 0.5))
        self.pmin = float(getattr(args, "pose_aux_pmin", 0.70))
        self.bg_thresh = float(getattr(args, "bct_bg_thresh", 0.65))
        self.fg_dilate = int(getattr(args, "pose_aux_fg_dilate", 9))

        self.text_feats = {}      # cls -> (정상 임베딩, 이상 임베딩)
        self.batch_stats = {}     # cls -> (mu, std)
        if not self.enabled:
            return

        for cls in self.CLASSES:
            norm_key, abn_key = _ARG_KEYS[cls]
            norm_text = getattr(args, norm_key, DEFAULT_PROMPTS[cls][0])
            abn_text = getattr(args, abn_key, DEFAULT_PROMPTS[cls][1])
            self.text_feats[cls] = (
                encode_text(model, prompt_learner, norm_text, device),
                encode_text(model, prompt_learner, abn_text, device),
            )

    def _score(self, img_feat, cls):
        """이상 문장 쪽으로 얼마나 더 가까운지. img_feat: [D] 또는 [K, D]"""
        t_norm, t_abn = self.text_feats[cls]
        return ((img_feat @ t_abn) - (img_feat @ t_norm)) / TEMPERATURE

    @torch.no_grad()
    def calibrate(self, bank):
        """클래스별로 테스트셋 전체의 점수 평균/표준편차를 구한다(라벨 미사용)."""
        self.batch_stats = {}
        if not self.enabled:
            return

        for cls in self.CLASSES:
            feats = [b["img_feat"] for b in bank
                     if parse_mvtec_cls_type(b.get("img_path", ""))[0] == cls]
            if len(feats) == 0:
                continue
            feats = F.normalize(torch.stack(feats, dim=0).float().to(self.device), dim=-1)
            score = self._score(feats, cls)
            self.batch_stats[cls] = (
                float(score.mean().item()),
                max(float(score.std().item()), 0.1),
            )

    @torch.no_grad()
    def _foreground_mask(self, bank_item, H, W):
        """objectness가 낮은 영역(물체)을 조금 넓힌 전경 마스크 [H, W]"""
        obj_full = bank_item.get("obj_full", None)
        if obj_full is None:
            return torch.zeros((H, W), device=self.device)

        fg = (obj_full.to(self.device) < self.bg_thresh).float()
        if fg.shape != (H, W):
            fg = F.interpolate(fg[None, None], size=(H, W), mode="nearest")[0, 0]

        if self.fg_dilate > 1:
            fg = F.max_pool2d(
                fg[None, None], kernel_size=self.fg_dilate, stride=1, padding=self.fg_dilate // 2
            )[0, 0]
        return fg.clamp(0, 1)

    @torch.no_grad()
    def anomaly_map(self, bank_item, cls, H, W):
        """
        이미지가 "자세 이상"으로 판단되면 전경 전체에 그 확률을 칠한 맵 [H, W].
        대상 클래스가 아니거나 확률이 pmin 미만이면 0 맵.
        """
        zeros = torch.zeros((H, W), device=self.device)
        if not self.enabled or cls not in self.text_feats:
            return zeros

        mu, std = self.batch_stats.get(cls, (0.0, 1.0))
        g = F.normalize(bank_item["img_feat"].to(self.device).float(), dim=-1)

        # 테스트셋 통계로 z-score 변환 후 게이트
        z = (self._score(g, cls) - mu) / max(std, 1e-6)
        p_raw = torch.sigmoid(self.beta * (z - self.delta))
        p = torch.clamp(2.0 * p_raw - 1.0, 0.0, 1.0)
        if p.item() < self.pmin:
            return zeros

        return (p * self._foreground_mask(bank_item, H, W)).clamp(0, 1)
