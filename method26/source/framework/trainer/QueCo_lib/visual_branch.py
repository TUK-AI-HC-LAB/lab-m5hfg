"""
시각 브랜치: 테스트 이미지끼리 패치를 비교해 이상 점수를 만든다.

    1. objectness   DINO 특징으로 전경/배경 맵과 outlier 점수를 구한다.
    2. 패치 매칭     OT(논문 QueCo에서는 TGC)와 mutual-kNN으로 패치별 정상도를 구한다.
    3. 집계          이웃 이미지들의 정상도를 패치별 하나의 값으로 모아 픽셀 로짓으로 바꾼다.

표기: Zi [N, D] 타깃 패치, Zj [M, D] 이웃 한 장의 패치, Zjs [B, M, D] 이웃 B장.
"""
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from .fusion import prob_to_logits


# ----------------------------------------------------------------------
# 1. Objectness (DINO)
# ----------------------------------------------------------------------


def _minmax(x, dims):
    mn = x.amin(dim=dims, keepdim=True)
    mx = x.amax(dim=dims, keepdim=True)
    return (x - mn) / (mx - mn + 1e-6)


@torch.no_grad()
def objectness_map(dino_feat, out_size):
    """
    패치 간 평균 유사도(자기 자신 제외)를 0~1로 정규화한 맵.

    dino_feat: [B, C, H, W]
    return:    [B, H_out, W_out]   값이 클수록 배경(다른 패치와 비슷한 영역)
    """
    B, C, H, W = dino_feat.shape
    N = H * W

    patches = F.normalize(dino_feat.view(B, C, N).permute(0, 2, 1), dim=2)   # [B,N,C]
    sim = torch.bmm(patches, patches.transpose(1, 2))                         # [B,N,N]

    off_diag = ~torch.eye(N, dtype=torch.bool, device=dino_feat.device)
    agree = sim.masked_select(off_diag.unsqueeze(0)).view(B, N, N - 1).mean(dim=-1)

    up = F.interpolate(
        agree.view(B, H, W).unsqueeze(1), size=out_size,
        mode="bilinear", align_corners=False,
    ).squeeze(1)
    return _minmax(up, dims=(1, 2))


@torch.no_grad()
def dino_agreement_map(dino_feats, out_size):
    """
    objectness_map과 같은 개념의 시각화용 버전(대각 성분을 0으로 채워 평균).

    dino_feats: [B, C, H, W] -> [B, H_out, W_out] (0~1)
    """
    if isinstance(out_size, int):
        out_size = (out_size, out_size)

    B, C, H, W = dino_feats.shape
    L = H * W

    X = F.normalize(dino_feats.view(B, C, L), dim=1)       # [B,C,L]
    S = torch.bmm(X.transpose(1, 2), X)                    # [B,L,L]

    eye = torch.eye(L, device=S.device, dtype=torch.bool).unsqueeze(0)
    S = S.masked_fill(eye, 0.0)

    agree = (S.sum(dim=-1) / max(L - 1, 1)).view(B, H, W)
    up = F.interpolate(agree.unsqueeze(1), size=out_size, mode="bilinear", align_corners=False).squeeze(1)
    return _minmax(up, dims=(1, 2))


@torch.no_grad()
def dino_outlier_score(dino_feats, target_grid, topk=32):
    """
    패치별 kNN 유사도가 낮을수록 높은 outlier 점수.

    dino_feats: [B, C, Hd, Wd]
    return:     [B, target_grid^2]  (0~1, 클수록 이상 후보)
    """
    B, C, Hd, Wd = dino_feats.shape
    L = Hd * Wd

    X = F.normalize(dino_feats.view(B, C, L), dim=1)
    S = torch.bmm(X.transpose(1, 2), X)                    # [B,L,L]

    eye = torch.eye(L, device=S.device, dtype=torch.bool).unsqueeze(0)
    S = S.masked_fill(eye, -1e9)

    k = min(topk, L - 1)
    knn_sim = S.topk(k=k, dim=-1, largest=True).values.mean(dim=-1)   # [B,L]
    out = 1.0 - knn_sim

    out = F.interpolate(
        out.view(B, 1, Hd, Wd), size=(target_grid, target_grid),
        mode="bilinear", align_corners=False,
    ).view(B, -1)

    mn = out.min(dim=-1, keepdim=True).values
    mx = out.max(dim=-1, keepdim=True).values
    return (out - mn) / (mx - mn + 1e-6)


def grid_positions(G, device):
    """G x G 그리드의 패치 좌표. return: [G*G, 2] (y, x), 범위 [0, 1]"""
    coords = torch.linspace(0, 1, G, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    return torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)


def downsample_to_grid(obj_full, G, device):
    """픽셀 해상도 objectness [H, W]를 패치 그리드 [G*G]로 줄인다."""
    if obj_full is None:
        return torch.zeros(G * G, device=device)
    t = obj_full.to(device).unsqueeze(0).unsqueeze(0)      # [1,1,H,W]
    t = F.interpolate(t, size=(G, G), mode="bilinear", align_corners=False)[0, 0]
    return t.reshape(-1).contiguous()


# ----------------------------------------------------------------------
# 2. 패치 매칭: OT, mutual-kNN
# ----------------------------------------------------------------------


def sq_cdist(P, Q):
    """제곱 유클리드 거리. P: [N, 2], Q: [M, 2] -> [N, M]"""
    P2 = (P ** 2).sum(dim=1, keepdim=True)
    Q2 = (Q ** 2).sum(dim=1, keepdim=True).t()
    return (P2 + Q2 - 2 * (P @ Q.t())).clamp_min(0.0)


@torch.no_grad()
def sinkhorn_plan_batched(C, eps=0.05, iters=30):
    """
    균등 주변분포에 대한 엔트로피 정규화 OT 계획.

    C: [B, N, M] 비용 행렬 -> [B, N, M] 수송 계획
    """
    K = torch.exp(-C / eps).clamp_min(1e-12)
    B, N, M = C.shape
    a = C.new_full((B, N), 1.0 / max(N, 1))
    b = C.new_full((B, M), 1.0 / max(M, 1))
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for _ in range(int(iters)):
        u = a / torch.bmm(K, v.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
        v = b / torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
    return u.unsqueeze(-1) * K * v.unsqueeze(-2)


@torch.no_grad()
def ot_normality_batch(Zi, Zjs, Pi, Pj, Oi, Ojs, *,
                       ws=1.0, wp=0.05, wo=0.05, wb=0.10,
                       obj_tau=0.5, eps=0.05, iters=30):
    """
    OT 계획으로 가중 평균한 유사도 = 타깃 패치의 정상도.

    Zi:  [N, D]     타깃 패치
    Zjs: [B, M, D]  이웃 B장의 패치
    Pi:  [N, 2]     타깃 패치 좌표 (0~1)
    Pj:  [M, 2]     이웃 패치 좌표 (모든 이웃이 같은 그리드)
    Oi:  [N]        타깃 objectness
    Ojs: [B, M]     이웃 objectness
    ws/wp/wo/wb: 비용 가중치 (내용 / 위치 / objectness 차이 / 전경·배경 불일치)
    return: [B, N]  높을수록 정상
    """
    N = Zi.shape[0]
    B, M, _ = Zjs.shape

    Zi = F.normalize(Zi, dim=-1)
    Zjs = F.normalize(Zjs, dim=-1)
    S = torch.einsum("nd,bmd->bnm", Zi, Zjs)                # [B,N,M] 유사도

    C = ws * (1.0 - S)
    if wp > 0:
        C = C + wp * sq_cdist(Pi, Pj).unsqueeze(0)
    if wo > 0:
        C = C + wo * (Oi.view(1, N, 1) - Ojs.view(B, 1, M)).abs()
    if wb > 0:
        fi = (Oi > obj_tau).float().view(1, N, 1)
        fj = (Ojs > obj_tau).float().view(B, 1, M)
        C = C + wb * (fi - fj).abs()

    plan = sinkhorn_plan_batched(C, eps=eps, iters=iters)   # [B,N,M]
    row_mass = plan.sum(dim=2).clamp_min(1e-8)
    return ((plan * S).sum(dim=2) / row_mass).clamp(0, 1)


def _within_radius(Pi, Pj, G, radius):
    """체비셰프 거리로 ±radius 칸 이내인 (i, j) 쌍. return: [N, M] bool"""
    if Pi.max() <= 1.0 and Pj.max() <= 1.0:
        # 좌표가 0~1로 정규화된 경우
        limit = float(radius) / max(G - 1, 1)
    else:
        # 그리드 좌표(0..G-1)로 간주
        Pi = Pi if Pi.max() > 1.0 else Pi * (G - 1)
        Pj = Pj if Pj.max() > 1.0 else Pj * (G - 1)
        limit = radius
    dy = (Pi[:, 0:1] - Pj[:, 0].unsqueeze(0)).abs()
    dx = (Pi[:, 1:2] - Pj[:, 1].unsqueeze(0)).abs()
    return torch.maximum(dy, dx) <= limit


@torch.no_grad()
def mutual_maxsim(Zi, Zj, topk=5, tau=0.85, Pi=None, Pj=None, G=None, radius=0):
    """
    서로가 서로의 top-k에 드는(mutual) 패치 쌍 가운데 최대 유사도.

    radius > 0 이면 Pi, Pj 좌표 기준 ±radius 칸 이내에서만 매칭을 허용한다.
    return: [N]  mutual이 아니거나 tau 미만이면 0
    """
    Zi = F.normalize(Zi, dim=-1)
    Zj = F.normalize(Zj, dim=-1)
    Sij = Zi @ Zj.t()                                       # [N,M]

    if radius > 0:
        loc_ok = _within_radius(Pi, Pj, G, radius)
        neg_inf = Sij.new_full((), -1e9)
        Sij = torch.where(loc_ok, Sij, neg_inf.expand_as(Sij))

    N, M = Sij.shape
    _, idx_i = torch.topk(Sij, k=min(topk, M), dim=1, largest=True)       # [N,k]
    _, idx_j = torch.topk(Sij.t(), k=min(topk, N), dim=1, largest=True)   # [M,k]

    in_row_topk = torch.zeros((N, M), device=Zi.device, dtype=torch.bool)
    in_row_topk.scatter_(1, idx_i, True)

    in_col_topk = torch.zeros((N, M), device=Zi.device, dtype=torch.bool)
    cols = torch.arange(M, device=Zi.device).unsqueeze(1).expand_as(idx_j).reshape(-1)
    in_col_topk[idx_j.reshape(-1), cols] = True

    mutual = in_row_topk & in_col_topk & (Sij >= tau)
    return torch.where(mutual, Sij, Sij.new_zeros(1)).max(dim=1).values.clamp(0, 1)


# ----------------------------------------------------------------------
# 3. 이웃 집계, 픽셀 로짓 변환
# ----------------------------------------------------------------------


@torch.no_grad()
def aggregate_over_neighbors(S, tau, m, q):
    """
    이웃 R장의 정상도를 패치별로 하나의 값으로 모은다.

    S: [R, N]  이웃별 정상도
    tau 이상인 값이 m개 이상인 패치만, 그 값들의 상위 q 분위 지점을 쓴다(나머지는 0).
    return: [N]
    """
    valid = S.clone()
    valid[valid < tau] = -float("inf")
    support = torch.isfinite(valid).sum(dim=0)              # [N]

    out = torch.zeros(S.size(1), device=S.device)
    good = support >= max(int(m), 1)
    if good.any():
        sorted_vals, _ = torch.sort(valid[:, good], dim=0, descending=True)
        k = torch.ceil(q * support[good].float()).clamp_min(1).long() - 1
        col = torch.arange(k.numel(), device=S.device)
        qvals = sorted_vals[k, col]
        qvals[~torch.isfinite(qvals)] = 0
        out[good] = qvals.clamp_min(0)
    return out


@torch.no_grad()
def patch_scores_to_logits(patch_scores, grid, imagesize, sigma, device):
    """
    패치 이상 점수 [G*G]를 이미지 크기로 키우고 스무딩한 뒤 로짓으로 바꾼다.

    return: [1, H, W, 2]
    """
    amap = patch_scores.view(1, 1, grid, grid)
    amap = F.interpolate(amap, size=imagesize, mode="bilinear", align_corners=False)[0, 0]
    amap = torch.from_numpy(
        gaussian_filter(amap.detach().cpu().numpy(), sigma=float(sigma))
    ).to(device)
    return prob_to_logits(amap.unsqueeze(0)).to(device)
