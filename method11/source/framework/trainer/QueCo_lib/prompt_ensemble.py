"""
텍스트 브랜치: 이미지에서 뽑은 프로토타입으로 정상/이상 프롬프트를 만든다(DPCS).

    tokenize        문장 -> CLIP 토큰
    PromptLearner   패치 특징 F와 이미지 특징 g로 프롬프트 생성 (학습 없음)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List
from pkg_resources import packaging
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    if isinstance(texts, str):
        texts = [texts]
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)
    return result

class PromptLearner(nn.Module):
    def __init__(self, clip_model, design_details):
        super().__init__()
        dtype = clip_model.transformer.get_cast_dtype()

        # ---------------- Common Settings ----------------
        self.text_encoder_n_ctx = design_details["learnabel_text_embedding_length"]
        self.n_ctx = design_details["Prompt_length"]
        
        # DPCS Context Split (Dual Prototype-Conditioned Prompt Synthesis)
        self.K = int(design_details.get("ctx_K", 0))   
        self.M = int(design_details.get("ctx_M", self.n_ctx // 2))
        self.N = self.n_ctx - self.K - self.M

        self.n_ctx_pos = self.n_ctx
        self.n_ctx_neg = self.n_ctx
        self.classnames = ["object"]
        self.n_cls = len(self.classnames)
        self.state_normal_list = ["{}"]
        self.state_anomaly_list = ["damaged {}"]
        self.normal_num = len(self.state_normal_list)
        self.anormaly_num = len(self.state_anomaly_list)
        
        self.ctx_dim = clip_model.ln_final.weight.shape[0]

        # ---------------- DPCS Specific Params ----------------
        # 1. K-Means / Prototype related
        self.proto_top_p = design_details.get("proto_top_p", 0.2)
        
        # 2. Alpha scaling factors
        runtime_args = design_details["others"]

        self.register_buffer(
            "alpha_n",
            torch.tensor(
                float(getattr(runtime_args, "alpha_n", 0.10)),
                dtype=dtype
            )
        )

        self.register_buffer(
            "alpha_a",
            torch.tensor(
                float(getattr(runtime_args, "alpha_a", 0.20)),
                dtype=dtype
            )
        )

        self.register_buffer(
            "alpha_g",
            torch.tensor(
                float(getattr(runtime_args, "alpha_g", 0.10)),
                dtype=dtype
            )
        )

        # ---------------- Reviewer experiments: prototype injection ----------------
        # direct / none / shuffled / random / linear
        self.prototype_mode = str(getattr(runtime_args, "prototype_mode", "direct")).lower()
        self.prototype_random_seed = int(getattr(runtime_args, "prototype_random_seed", 42))
        self.prototype_projection_ckpt = str(
            getattr(runtime_args, "prototype_projection_ckpt", "") or ""
        )
        self.enable_prompt_probe = bool(getattr(runtime_args, "enable_prompt_probe", False))

        # A deterministic orthogonal signed-permutation projection.
        # If --prototype_projection_ckpt is given, a D x D matrix can replace it.
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.prototype_random_seed)
        perm = torch.randperm(self.ctx_dim, generator=gen)
        signs = torch.where(
            torch.rand(self.ctx_dim, generator=gen) > 0.5,
            torch.ones(self.ctx_dim),
            -torch.ones(self.ctx_dim),
        )
        projection = torch.zeros(self.ctx_dim, self.ctx_dim, dtype=torch.float32)
        projection[torch.arange(self.ctx_dim), perm] = signs

        if self.prototype_projection_ckpt:
            ckpt = torch.load(self.prototype_projection_ckpt, map_location="cpu")
            if isinstance(ckpt, dict):
                for key in ("weight", "projection", "visual_to_text", "state_dict"):
                    if key in ckpt:
                        ckpt = ckpt[key]
                        break
                if isinstance(ckpt, dict):
                    for key in ("weight", "visual_to_text.weight", "projection.weight"):
                        if key in ckpt:
                            ckpt = ckpt[key]
                            break
            ckpt = torch.as_tensor(ckpt, dtype=torch.float32)
            if ckpt.shape != (self.ctx_dim, self.ctx_dim):
                raise ValueError(
                    f"Projection checkpoint must have shape "
                    f"({self.ctx_dim}, {self.ctx_dim}), got {tuple(ckpt.shape)}"
                )
            projection = ckpt

        self.register_buffer("prototype_projection", projection.to(dtype=dtype))
        self.last_prompt_probe = None


        # ---------------- Prompt Initialization ----------------
        self.compound_prompts_depth = design_details["learnabel_text_embedding_depth"]
        self.compound_prompts_text = nn.ParameterList([
            nn.Parameter(torch.zeros(self.text_encoder_n_ctx, self.ctx_dim, dtype=dtype), requires_grad=False)
            for _ in range(self.compound_prompts_depth - 1)
        ])

        # NOTE: 학습형 프롬프트의 흔적. DPCS에서는 쓰지 않지만, 여기서 난수를 소비하므로
        #       지우면 이후 k-means의 난수 순서가 달라져 결과 수치가 미세하게 바뀐다.
        #       수치를 다시 뽑아도 될 때 이 블록(6줄)을 지우면 된다.
        ctx_vectors_pos = torch.empty(self.n_cls, self.normal_num, self.n_ctx_pos, self.ctx_dim, dtype=dtype)
        ctx_vectors_neg = torch.empty(self.n_cls, self.anormaly_num, self.n_ctx_pos, self.ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors_pos, std=0.02)
        nn.init.normal_(ctx_vectors_neg, std=0.02)
        self.ctx_pos = nn.Parameter(ctx_vectors_pos)
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)

        # Tokenized Prompts & Fixed Embeddings
        prompt_prefix_pos = [" ".join(["X"] * self.n_ctx_pos)]
        prompt_prefix_neg = [" ".join(["X"] * self.n_ctx_neg)]
        classnames = [name.replace("_", " ") for name in self.classnames]
        
        prompts_pos = [prompt_prefix_pos[idx] + " " + template.format(name) + "." for idx, template in enumerate(self.state_normal_list) for name in classnames]
        prompts_neg = [prompt_prefix_neg[idx] + " " + template.format(name) + "." for idx, template in enumerate(self.state_anomaly_list) for name in classnames]

        tokenized_prompts_pos = [tokenize(p_pos) for p_pos in prompts_pos]
        tokenized_prompts_neg = [tokenize(p_neg) for p_neg in prompts_neg]
        tokenized_prompts_pos = torch.cat(tokenized_prompts_pos)
        tokenized_prompts_neg = torch.cat(tokenized_prompts_neg)

        with torch.no_grad():
            embedding_pos = clip_model.token_embedding(tokenized_prompts_pos).type(dtype)
            embedding_neg = clip_model.token_embedding(tokenized_prompts_neg).type(dtype)
            n, l, d = embedding_pos.shape
            embedding_pos = embedding_pos.reshape(self.normal_num, self.n_cls, l, d).permute(1, 0, 2, 3)
            embedding_neg = embedding_neg.reshape(self.anormaly_num, self.n_cls, l, d).permute(1, 0, 2, 3)

        n, d = tokenized_prompts_pos.shape
        tokenized_prompts_pos = tokenized_prompts_pos.reshape(self.normal_num, self.n_cls, d).permute(1, 0, 2)
        n, d = tokenized_prompts_neg.shape
        tokenized_prompts_neg = tokenized_prompts_neg.reshape(self.anormaly_num, self.n_cls, d).permute(1, 0, 2)

        self.register_buffer("tokenized_prompts_pos", tokenized_prompts_pos)
        self.register_buffer("tokenized_prompts_neg", tokenized_prompts_neg)
        
        self.register_buffer("token_prefix_pos", embedding_pos[:, :, :1, :])
        self.register_buffer("token_suffix_pos", embedding_pos[:, :, 1 + self.n_ctx_pos:, :])
        self.register_buffer("token_prefix_neg", embedding_neg[:, :, :1, :])
        self.register_buffer("token_suffix_neg", embedding_neg[:, :, 1 + self.n_ctx_neg:, :])
        
        # 프로토타입 잔차를 더할 기준 토큰 ('X' 자리의 CLIP 임베딩)
        self.register_buffer("token_ctx_pos", embedding_pos[:, :, 1:1+self.n_ctx_pos, :])
        self.register_buffer("token_ctx_neg", embedding_neg[:, :, 1:1+self.n_ctx_neg, :])

        self.constructed_prompts = None

    # =====================================================================
    #  프로토타입 추출
    # =====================================================================
    def _run_kmeans(self, x: torch.Tensor, num_clusters: int, num_iters: int = 5):
        B, N, D = x.shape
        rand_idx = torch.randperm(N, device=x.device)[:num_clusters]
        centroids = x[:, rand_idx, :]
        for _ in range(num_iters):
            dist = torch.norm(x.unsqueeze(2) - centroids.unsqueeze(1), dim=-1)
            labels = torch.argmin(dist, dim=-1)
            new_centroids = []
            for k in range(num_clusters):
                mask = (labels == k).unsqueeze(-1).float()
                cluster_sum = (x * mask).sum(dim=1)
                cluster_count = mask.sum(dim=1) + 1e-6
                new_centroids.append(cluster_sum / cluster_count)
            centroids = torch.stack(new_centroids, dim=1)
        return centroids

    def _local_prototypes_fallback(self, F_patch: torch.Tensor, g: torch.Tensor):
        """Blind Injection Mode (동일 정보 주입)"""
        B, N, D = F_patch.shape
        # 전체 이미지 대표 토큰 추출
        P_common = self._run_kmeans(F_patch, self.M)
        # Normal/Abnormal에 동일하게 주입
        Pn_tokens = P_common
        Pa_tokens = P_common
        return Pn_tokens, Pa_tokens

    def _local_prototypes(self, F_patch, g=None, dino_score=None):
        """Hybrid Selection Mode (선별 주입)"""
        B, N, D = F_patch.shape
        
        if dino_score is None:
            return self._local_prototypes_fallback(F_patch, g)

        if dino_score.dim() == 1:
            dino_score = dino_score.unsqueeze(0)
        
        # Centrality Calculation
        F_norm = torch.nn.functional.normalize(F_patch, dim=-1)
        sim_matrix = torch.matmul(F_norm, F_norm.transpose(1, 2))
        centrality = sim_matrix.mean(dim=-1)
        c_min = centrality.min(dim=-1, keepdim=True)[0]
        c_max = centrality.max(dim=-1, keepdim=True)[0]
        centrality_norm = (centrality - c_min) / (c_max - c_min + 1e-6)

        # Ensemble Score
        combined_anomaly_score = 0.5 * dino_score + 0.5 * (1.0 - centrality_norm)
        
        # Foreground Gating
        if g is not None:
            g_n = torch.nn.functional.normalize(g, dim=-1).unsqueeze(1)
            sim_g = (F_norm * g_n).sum(dim=-1)
            sim_g = (sim_g + 1.0) * 0.5
        else:
            sim_g = torch.ones((B, N), device=F_patch.device)

        score_a = combined_anomaly_score * sim_g
        score_n = (1.0 - combined_anomaly_score) * sim_g

        # Top-K Selection
        p = float(self.proto_top_p)
        k = max(self.M * 5, int(p * N)) 
        idx_n = score_n.topk(k, dim=-1).indices
        idx_a = score_a.topk(k, dim=-1).indices
        
        F_n_sel = torch.gather(F_patch, 1, idx_n.unsqueeze(-1).expand(-1, -1, D))
        F_a_sel = torch.gather(F_patch, 1, idx_a.unsqueeze(-1).expand(-1, -1, D))

        # K-Means
        Pn_tokens = self._run_kmeans(F_n_sel, self.M)
        Pa_tokens = self._run_kmeans(F_a_sel, self.M)

        return Pn_tokens, Pa_tokens

    def _apply_prototype_mode(self, Pn_tokens, Pa_tokens, mode, F_patch):
        """Apply reviewer-control transformations without changing tensor shape."""
        mode = str(mode).lower()

        if mode == "direct":
            return Pn_tokens, Pa_tokens

        if mode == "none":
            return torch.zeros_like(Pn_tokens), torch.zeros_like(Pa_tokens)

        if mode == "shuffled":
            # Batch size is normally one. To create a semantic-mismatch control,
            # exchange normal/anomaly assignments and reverse context-slot order.
            return torch.flip(Pa_tokens, dims=[1]), torch.flip(Pn_tokens, dims=[1])

        if mode == "random":
            # Norm-matched random directions: preserves prototype magnitude but
            # removes query-specific semantic direction.
            gen = torch.Generator(device=Pn_tokens.device)
            gen.manual_seed(self.prototype_random_seed)
            rn = torch.randn(Pn_tokens.shape, generator=gen, device=Pn_tokens.device, dtype=Pn_tokens.dtype)
            ra = torch.randn(Pa_tokens.shape, generator=gen, device=Pa_tokens.device, dtype=Pa_tokens.dtype)
            rn = F.normalize(rn, dim=-1) * Pn_tokens.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            ra = F.normalize(ra, dim=-1) * Pa_tokens.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            return rn, ra

        if mode == "linear":
            W = self.prototype_projection.to(device=Pn_tokens.device, dtype=Pn_tokens.dtype)
            return Pn_tokens @ W.t(), Pa_tokens @ W.t()

        raise ValueError(
            f"Unsupported prototype_mode={mode}. "
            "Choose direct, none, shuffled, random, or linear."
        )

    @staticmethod
    def _mean_token_cosine(base_tokens, conditioned_tokens):
        base = F.normalize(base_tokens.float(), dim=-1)
        cond = F.normalize(conditioned_tokens.float(), dim=-1)
        return (base * cond).sum(dim=-1).mean()

    # =====================================================================
    #  DPCS: 프로토타입 기반 프롬프트 합성
    # =====================================================================
    def _forward_dpcs(self, F, g, dino_score=None, prototype_mode_override=None):
        B, P, D = F.shape
        device = F.device
        dtype = self.token_prefix_pos.dtype

        # 1. Expand Basic Components
        prefix_pos = self.token_prefix_pos.to(device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1, -1, -1)
        suffix_pos = self.token_suffix_pos.to(device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1, -1, -1)
        prefix_neg = self.token_prefix_neg.to(device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1, -1, -1)
        suffix_neg = self.token_suffix_neg.to(device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1, -1, -1)

        tokenized_prompts_pos = self.tokenized_prompts_pos.to(device).unsqueeze(0).repeat(B, 1, 1, 1)
        tokenized_prompts_neg = self.tokenized_prompts_neg.to(device).unsqueeze(0).repeat(B, 1, 1, 1)

        # 2. Use Fixed Context (CLIP 'X' tokens) as base for residuals
        ctx_pos = self.token_ctx_pos.to(device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1, -1, -1).clone()
        ctx_neg = self.token_ctx_neg.to(device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1, -1, -1).clone()

        v_pos, w_pos = ctx_pos[:, :, :, :self.M, :], ctx_pos[:, :, :, self.M:, :]
        v_neg, w_neg = ctx_neg[:, :, :, :self.M, :], ctx_neg[:, :, :, self.M:, :]

        # 3. Local prototype injection / reviewer controls
        Pn_raw, Pa_raw = self._local_prototypes(
            F.to(dtype=dtype), g.to(dtype=dtype), dino_score
        )
        mode = (
            str(prototype_mode_override).lower()
            if prototype_mode_override is not None
            else self.prototype_mode
        )
        Pn_tokens, Pa_tokens = self._apply_prototype_mode(Pn_raw, Pa_raw, mode, F)

        base_v_pos = v_pos.clone()
        base_v_neg = v_neg.clone()
        base_w_pos = w_pos.clone()
        base_w_neg = w_neg.clone()

        local_enabled = mode != "none"
        global_enabled = mode != "none"

        if local_enabled:
            v_pos = v_pos + self.alpha_n * Pn_tokens.unsqueeze(1).unsqueeze(2)
            v_neg = v_neg + self.alpha_a * Pa_tokens.unsqueeze(1).unsqueeze(2)

        # 4. Global feature injection
        g_norm = torch.nn.functional.normalize(g.to(dtype=dtype), dim=-1)
        w_add = g_norm.view(B, 1, 1, 1, D).repeat(1, 1, 1, self.N, 1)
        if global_enabled:
            w_pos = w_pos + self.alpha_g * w_add
            w_neg = w_neg + self.alpha_g * w_add

        # Experiment A: input-token drift before the text encoder.
        self.last_prompt_probe = {
            "prototype_mode": mode,
            "token_cosine_normal_local": float(
                self._mean_token_cosine(base_v_pos, v_pos).detach().cpu()
            ),
            "token_cosine_anomaly_local": float(
                self._mean_token_cosine(base_v_neg, v_neg).detach().cpu()
            ),
            "token_cosine_normal_global": float(
                self._mean_token_cosine(base_w_pos, w_pos).detach().cpu()
            ),
            "token_cosine_anomaly_global": float(
                self._mean_token_cosine(base_w_neg, w_neg).detach().cpu()
            ),
            "prototype_norm_normal": float(Pn_raw.float().norm(dim=-1).mean().detach().cpu()),
            "prototype_norm_anomaly": float(Pa_raw.float().norm(dim=-1).mean().detach().cpu()),
        }
        # 5. Concatenate
        ctx_pos = torch.cat([v_pos, w_pos], dim=3)
        ctx_neg = torch.cat([v_neg, w_neg], dim=3)

        prompts_pos = torch.cat([prefix_pos, ctx_pos, suffix_pos], dim=3)
        prompts_neg = torch.cat([prefix_neg, ctx_neg, suffix_neg], dim=3)

        # Reshape & Stack
        l, d = prompts_pos.shape[-2:]
        prompts_pos = prompts_pos.reshape(-1, l, d)
        prompts_neg = prompts_neg.reshape(-1, l, d)
        
        _, L = tokenized_prompts_pos.shape[-2:]
        tokenized_prompts_pos = tokenized_prompts_pos.reshape(-1, L)
        tokenized_prompts_neg = tokenized_prompts_neg.reshape(-1, L)

        prompts = torch.stack([prompts_pos, prompts_neg], dim=0)
        tokenized_prompts = torch.stack([tokenized_prompts_pos, tokenized_prompts_neg], dim=0)

        return prompts, tokenized_prompts, self.compound_prompts_text, True

    # =====================================================================
    #  Forward
    # =====================================================================
    def forward(self, F=None, g=None, dino_score=None, prototype_mode_override=None, **kwargs):
        """
        F: [B, N, D] 패치 특징, g: [B, D] 이미지 특징, dino_score: [B, N] (선택)
        return: (prompts, tokenized_prompts, compound_prompts_text, True)
                prompts[0] = 정상, prompts[1] = 이상
        """
        if dino_score is None and ("dino_socre" in kwargs):
            dino_score = kwargs["dino_socre"]

        if F is None or g is None:
            raise ValueError("PromptLearner requires both patch features (F) and the image feature (g).")

        self.constructed_prompts = self._forward_dpcs(
            F=F, g=g, dino_score=dino_score,
            prototype_mode_override=prototype_mode_override,
        )
        return self.constructed_prompts