"""
Decoder variants for Dinomaly.
Registry pattern: DECODER_REGISTRY[args.decoder_type] → class
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from trainer.Dinomaly_lib.models.vision_transformer import Block as VitBlock, LinearAttention2


class DefaultDecoder(nn.Module):
    """Original Dinomaly decoder: 8x VitBlock with LinearAttention2."""

    def __init__(self, embed_dim=768, num_heads=12, num_layers=8, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList([
            VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                     qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
                     attn_drop=0., attn=LinearAttention2)
            for _ in range(num_layers)
        ])

    def forward(self, x, attn_mask=None):
        de_list = []
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)
            de_list.append(x)
        return de_list[::-1]


class PruningDecoder(nn.Module):
    """Decoder with token pruning between layers.

    Based on TPS (CVPR23) and SViT (WACV24):
    - After the first N layers, compute token importance
    - Prune low-importance tokens, squeeze their info into nearest kept tokens
    - Remaining layers operate on fewer tokens (quadratic speedup)
    - Pruned token positions are restored via saved features for anomaly map
    """

    def __init__(self, embed_dim=768, num_heads=12, num_layers=8,
                 pruning_ratio=0.5, pruning_after_layer=2, **kwargs):
        super().__init__()
        self.pruning_ratio = pruning_ratio
        self.pruning_after_layer = pruning_after_layer

        self.blocks = nn.ModuleList([
            VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                     qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
                     attn_drop=0., attn=LinearAttention2)
            for _ in range(num_layers)
        ])

    def forward(self, x, attn_mask=None):
        de_list = []
        B, N, C = x.shape
        pruned_info = None  # (pruned_features, pruned_indices, kept_indices)

        for i, blk in enumerate(self.blocks):
            x = blk(x, attn_mask=attn_mask)
            de_list.append(x)

            # Prune after designated layer (once)
            if i == self.pruning_after_layer - 1 and pruned_info is None:
                x, pruned_info = self._prune_and_squeeze(x)
                # Disable attn_mask after pruning (token count changed)
                attn_mask = None

        # Restore pruned tokens for anomaly map reconstruction
        if pruned_info is not None:
            de_list = self._restore_pruned(de_list, pruned_info, B, N, C)

        return de_list[::-1]

    def _prune_and_squeeze(self, x):
        """Prune tokens and squeeze pruned info into kept tokens (TPS-style)."""
        B, N, C = x.shape

        # Handle class/register tokens: first token is cls, rest are patch tokens
        has_cls = True
        if has_cls:
            cls_token = x[:, :1]
            patch_tokens = x[:, 1:]
            N_patch = N - 1
        else:
            patch_tokens = x
            N_patch = N

        # Compute importance: L2 norm of each token (proxy for information content)
        importance = patch_tokens.norm(dim=-1)  # (B, N_patch)

        # Number of tokens to keep
        n_keep = max(int(N_patch * (1 - self.pruning_ratio)), 1)

        # Top-k important tokens
        _, keep_idx = importance.topk(n_keep, dim=-1, sorted=False)  # (B, n_keep)
        _, prune_idx = importance.topk(N_patch - n_keep, dim=-1, largest=False, sorted=False)

        # Gather kept and pruned tokens
        keep_idx_sorted, _ = keep_idx.sort(dim=-1)
        prune_idx_sorted, _ = prune_idx.sort(dim=-1)

        kept_tokens = torch.gather(patch_tokens, 1,
                                   keep_idx_sorted.unsqueeze(-1).expand(-1, -1, C))
        pruned_tokens = torch.gather(patch_tokens, 1,
                                     prune_idx_sorted.unsqueeze(-1).expand(-1, -1, C))

        # Squeeze: for each pruned token, find nearest kept token and add weighted info
        # Cosine similarity between pruned and kept tokens
        pruned_norm = F.normalize(pruned_tokens, dim=-1)
        kept_norm = F.normalize(kept_tokens, dim=-1)
        sim = torch.bmm(pruned_norm, kept_norm.transpose(1, 2))  # (B, n_prune, n_keep)
        nearest = sim.argmax(dim=-1)  # (B, n_prune)

        # Accumulate pruned token info into nearest kept token (vectorized)
        squeezed = kept_tokens.clone()
        # scatter_add: accumulate pruned token features into their nearest kept token
        target_idx = nearest.unsqueeze(-1).expand(-1, -1, C)  # (B, n_prune, C)
        squeezed.scatter_add_(1, target_idx, pruned_tokens * 0.5)

        # Reconstruct with cls token
        if has_cls:
            x_pruned = torch.cat([cls_token, squeezed], dim=1)
        else:
            x_pruned = squeezed

        pruned_info = {
            'pruned_features': pruned_tokens.detach(),
            'keep_idx': keep_idx_sorted,
            'prune_idx': prune_idx_sorted,
            'N_patch': N_patch,
            'has_cls': has_cls,
        }
        return x_pruned, pruned_info

    def _restore_pruned(self, de_list, pruned_info, B, N, C):
        """Restore full-length token sequence for anomaly map generation."""
        keep_idx = pruned_info['keep_idx']
        prune_idx = pruned_info['prune_idx']
        N_patch = pruned_info['N_patch']
        has_cls = pruned_info['has_cls']

        restored_list = []
        for layer_idx, feat in enumerate(de_list):
            if layer_idx <= self.pruning_after_layer - 1:
                # Pre-pruning layers: already full length
                restored_list.append(feat)
            else:
                # Post-pruning layers: restore by scattering
                if has_cls:
                    cls_feat = feat[:, :1]
                    patch_feat = feat[:, 1:]
                else:
                    patch_feat = feat

                full_patch = torch.zeros(B, N_patch, C, device=feat.device, dtype=feat.dtype)
                full_patch.scatter_(1, keep_idx.unsqueeze(-1).expand(-1, -1, C), patch_feat)
                # Pruned positions: use last pre-pruning layer's features (SViT reactivation)
                pre_prune_feat = de_list[self.pruning_after_layer - 1]
                if has_cls:
                    pre_prune_patch = pre_prune_feat[:, 1:]
                else:
                    pre_prune_patch = pre_prune_feat
                pruned_fill = torch.gather(pre_prune_patch, 1,
                                           prune_idx.unsqueeze(-1).expand(-1, -1, C))
                full_patch.scatter_(1, prune_idx.unsqueeze(-1).expand(-1, -1, C), pruned_fill)

                if has_cls:
                    full_feat = torch.cat([cls_feat, full_patch], dim=1)
                else:
                    full_feat = full_patch
                restored_list.append(full_feat)

        return restored_list


DECODER_REGISTRY = {
    "default": DefaultDecoder,
    "token_pruning": PruningDecoder,
}
