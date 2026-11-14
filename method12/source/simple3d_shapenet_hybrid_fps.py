"""Capsule hybrid recipe, with an exact feature FPS budget for variable N."""
from functools import lru_cache

import torch


def install(native_fps, variant):
    if variant == 'original_fps':
        return native_fps, {'variant': variant, 'candidate_fraction': 1.0,
                            'feature_exact': 4096, 'smoothing_exact': 1024}
    if variant != 'hybrid':
        raise ValueError(variant)

    @lru_cache(maxsize=8)
    def candidate_indices(device, n):
        generator = torch.Generator(device=device).manual_seed(42)
        return torch.randperm(n, generator=generator, device=device)[:n//4].sort().values

    def sample(xyz, count):
        if count == 1024:
            return native_fps(xyz, count)
        if count != 4096 or xyz.shape[0] != 1 or xyz.shape[1]//4 < count:
            raise ValueError(f'Cannot produce {count} centers from shape {xyz.shape}')
        idx = candidate_indices(xyz.device, xyz.shape[1])
        direct = native_fps(xyz.index_select(1, idx), 2048).long()
        unused = torch.ones(idx.numel(), device=idx.device, dtype=torch.bool)
        unused[direct[0]] = False
        fill = torch.nonzero(unused, as_tuple=True)[0][:2048]
        result = idx[torch.cat((direct, fill.unsqueeze(0)), dim=1)]
        if result.shape != (1, count):
            raise RuntimeError('Incorrect feature center count')
        return result.to(torch.int32).contiguous()

    return sample, {'variant': variant, 'candidate_fraction': 0.25, 'candidate_seed': 42,
                    'feature_exact': 2048, 'feature_fill': 2048,
                    'fill_order': 'original candidate index order',
                    'smoothing_exact': 1024, 'smoothing_candidate_fraction': 1.0}
