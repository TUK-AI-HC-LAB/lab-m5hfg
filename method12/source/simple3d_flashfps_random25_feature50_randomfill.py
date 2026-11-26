"""Replace only the ordered fill of feature FPS with a uniform random fill."""
import torch

from simple3d_flashfps import KEEP
from simple3d_flashfps_random25 import SEED
from simple3d_flashfps_random25_feature50 import install as install_reference

FILL_SEED = 43
DIRECT = 2048
TOTAL = 4096


def install(native_fps):
    reference, metadata = install_reference(native_fps)
    sets = {}
    checked = False

    def select(xyz, count):
        nonlocal checked
        if count != TOTAL:
            return reference(xyz, count)
        if xyz.shape != (1, 500000, 3) or not xyz.is_cuda or not xyz.is_contiguous() or xyz.dtype != torch.float32:
            raise ValueError('Expected a full-resolution capsule cloud on CUDA')
        key = (xyz.device, xyz.shape[1])
        if key not in sets:
            generator = torch.Generator(device=xyz.device).manual_seed(SEED)
            n = int(xyz.shape[1] * KEEP)
            candidates = torch.randperm(xyz.shape[1], device=xyz.device, generator=generator)[:n].sort().values
            generator = torch.Generator(device=xyz.device).manual_seed(FILL_SEED)
            # With only 2048 excluded positions, the first 4096 in a
            # uniform permutation always contain at least 2048 eligible ones.
            priority = torch.randperm(n, device=xyz.device, generator=generator)[:TOTAL]
            sets[key] = candidates, priority
        candidates, priority = sets[key]
        direct = native_fps(xyz.index_select(1, candidates), DIRECT).long()
        used = torch.zeros((1, candidates.numel()), dtype=torch.bool, device=xyz.device)
        used.scatter_(1, direct, True)
        fill = priority[~used[0, priority]][:TOTAL-DIRECT]
        result = candidates[torch.cat((direct, fill[None, :]), dim=1)].to(torch.int32).contiguous()
        if not checked:
            expected = reference(xyz, TOTAL)
            if not torch.equal(expected[:, :DIRECT], result[:, :DIRECT]) or torch.equal(expected[:, DIRECT:], result[:, DIRECT:]):
                raise RuntimeError('Random fill must preserve FPS prefix and differ from ordered fill')
            if result.shape != (1, TOTAL) or result.unique().numel() != TOTAL:
                raise RuntimeError('Unexpected random fill output shape or duplicates')
            checked = True
        return result

    metadata.update(feature_fill='uniform without replacement from remaining candidates',
                    feature_fill_seed=FILL_SEED, feature_fill_count=TOTAL-DIRECT,
                    fill_priority_reuse='fixed per point count',
                    feature_fps_prefix_audit='first warmup call must match ordered-fill reference')
    return select, metadata
