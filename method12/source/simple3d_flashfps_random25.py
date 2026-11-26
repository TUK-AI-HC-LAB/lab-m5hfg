"""Isolate the candidate-set effect in upstream FlashFPS FPS-Prune."""
import torch
from simple3d_flashfps import KEEP, install as install_prefix

SEED = 42


def install(native_fps):
    candidates = {}
    active = None

    def fps_on_random_candidates(prefix, count):
        assert active is not None and prefix.shape == active.shape
        return native_fps(active, count)

    prune, metadata = install_prefix(fps_on_random_candidates)

    def sample(xyz, count):
        nonlocal active
        n = xyz.shape[1]
        key = (xyz.device, n)
        if key not in candidates:
            generator = torch.Generator(device=xyz.device).manual_seed(SEED)
            # Sort the sampled index set to preserve the original relative order
            # for the existing first-point and fill rules.
            candidates[key] = torch.randperm(n, device=xyz.device, generator=generator)[:int(n * KEEP)].sort().values
        index = candidates[key]
        active = xyz.index_select(1, index)
        try:
            # The official algorithm still does the same iteration pruning and
            # fills unused indices within the candidate set in its original order.
            local = prune(xyz, count)
            return index[local.long()].to(torch.int32).contiguous()
        finally:
            active = None

    metadata.update(candidate_selection='uniform without replacement over all point indices',
                    candidate_seed=SEED, candidate_order='original relative order',
                    reuse_candidate_indices='fixed per point count across clouds and both FPS calls')
    return sample, metadata
