"""Change only the 768 non-FPS smoothing centers to seeded random fill."""
import torch

from simple3d_flashfps_random25_feature50 import install as install_reference


def install(native_fps):
    reference, metadata = install_reference(native_fps)
    cache = {}

    def select(xyz, count):
        ordered = reference(xyz, count)
        if count == 4096:
            return ordered
        if count != 1024 or xyz.shape != (1, 500000, 3):
            raise RuntimeError(f'Unexpected smoothing FPS input: {xyz.shape}, {count}')
        key = (xyz.device, xyz.shape[1])
        if key not in cache:
            candidates = torch.randperm(xyz.shape[1], device=xyz.device,
                                        generator=torch.Generator(device=xyz.device).manual_seed(42))[:xyz.shape[1] // 4].sort().values
            priority = torch.randperm(candidates.numel(), device=xyz.device,
                                      generator=torch.Generator(device=xyz.device).manual_seed(43))
            cache[key] = candidates, priority
        candidates, priority = cache[key]
        direct = ordered[:, :256].long()
        used = torch.isin(candidates, direct[0])
        fill = candidates[priority[~used[priority]][:768]]
        result = torch.cat((direct, fill.unsqueeze(0)), 1).to(torch.int32).contiguous()
        assert torch.equal(result[:, :256], ordered[:, :256])
        assert result.unique().numel() == 1024
        return result

    metadata.update(smoothing_fill='random without replacement from unused candidates',
                    smoothing_fill_seed=43, smoothing_fill_count=768)
    return select, metadata
