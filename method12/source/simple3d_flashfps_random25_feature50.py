"""Change only exact FPS iterations for 4096-point feature aggregation."""
from simple3d_flashfps_random25 import install as install_random25


def install(native_fps):
    output_count = None

    def adjusted_native_fps(candidate_points, count):
        if output_count == 4096:
            if count != 1024:
                raise RuntimeError(f'Unexpected original feature FPS count: {count}')
            count = 2048
        elif output_count != 1024 or count != 256:
            raise RuntimeError(f'Unexpected smoothing FPS request: {output_count}, {count}')
        return native_fps(candidate_points, count)

    sample, metadata = install_random25(adjusted_native_fps)

    def select(xyz, count):
        nonlocal output_count
        output_count = count
        try:
            # Upstream rearrange_indices reads the actual inner output length,
            # so only the number of indices filled without FPS decreases.
            return sample(xyz, count)
        finally:
            output_count = None

    metadata.update(feature_native_fps_count=2048, smoothing_native_fps_count=256,
                    feature_iteration_prune_rate=0.50, smoothing_iteration_prune_rate=0.75,
                    candidate_prune_rate=0.75)
    return select, metadata
