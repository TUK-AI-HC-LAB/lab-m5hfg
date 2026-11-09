"""Adapter for the official FlashFPS FPS-Prune functions.

Load the two upstream functions verbatim without importing OpenPoints or its
unrelated extensions. Their standard_fps hook uses the already built Simple3D
PointNet2 CUDA operator; keep Simple3D's fixed output size and int32 ABI.
"""
import ast
import hashlib
from pathlib import Path

import torch

UPSTREAM = Path(__file__).resolve().parents[3] / (
    'Project/FlashFPS/FlashFPS-Openpoints/openpoints/models/layers/subsample.py'
)
UPSTREAM_COMMIT = 'f32a4b7886501873f2d7d380dc84f07ae77492df'
KEEP = 0.25


def install(native_fps):
    upstream = UPSTREAM.read_bytes()
    tree = ast.parse(upstream, filename=str(UPSTREAM))
    names = {'FPS_Prune', 'rearrange_indices'}
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != names:
        raise RuntimeError('Missing upstream FPS-Prune functions')
    namespace = {'torch': torch, 'standard_fps': native_fps}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(UPSTREAM), 'exec'), namespace)
    prune = namespace['FPS_Prune']

    def sample(xyz, npoint):
        if not xyz.is_cuda or xyz.dtype != torch.float32 or not xyz.is_contiguous():
            raise ValueError('Expected contiguous float32 CUDA point cloud')
        if xyz.shape[1] < npoint * 4:
            raise ValueError('Experiment requires a full-resolution cloud')
        indices = prune(xyz, npoint, PruneRage=KEEP)
        # Upstream rounds via an integer stride: 500k/4096 can give 4098.
        # Retain the FPS-ordered prefix and trim surplus fill indices only.
        if indices.shape[1] < npoint:
            raise RuntimeError('Upstream FPS-Prune returned too few indices')
        return indices[:, :npoint].to(torch.int32).contiguous()

    meta = dict(upstream_commit=UPSTREAM_COMMIT, upstream_sha256=hashlib.sha256(upstream).hexdigest(),
                method='FPS-Prune', prune_rate=1-KEEP, fps_cache=False,
                native_inner_fps='Simple3D PointNet2 CUDA', input_order='original',
                fixed_output_counts=[4096, 1024])
    return sample, meta
