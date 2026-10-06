"""Training-free MuSc using the framework's unchanged inputs and evaluation.

LNAMD, mutual scoring and RsCIN follow the ICLR 2024 MuSc implementation.
Test images from one category form an unlabeled, transductive reference pool.
"""
import logging
import math

import clip
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import SequentialSampler

from .trainer import Trainer

logger = logging.getLogger(__name__)


class Trainer_MuSc(Trainer):
    def load(self, backbone, device, input_shape, args, **kwargs):
        self.args, self.device, self.backbone = args, torch.device(device), backbone
        self.input_shape = input_shape
        if getattr(args, 'subtest', False):
            raise ValueError('MuSc requires the complete category test pool; disable subtest.')
        self.layers = tuple(args.musc_layers)
        self.radii = tuple(args.musc_radii)
        self.fraction = float(args.musc_reference_fraction)
        self.neighbors = tuple(args.musc_score_neighbors)
        if not 0 < self.fraction <= 1:
            raise ValueError('MuSc reference fraction must be in (0, 1].')
        if not self.radii or any(r < 1 or r % 2 == 0 for r in self.radii):
            raise ValueError('MuSc aggregation radii must be positive odd integers.')
        model, _ = clip.load(args.musc_clip_model, device='cpu', jit=False)
        self.visual = model.visual.float().to(self.device).eval()
        self.visual.requires_grad_(False)
        if not self.layers or min(self.layers) < 1 or max(self.layers) > len(self.visual.transformer.resblocks):
            raise ValueError('MuSc layers must be valid one-based CLIP block indices.')
        del model
        self.evaluation_results = []

    def train(self, training_data, val_data, test_data, dataset_name):
        # No fitting, normal-image memory bank or validation-based selection.
        self.evaluation_results = []
        return self.record_evaluation_epoch(test_data)

    def _encode(self, images):
        v = self.visual
        x = v.conv1(images)
        grid = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        cls = v.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        x = torch.cat((cls, x), dim=1)
        pos = v.positional_embedding.float()
        old_grid = math.isqrt(pos.shape[0] - 1)
        if grid != (old_grid, old_grid):
            spatial = pos[1:].reshape(old_grid, old_grid, -1).permute(2, 0, 1)[None]
            spatial = F.interpolate(spatial, size=grid, mode='bicubic', align_corners=False, antialias=True)
            pos = torch.cat((pos[:1], spatial[0].permute(1, 2, 0).reshape(-1, pos.shape[-1])))
        x = v.ln_pre(x + pos.to(x.dtype)).transpose(0, 1)
        stages = []
        for index, block in enumerate(v.transformer.resblocks, start=1):
            x = block(x)
            if index in self.layers:
                stages.append(x.transpose(0, 1)[:, 1:].float().cpu())
        cls = v.ln_post(x[0]) @ v.proj
        return stages, F.normalize(cls.float(), dim=-1).cpu(), grid

    @staticmethod
    def _aggregate(tokens, grid, radius):
        x = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[-1], *grid)
        x = F.layer_norm(x, x.shape[1:])
        if radius != 1:
            x = F.avg_pool2d(x, radius, stride=1, padding=radius // 2, count_include_pad=True)
        return F.normalize(x.flatten(2).transpose(1, 2), dim=-1)

    def _mutual_score(self, features):
        # Reference-image chunks bound distance memory without changing
        # the full-category pool or the nearest-patch / lowest-30% reductions.
        n, patches, _ = features.shape
        k = int((n - 1) * self.fraction)
        if k < 1:
            raise ValueError('MuSc needs enough test images for its reference fraction (at least 5 at 0.3).')
        result = torch.empty(n, patches, dtype=torch.float64)
        features = features.to(self.device)
        for query in range(n):
            nearest = []
            refs = [ref for ref in range(n) if ref != query]
            for start in range(0, len(refs), 16):
                distances = torch.cdist(features[query][None], features[refs[start:start + 16]])
                nearest.append(distances.min(dim=-1).values.transpose(0, 1))
            values = torch.cat(nearest, dim=-1)
            result[query] = values.topk(k, largest=False, sorted=True).values.mean(-1).cpu()
        return result

    def _refine_scores(self, scores, cls):
        if 0 in self.neighbors:
            return scores
        if not self.neighbors or min(self.neighbors) < 1 or max(self.neighbors) > len(scores):
            raise ValueError('MuSc RsCIN neighbor counts must fit the category test pool.')
        span = scores.max() - scores.min()
        scores = ((scores - scores.min()) / span if span > 0 else torch.zeros_like(scores)).float()
        similarity = cls @ cls.T
        refined = []
        for k in self.neighbors:
            # Match official removal of the N-k smallest similarities, including self.
            remove = similarity.topk(len(scores) - k, largest=False, sorted=True).indices
            weights = similarity.clone().scatter(1, remove, 0)
            denominator = weights.sum(-1, keepdim=True)
            if torch.any(denominator.abs() < 1e-12):
                raise ValueError('MuSc RsCIN has a zero similarity row sum.')
            refined.append((weights / denominator) @ scores)
        return torch.stack(refined).mean(0)

    @torch.inference_mode()
    def predict(self, data_loader):
        if data_loader.drop_last or not isinstance(data_loader.sampler, SequentialSampler):
            raise ValueError('MuSc requires the complete, sequential category test loader.')
        categories = {row[0] for row in data_loader.dataset.data_to_iterate}
        if len(categories) != 1:
            raise ValueError('MuSc requires one category per unlabeled test pool.')
        collected, class_tokens, labels, masks = None, [], [], []
        grid = None
        for batch in data_loader:
            images = batch['image'].to(self.device, dtype=torch.float32)
            # Input normalization/resizing belongs to the common dataset.
            with torch.autocast(device_type=self.device.type, enabled=self.device.type == 'cuda'):
                stages, cls, current_grid = self._encode(images)
            if grid is not None and grid != current_grid:
                raise ValueError('MuSc requires a consistent patch grid.')
            grid = current_grid
            if collected is None:
                collected = [[] for _ in stages]
            for bucket, tokens in zip(collected, stages):
                bucket.append(tokens)
            class_tokens.append(cls)
            # Ground truth is returned to common evaluation only, never scored.
            labels.extend(batch['is_anomaly'].cpu().tolist())
            masks.extend(batch['mask'].cpu().numpy())
        if not class_tokens:
            raise ValueError('MuSc received an empty test pool.')
        stages = [torch.cat(bucket) for bucket in collected]
        maps = torch.zeros(len(labels), grid[0] * grid[1], dtype=torch.float64)
        for radius in self.radii:
            for layer, tokens in zip(self.layers, stages):
                logger.info('MuSc: LNAMD r=%s, MSM layer=%s, pool=%s', radius, layer, len(labels))
                features = self._aggregate(tokens, grid, radius)
                maps += self._mutual_score(features) / (len(self.radii) * len(stages))
        maps = F.interpolate(maps.reshape(-1, 1, *grid), size=self.input_shape[-2:], mode='bilinear', align_corners=True)[:, 0]
        scores = self._refine_scores(maps.flatten(1).max(-1).values, torch.cat(class_tokens))
        return scores.numpy(), maps.numpy(), None, labels, masks
