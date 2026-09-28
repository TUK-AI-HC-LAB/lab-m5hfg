"""공식 AnomalyCLIP을 공통 framework Trainer contract에 연결하는 adapter입니다.

공식 source의 CLIP과 학습된 prompt checkpoint를 재사용하고, 공통 DataLoader의
image, mask, is_anomaly를 AnomalyCLIP 입력과 공통 평가 결과로 변환합니다.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

try:
    # 공통 framework 환경에서는 기존 loguru logger를 그대로 사용합니다.
    from loguru import logger
except ImportError:  # 공식 AnomalyCLIP WSL 환경처럼 loguru가 없는 경우를 허용합니다.
    import logging

    logger = logging.getLogger(__name__)


class Trainer_AnomalyCLIP:
    """공식 AnomalyCLIP prompt checkpoint를 공통 runner에서 평가하는 Trainer입니다.

    일반 Trainer와 달리 backbone은 공식 ViT-L/14 CLIP을 source repository에서
    직접 생성합니다. 하지만 load, train, predict, metric API는 동일하게 제공합니다.
    """

    def __init__(self, device):
        # 역할: 공통 runner가 사용할 AnomalyCLIP adapter의 초기 상태를 만듭니다.
        # 매개변수: device - 모델을 실행할 GPU 또는 CPU device입니다.
        # 반환값: 없음. 초기화된 Trainer_AnomalyCLIP 객체가 생성됩니다.
        self.device = device
        self.model = None
        self.prompt_learner = None
        self.args = None
        self.metrics = {}
        self.backbone = SimpleNamespace(name="anomalyclip", seed=None)
        self.checkpoint_loaded = False

    def _import_official_modules(self):
        """config의 공식 AnomalyCLIP source root에서 필요한 module을 lazy import합니다."""

        # 역할: 공식 AnomalyCLIP package와 PromptLearner class를 필요할 때만 불러옵니다.
        # 매개변수: 없음. self.args.anomalyclip_source_root 설정을 사용합니다.
        # 반환값: (AnomalyCLIP_lib, AnomalyCLIP_PromptLearner) tuple입니다.
        source_root = Path(self.args.anomalyclip_source_root).expanduser()
        if not source_root.is_dir():
            raise FileNotFoundError(
                "AnomalyCLIP official source was not found: "
                f"{source_root}. Set anomalyclip_source_root to the cloned repository."
            )
        source_root_text = str(source_root)
        if source_root_text not in sys.path:
            sys.path.insert(0, source_root_text)
        anomalyclip_lib = importlib.import_module("AnomalyCLIP_lib")
        prompt_module = importlib.import_module("prompt_ensemble")
        return anomalyclip_lib, prompt_module.AnomalyCLIP_PromptLearner

    def load(self, backbone, layers_to_extract_from, device, input_shape, **kwargs):
        """공식 CLIP, DPAM 설정, prompt learner와 선택적 checkpoint를 준비합니다."""

        # 역할: 공통 설정을 받고 공식 AnomalyCLIP 추론 모델을 초기화합니다.
        # 매개변수: backbone, layers_to_extract_from, device, input_shape, **kwargs.
        # 반환값: 없음. 이후 train과 predict가 가능한 객체 상태를 만듭니다.
        self.device = device
        self.args = kwargs["args"]
        self.backbone = backbone
        self.input_shape = input_shape
        self.layers_to_extract_from = layers_to_extract_from
        self._anomalyclip_lib, prompt_class = self._import_official_modules()
        design_details = {
            "Prompt_length": self.args.anomalyclip_prompt_length,
            "learnabel_text_embedding_depth": self.args.anomalyclip_prompt_depth,
            "learnabel_text_embedding_length": self.args.anomalyclip_text_context_length,
        }
        self.model, _ = self._anomalyclip_lib.load(
            self.args.anomalyclip_model_name,
            device=self.device,
            design_details=design_details,
        )
        self.model.eval()
        self.model.visual.DAPM_replace(DPAM_layer=self.args.anomalyclip_dapm_layer)
        # 공식 구현과 동일하게 CPU에서 prompt learner를 만든 뒤 실행 device로 옮깁니다.
        self.prompt_learner = prompt_class(self.model.to("cpu"), design_details)
        self.prompt_learner.to(self.device)
        self.model.to(self.device).eval()

        checkpoint_path = str(self.args.anomalyclip_checkpoint_path).strip()
        if checkpoint_path:
            checkpoint_file = Path(checkpoint_path).expanduser()
            if not checkpoint_file.is_file():
                raise FileNotFoundError(
                    f"AnomalyCLIP prompt checkpoint was not found: {checkpoint_file}"
                )
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            self.prompt_learner.load_state_dict(checkpoint["prompt_learner"])
            self.checkpoint_loaded = True
            logger.info(f"Loaded AnomalyCLIP prompt checkpoint: {checkpoint_file}")

    def set_model_dir(self, model_dir, dataset_name):
        """공통 runner가 요청한 결과 저장 경로를 보관합니다."""

        # 역할: runner contract를 위해 모델 저장 경로와 dataset 이름을 기록합니다.
        # 매개변수: model_dir - 상위 저장 경로, dataset_name - 현재 category 이름.
        # 반환값: 없음. 객체의 저장 경로 상태가 갱신됩니다.
        self.model_dir = model_dir or ""
        self.dataset_name = dataset_name
        if self.model_dir:
            os.makedirs(self.model_dir, exist_ok=True)

    def _text_features(self):
        """학습된 prompt token을 normal/abnormal text feature로 바꿉니다."""

        # 역할: prompt learner 출력을 공식 CLIP text encoder로 통과시킵니다.
        # 매개변수: 없음.
        # 반환값: [2, D] normal/abnormal 정규화 text feature입니다.
        prompts, tokenized_prompts, compound_prompts_text = self.prompt_learner(
            cls_id=None
        )
        features = self.model.encode_text_learn(
            prompts, tokenized_prompts, compound_prompts_text
        ).float()
        features = torch.stack(torch.chunk(features, dim=0, chunks=2), dim=1)
        features = features / features.norm(dim=-1, keepdim=True)
        return features[0]

    def train(self, training_data, validation_data, test_data, dataset_name):
        """checkpoint 기반 AnomalyCLIP 평가를 실행하고 metric을 보관합니다.

        공식 prompt 학습은 label과 pixel mask가 있는 source dataset을 요구합니다.
        공통 MVTec training split은 normal image만 제공하므로, 이 adapter의 기본 경로는
        이미 학습된 source prompt checkpoint를 target data에 적용하는 평가입니다.
        """

        # 역할: checkpoint 준비 상태를 확인한 뒤 공통 test DataLoader를 평가합니다.
        # 매개변수: training_data, validation_data, test_data, dataset_name.
        # 반환값: 없음. self.metrics에 image와 pixel AUROC를 저장합니다.
        if not self.checkpoint_loaded:
            raise RuntimeError(
                "Set anomalyclip_checkpoint_path. Official AnomalyCLIP prompt "
                "training needs labelled normal and anomalous source images, "
                "while the common MVTec-style training split contains normal images only."
            )
        scores, maps, _, labels, masks = self.predict(test_data)
        self.metrics = self._compute_metrics(scores, maps, labels, masks)
        logger.info(f"AnomalyCLIP {dataset_name} metrics: {self.metrics}")

    @torch.no_grad()
    def predict(self, data):
        """공통 DataLoader image batch를 AnomalyCLIP score와 anomaly map으로 바꿉니다."""

        # 역할: image-text similarity와 patch-text similarity로 image/pixel 이상 점수를 계산합니다.
        # 매개변수: data - image, mask, is_anomaly key를 가진 공통 DataLoader입니다.
        # 반환값: (scores, maps, None, labels, masks) 공통 Trainer 형식 tuple입니다.
        if self.model is None or self.prompt_learner is None:
            raise RuntimeError("Call load() before predict().")
        self.model.eval()
        self.prompt_learner.eval()
        text_features = self._text_features()
        scores, maps, labels, masks = [], [], [], []
        feature_layers = list(self.args.anomalyclip_features_list)
        start_layer = self.args.anomalyclip_feature_map_start

        for batch in data:
            # WinCLIP과 같은 공통 입력 정책을 사용합니다.
            # Dataset이 만든 ImageNet-normalized tensor를 추가 변환 없이 공식 모델에 전달합니다.
            image = batch["image"].to(self.device, dtype=torch.float32)
            gt_mask = batch["mask"].to(self.device, dtype=torch.float32)
            image_features, patch_features = self.model.encode_image(
                image, feature_layers, DPAM_layer=self.args.anomalyclip_dapm_layer
            )
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_logits = image_features @ text_features.T
            image_scores = (image_logits / 0.07).softmax(dim=-1)[:, 1]

            map_per_layer = []
            for layer_index, patch_feature in enumerate(patch_features):
                if layer_index < start_layer:
                    continue
                patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                similarity, _ = self._anomalyclip_lib.compute_similarity(
                    patch_feature, text_features
                )
                similarity_map = self._anomalyclip_lib.get_similarity_map(
                    similarity[:, 1:, :], self.args.imagesize
                )
                map_per_layer.append(
                    (similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0
                )
            if not map_per_layer:
                raise RuntimeError(
                    "No patch feature map was selected. Check anomalyclip_features_list."
                )
            anomaly_map = torch.stack(map_per_layer, dim=0).sum(dim=0)
            anomaly_map = F.interpolate(
                anomaly_map.unsqueeze(1),
                size=gt_mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            anomaly_map = torch.stack([
                torch.from_numpy(
                    gaussian_filter(item.detach().cpu().numpy(), sigma=self.args.anomalyclip_sigma)
                )
                for item in anomaly_map
            ])

            scores.extend(image_scores.detach().cpu().tolist())
            maps.extend(anomaly_map.numpy())
            labels.extend(batch["is_anomaly"].detach().cpu().tolist())
            masks.extend(gt_mask.squeeze(1).detach().cpu().numpy())
        return scores, np.asarray(maps), None, labels, np.asarray(masks)

    def _compute_metrics(self, scores, maps, labels, masks):
        """예측 score/map과 정답 label/mask로 AUROC를 안전하게 계산합니다."""

        # 역할: 공통 runner가 저장할 image/pixel AUROC 값을 계산합니다.
        # 매개변수: scores, maps, labels, masks.
        # 반환값: auroc_mean, pixel_auroc_mean을 가진 metric dict입니다.
        metrics = {"auroc_mean": -1.0, "pixel_auroc_mean": -1.0}
        labels = np.asarray(labels, dtype=np.int64)
        if np.unique(labels).size == 2:
            metrics["auroc_mean"] = float(roc_auc_score(labels, scores))
        flat_masks = np.asarray(masks).reshape(-1)
        flat_maps = np.asarray(maps).reshape(-1)
        if np.unique(flat_masks).size == 2:
            metrics["pixel_auroc_mean"] = float(
                roc_auc_score(flat_masks.astype(np.int64), flat_maps)
            )
        return metrics

    def get_evaluation_metrics(self):
        """공통 runner가 요청한 metric dictionary를 반환합니다."""

        # 역할: train 단계에서 계산한 metric을 runner에 전달합니다.
        # 매개변수: 없음.
        # 반환값: 문자열 key와 숫자 값을 가진 metric dict입니다.
        return self.metrics


