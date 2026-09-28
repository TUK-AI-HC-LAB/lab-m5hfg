import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import math
import pandas as pd
from loguru import logger
from time import time

import common
import metrics_gpu
from trainer.classes import EarlyStopping, ElapsedTimer
from trainer.mixins import DiscriminatorMixin, VisualizationMixin
import psutil
from simplenet import Discriminator, Projection


class Trainer(DiscriminatorMixin, VisualizationMixin):
    def __init__(self, device):
        # 역할: 객체를 사용할 수 있도록 필요한 속성과 구성요소를 초기화.
        # 매개변수: device.
        # 반환값: 없음. 초기화가 끝난 객체를 Python이 생성합니다..
        self.device = device

    # ---- Setup ----

    def set_backbone(self, backbone, device):
        # 역할: 객체의 설정 또는 내부 상태를 지정.
        # 매개변수: backbone, device.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        self.backbone = backbone.to(device)

    def set_aggregator(self, train_backbone, target_embed_dimension, input_shape):
        # 역할: 객체의 설정 또는 내부 상태를 지정.
        # 매개변수: train_backbone, target_embed_dimension, input_shape.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device, train_backbone)
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = common.Preprocessing(feature_dimensions, target_embed_dimension)
        self.forward_modules["preprocessing"] = preprocessing
        self.target_embed_dimension = target_embed_dimension

        preadapt_aggregator = common.Aggregator(target_dim=target_embed_dimension)
        _ = preadapt_aggregator.to(self.device)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

    def load(self, backbone, layers_to_extract_from, device, input_shape,
             pretrain_embed_dimension, target_embed_dimension,
             patchsize=3, patchstride=1, embedding_size=None, meta_epochs=1,
             aed_meta_epochs=1, gan_epochs=1, noise_std=0.05, mix_noise=1,
             noise_type="GAU", dsc_layers=2, dsc_hidden=None, dsc_margin=.8,
             dsc_lr=0.0002, train_backbone=False, auto_noise=0, cos_lr=False,
             lr=1e-3, pre_proj=0, proj_layer_type=0, onnx=None, **kwargs):
        # 역할: 필요한 모델·가중치·설정을 준비.
        # 매개변수: backbone, layers_to_extract_from, device, input_shape, pretrain_embed_dimension, target_embed_dimension, patchsize, patchstride, embedding_size, meta_epochs, aed_meta_epochs, gan_epochs, noise_std, mix_noise, noise_type, dsc_layers, dsc_hidden, dsc_margin, dsc_lr, train_backbone, auto_noise, cos_lr, lr, pre_proj, proj_layer_type, onnx, **kwargs.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        self.onnx = onnx
        self.elapsed_timer = ElapsedTimer()
        self.args = kwargs['args']

        self.set_backbone(backbone, device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape
        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)
        self.forward_modules = torch.nn.ModuleDict({})
        self.target_embed_dimension = target_embed_dimension

        self.set_aggregator(train_backbone, target_embed_dimension, input_shape)
        self.anomaly_segmentor = common.RescaleSegmentor(
            device=self.device, target_size=(self.args.masksize, self.args.masksize),
            smoothing=self.args.last_score_blur_sigma)

        self.meta_epochs = meta_epochs
        self.lr = lr
        self.cos_lr = cos_lr
        self.train_backbone = train_backbone
        if self.train_backbone:
            self.backbone_opt = torch.optim.AdamW(self.backbone.parameters(), lr)

        self.auto_noise = [auto_noise, None]
        self.dsc_lr = dsc_lr
        self.gan_epochs = gan_epochs
        self.mix_noise = mix_noise
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.aed_meta_epochs = aed_meta_epochs
        self.pre_proj = pre_proj
        self.model_dir = ""
        self.dataset_name = ""
        self.tau = 1

        self.initialize_model(
            dsc_layers=dsc_layers, dsc_hidden=dsc_hidden, pre_proj=pre_proj,
            proj_layer_type=proj_layer_type, meta_epochs=meta_epochs,
            aed_meta_epochs=aed_meta_epochs, gan_epochs=gan_epochs,
            dsc_margin=dsc_margin, dsc_lr=dsc_lr, lr=lr,
            device=self.device, args=self.args)
        self.set_ea_modules()

    def set_ea_modules(self):
        # 역할: 객체의 설정 또는 내부 상태를 지정.
        # 매개변수: 없음.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        if self.pre_proj > 0:
            self.ea_modules = [self.pre_projection, self.discriminator]
        else:
            self.ea_modules = [self.discriminator]

    def initialize_model(self, **kwargs):
        # 역할: `initialize_model`에 해당하는 작업을 수행.
        # 매개변수: **kwargs.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        raise NotImplementedError

    def set_model_dir(self, model_dir, dataset_name):
        # 역할: 객체의 설정 또는 내부 상태를 지정.
        # 매개변수: model_dir, dataset_name.
        # 반환값: 보통 없음(None)이며, 객체 상태가 변경됩니다..
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.model_dir, dataset_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.ckpt_dir, "tb")
        os.makedirs(self.tb_dir, exist_ok=True)

    # ---- Feature Extraction ----

    def embed(self, data):
        # 역할: `embed`에 해당하는 작업을 수행.
        # 매개변수: data.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                    input_image = image.to(torch.float).to(self.device)
                with torch.no_grad():
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images, evaluation=False, original_feature_list=False):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: images, evaluation, original_feature_list.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        B = len(images)
        if not evaluation and self.train_backbone:
            self.forward_modules["feature_aggregator"].train()
            features = self.forward_modules["feature_aggregator"](images, eval=evaluation)
        else:
            _ = self.forward_modules["feature_aggregator"].eval()
            with torch.no_grad():
                features = self.forward_modules["feature_aggregator"](images)

        # forward hook이 저장한 여러 backbone layer 중 설정에서 선택한 layer만 사용합니다.
        features = [features[layer] for layer in self.layers_to_extract_from]
        for i, feat in enumerate(features):
            if len(feat.shape) == 3:
                B, L, C = feat.shape
                features[i] = feat.reshape(B, int(math.sqrt(L)), int(math.sqrt(L)), C).permute(0, 3, 1, 2)

        if original_feature_list:
            return features

        # 큰 feature map을 위치별 작은 patch 묶음으로 나눕니다.
        # patch_shapes는 나중에 patch score를 원래 2차원 위치로 되돌릴 때 필요합니다.
        features = [self.patch_maker.patchify(x, return_spatial_info=True) for x in features]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        # 서로 다른 layer는 feature map 해상도가 다를 수 있습니다.
        # 첫 layer의 patch 격자를 기준으로 맞춰야 같은 위치끼리 합칠 수 있습니다.
        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]
            _features = _features.reshape(_features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:])
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear", align_corners=False).squeeze(1)
            _features = _features.reshape(*perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1])
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features

        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        # layer마다 다른 channel/patch 크기를 공통 embedding 차원으로 압축합니다.
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)
        return features, patch_shapes

    def _channel_pooling(self, features):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: features.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        C = features[0].shape[1]
        H = features[0].shape[2]
        W = features[0].shape[3]
        for i, feat in enumerate(features):
            feat = feat.permute(0, 2, 3, 1).reshape(feat.shape[0], -1, feat.shape[1])
            feat = torch.nn.functional.adaptive_avg_pool1d(feat, C)
            feat = feat.reshape(feat.shape[0], H, W, C)
            features[i] = feat.permute(0, 3, 1, 2)
        return torch.stack(features, dim=1).mean(1)

    # ---- Training ----

    def train(self, training_data, val_data, test_data, dataset_name):
        # 역할: 학습 데이터를 사용해 모델 파라미터를 갱신.
        # 매개변수: training_data, val_data, test_data, dataset_name.
        # 반환값: 보통 없음(None)이며, 객체 상태가 변경됩니다..
        self.evaluation_results = []
        self._meta_train(training_data, val_data, test_data, dataset_name)
        if not self.evaluation_results:
            logger.warning("No evaluations recorded. Running one final evaluation.")
            self.record_evaluation_epoch(test_data)

    def _meta_train(self, training_data, val_data, test_data, dataset_name):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: training_data, val_data, test_data, dataset_name.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        self.dataset_name = dataset_name
        state_dict = {}
        ckpt_path = None
        best_record = None
        self._pre_meta_train(training_data, val_data, test_data, dataset_name)

        if self.args.epoch_test_mode:
            self.df = pd.DataFrame(columns=['IAUC', 'PAUC', 'SF1'])
            self.test_data = test_data

        with tqdm.tqdm(range(self.meta_epochs)) as pbar:
            for i_mepoch in pbar:
                self.i_mepoch = i_mepoch
                loss = self._train_discriminator(training_data)
                self._additional_process(training_data, val_data, test_data, dataset_name, i_mepoch)
                pbar.set_description_str(f"epoch:{i_mepoch} loss:{loss}")
                if self.args.subtest == 1:
                    break

        if ckpt_path is not None:
            torch.save(state_dict, ckpt_path)

        self.record_evaluation_epoch(test_data)
        if self.evaluation_results:
            scores, segmentations, features, labels_gt, masks_gt = self.predict(test_data)
            self._save_fault_images(test_data, scores)

    def _pre_meta_train(self, training_data, val_data, test_data, dataset_name):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: training_data, val_data, test_data, dataset_name.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        pass

    def _additional_process(self, training_data, val_data, test_data, dataset_name, i_mepoch):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: training_data, val_data, test_data, dataset_name, i_mepoch.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        pass

    # ---- Prediction ----

    def predict(self, data, prefix=""):
        # 역할: 입력의 예측 또는 이상 탐지 점수와 map을 계산.
        # 매개변수: data, prefix.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data, prefix)
        return self._predict(data)

    def _predict(self, images, mask_gt):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: images, mask_gt.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # DataLoader의 CPU tensor를 모델이 실행되는 GPU/CPU device로 옮깁니다.
        images = images.to(torch.float).to(self.device)
        self.forward_modules.eval()
        batchsize = images.shape[0]
        with torch.no_grad():
            st = time()
            features, patch_shapes = self._embed(images)
            features = self._preprocessing_features_predict(features)
            # 방법별 차이는 여기입니다. SimpleNet은 discriminator logit,
            # PatchCore는 memory bank와의 최근접 거리처럼 각 patch의 이상 정도를 만듭니다.
            patch_scores = image_scores = self._get_pred_scores(features)
            self.inftime += time() - st

            if isinstance(patch_scores, torch.Tensor):
                patch_scores = patch_scores.cpu().numpy()
            if isinstance(image_scores, torch.Tensor):
                image_scores = image_scores.cpu().numpy()

            # [B*P,...] patch score를 이미지별 [B,P,...]로 다시 묶은 뒤,
            # 가장 의심스러운 patch를 image-level score로 선택합니다.
            image_scores = self.patch_maker.unpatch_scores(image_scores, batchsize=batchsize)
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self._preprocessing_predict(images, image_scores)
            image_scores = self._score(image_scores)

            # 같은 patch score를 2차원 격자로 복원하고, mask 크기까지 확대하여 anomaly map을 만듭니다.
            patch_scores = self.patch_maker.unpatch_scores(patch_scores, batchsize=batchsize)
            masks = self._get_segmentations(patch_scores, batchsize, patch_shapes)
            self.inftime_pixel += time() - st
            return list(image_scores), list(masks), list(features), list(patch_scores)

    def _predict_dataloader(self, dataloader, prefix):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: dataloader, prefix.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        _ = self.forward_modules.eval()
        scores = []
        masks = []
        features = []
        labels_gt = []
        masks_gt = []

        self.inftime = 0
        self.inftime_pixel = 0
        num_samples = 0
        with tqdm.tqdm(enumerate(dataloader), desc="Inferring...", leave=False) as data_iterator:
            for i, data in data_iterator:
                if isinstance(data, dict):
                    image = data["image"]
                    masks_gt.extend(data["mask"].numpy().tolist())
                    labels_gt.extend(data["is_anomaly"].numpy().tolist())
                num_samples += image.shape[0]

                self._preprocessing_predict_dataloader(data)
                _scores, _masks, _feats, _patch_scores = self._predict(image, data["mask"].numpy().tolist())
                for score, mask, feat, ps in zip(_scores, _masks, _feats, _patch_scores):
                    scores.append(score)
                    masks.append(mask)

                if self.args.subtest and i > 2 and torch.tensor(labels_gt).unique().shape[0] >= 2:
                    if len(masks_gt) > 0:
                        if np.array(masks_gt).sum() > 0:
                            break
                        else:
                            continue
                    else:
                        break

        logger.info("Average inference time: {:.4f}".format(self.inftime / num_samples))
        logger.info("Average pixel inference time: {:.4f}".format(self.inftime_pixel / num_samples))
        return scores, masks, features, labels_gt, masks_gt

    def _preprocessing_predict_dataloader(self, data):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: data.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        pass

    def _preprocessing_predict(self, _image, image_scores):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: _image, image_scores.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        return image_scores

    def _score(self, scores):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: scores.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        return self.patch_maker.score(scores)

    def _get_pred_scores(self, features):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: features.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        raise NotImplementedError

    def _preprocessing_features_predict(self, features):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: features.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        raise NotImplementedError

    def _get_segmentations(self, patch_scores, batchsize, patch_shapes):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: patch_scores, batchsize, patch_shapes.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        scales = patch_shapes[0]
        patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
        return self.anomaly_segmentor.convert_to_segmentation(patch_scores)

    # ---- Evaluation ----

    def saliency_f1_score(self, pred, mask):
        # 역할: `saliency_f1_score`에 해당하는 작업을 수행.
        # 매개변수: pred, mask.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        if len(pred.shape) != 3:
            pred = pred.squeeze()
        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(pred)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        pred_mask = (pred >= 0.5).float()
        pred_mask = pred_mask.to(torch.float32)
        mask = mask.to(torch.float32)
        TP = pred_mask * mask
        FP = pred_mask * (1 - mask)
        FN = (1 - pred_mask) * mask
        precision = (pred * TP).sum() / ((pred * (TP + FP)).sum() + 1e-8)
        recall = (pred * TP).sum() / ((pred * (TP + FN)).sum() + 1e-8)
        return (2 * precision * recall / (precision + recall + 1e-8)).item()

    def _evaluate(self, scores, segmentations, features, labels_gt, masks_gt, norm=True):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: scores, segmentations, features, labels_gt, masks_gt, norm.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        if norm:
            scores = np.squeeze(np.array(scores))
            img_min_scores = scores.min(axis=-1)
            img_max_scores = scores.max(axis=-1)
            scores = (scores - img_min_scores) / (img_max_scores - img_min_scores)

        logger.info("compute auroc")
        auroc = metrics_gpu.compute_imagewise_retrieval_metrics(scores, labels_gt)["auroc"]
        logger.info("done")

        if self.args.pixel_auroc or self.args.save_segmentation_images:
            logger.info("segmentation preprocessing")
            norm_segmentations = metrics_gpu.normalize_segmentations(np.array(segmentations), len(scores))
            logger.info("done")
            masks_gt = np.ceil(np.array(masks_gt)).squeeze()
            logger.info("compute saliency cr f1")
            saliency_cr_f1 = self.saliency_f1_score(norm_segmentations, masks_gt)
            logger.info("compute full pixel auroc")
            pixel_scores = metrics_gpu.compute_pixelwise_retrieval_metrics(norm_segmentations, masks_gt)
            full_pixel_auroc = pixel_scores["auroc"]
            pro = -1
        else:
            full_pixel_auroc = -1
            pro = -1
            saliency_cr_f1 = -1

        return auroc, full_pixel_auroc, pro, saliency_cr_f1

    def record_evaluation_epoch(self, data_loader):
        # 역할: `record_evaluation_epoch`에 해당하는 작업을 수행.
        # 매개변수: data_loader.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        scores, segmentations, features, labels_gt, masks_gt = self.predict(data_loader)
        auroc, full_pixel_auroc, pro, saliency_cr_f1 = self._evaluate(
            scores, segmentations, features, labels_gt, masks_gt)
        epoch_metrics = {
            "auroc": auroc, "full_pixel_auroc": full_pixel_auroc,
            "pro": pro, "saliency_cr_f1": saliency_cr_f1,
        }
        self.evaluation_results.append(epoch_metrics)
        logger.info(f"Recorded Evaluation: AUROC {auroc:.4f}, Pixel-AUROC {full_pixel_auroc:.4f}")
        return epoch_metrics

    def get_evaluation_metrics(self):
        # 역할: 요청한 내부 정보 또는 계산 결과를 가져옴.
        # 매개변수: 없음.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        metrics_map = {"auroc": "auroc", "full_pixel_auroc": "pixel_auroc", "pro": "pro", "saliency_cr_f1": "sal_f1"}
        results = {}
        if not self.evaluation_results:
            for key in metrics_map.values():
                for stat in ["min", "max", "mean", "std"]:
                    results[f"{key}_{stat}"] = -1
            return results

        for original_key, new_key in metrics_map.items():
            values = [d.get(original_key) for d in self.evaluation_results
                      if d.get(original_key) is not None and d.get(original_key) != -1]
            if not values:
                for stat in ["min", "max", "mean", "std"]:
                    results[f"{new_key}_{stat}"] = -1
            else:
                results[f"{new_key}_min"] = np.min(values)
                results[f"{new_key}_max"] = np.max(values)
                results[f"{new_key}_mean"] = np.mean(values)
                results[f"{new_key}_std"] = np.std(values)
        return results

    # ---- Save/Load ----

    @staticmethod
    def _params_file(filepath, prepend=""):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: filepath, prepend.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        return os.path.join(filepath, prepend + "params.pkl")

    def save_to_path(self, save_path, prepend=""):
        # 역할: `save_to_path`에 해당하는 작업을 수행.
        # 매개변수: save_path, prepend.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        self.anomaly_scorer.save(save_path, save_features_separately=False, prepend=prepend)
        params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules["preprocessing"].output_dim,
            "target_embed_dimension": self.forward_modules["preadapt_aggregator"].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(params, save_file, pickle.HIGHEST_PROTOCOL)


class PatchMaker:
    def __init__(self, patchsize, top_k=0, stride=None):
        # 역할: 객체를 사용할 수 있도록 필요한 속성과 구성요소를 초기화.
        # 매개변수: patchsize, top_k, stride.
        # 반환값: 없음. 초기화가 끝난 객체를 Python이 생성합니다..
        self.patchsize = patchsize
        self.stride = stride
        self.top_k = top_k

    def patchify(self, features, return_spatial_info=False):
        # 역할: `patchify`에 해당하는 작업을 수행.
        # 매개변수: features, return_spatial_info.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1)
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (s + 2 * padding - 1 * (self.patchsize - 1) - 1) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(*features.shape[:2], self.patchsize, self.patchsize, -1)
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)
        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        # 역할: `unpatch_scores`에 해당하는 작업을 수행.
        # 매개변수: x, batchsize.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        # 역할: `score`에 해당하는 작업을 수행.
        # 매개변수: x.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 2:
            x = torch.max(x, dim=-1).values
        if x.ndim == 2:
            if self.top_k > 1:
                x = torch.topk(x, self.top_k, dim=1).values.mean(1)
            else:
                x = torch.max(x, dim=1).values
        if was_numpy:
            return x.numpy()
        return x
# 한국어 코드 안내: 이 파일은 이상 탐지 방법의 학습·예측 제어을 담당합니다.
# 원본 실행 로직은 변경하지 않고, 초보자용 설명 주석만 추가했습니다.
