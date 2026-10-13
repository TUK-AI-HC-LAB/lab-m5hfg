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
        self.device = device

    # ---- Setup ----

    def set_backbone(self, backbone, device):
        self.backbone = backbone.to(device)

    def set_aggregator(self, train_backbone, target_embed_dimension, input_shape):
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
        if self.pre_proj > 0:
            self.ea_modules = [self.pre_projection, self.discriminator]
        else:
            self.ea_modules = [self.discriminator]

    def initialize_model(self, **kwargs):
        raise NotImplementedError

    def set_model_dir(self, model_dir, dataset_name):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.model_dir, dataset_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.ckpt_dir, "tb")
        os.makedirs(self.tb_dir, exist_ok=True)

    # ---- Feature Extraction ----

    def embed(self, data):
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
        B = len(images)
        if not evaluation and self.train_backbone:
            self.forward_modules["feature_aggregator"].train()
            features = self.forward_modules["feature_aggregator"](images, eval=evaluation)
        else:
            _ = self.forward_modules["feature_aggregator"].eval()
            with torch.no_grad():
                features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]
        for i, feat in enumerate(features):
            if len(feat.shape) == 3:
                B, L, C = feat.shape
                features[i] = feat.reshape(B, int(math.sqrt(L)), int(math.sqrt(L)), C).permute(0, 3, 1, 2)

        if original_feature_list:
            return features

        features = [self.patch_maker.patchify(x, return_spatial_info=True) for x in features]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

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

        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)
        return features, patch_shapes

    def _channel_pooling(self, features):
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
        self.evaluation_results = []
        self._meta_train(training_data, val_data, test_data, dataset_name)
        if not self.evaluation_results:
            logger.warning("No evaluations recorded. Running one final evaluation.")
            self.record_evaluation_epoch(test_data)

    def _meta_train(self, training_data, val_data, test_data, dataset_name):
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
        pass

    def _additional_process(self, training_data, val_data, test_data, dataset_name, i_mepoch):
        pass

    # ---- Prediction ----

    def predict(self, data, prefix=""):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data, prefix)
        return self._predict(data)

    def _predict(self, images, mask_gt):
        images = images.to(torch.float).to(self.device)
        self.forward_modules.eval()
        batchsize = images.shape[0]
        with torch.no_grad():
            st = time()
            features, patch_shapes = self._embed(images)
            features = self._preprocessing_features_predict(features)
            patch_scores = image_scores = self._get_pred_scores(features)
            self.inftime += time() - st

            if isinstance(patch_scores, torch.Tensor):
                patch_scores = patch_scores.cpu().numpy()
            if isinstance(image_scores, torch.Tensor):
                image_scores = image_scores.cpu().numpy()

            image_scores = self.patch_maker.unpatch_scores(image_scores, batchsize=batchsize)
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self._preprocessing_predict(images, image_scores)
            image_scores = self._score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(patch_scores, batchsize=batchsize)
            masks = self._get_segmentations(patch_scores, batchsize, patch_shapes)
            self.inftime_pixel += time() - st
            return list(image_scores), list(masks), list(features), list(patch_scores)

    def _predict_dataloader(self, dataloader, prefix):
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
        pass

    def _preprocessing_predict(self, _image, image_scores):
        return image_scores

    def _score(self, scores):
        return self.patch_maker.score(scores)

    def _get_pred_scores(self, features):
        raise NotImplementedError

    def _preprocessing_features_predict(self, features):
        raise NotImplementedError

    def _get_segmentations(self, patch_scores, batchsize, patch_shapes):
        scales = patch_shapes[0]
        patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
        return self.anomaly_segmentor.convert_to_segmentation(patch_scores)

    # ---- Evaluation ----

    def saliency_f1_score(self, pred, mask):
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
        metrics_map = {
            "auroc": "auroc",
            "full_pixel_auroc": "pixel_auroc",
            "pro": "pro",
            "saliency_cr_f1": "sal_f1",
        }
        optional_metrics = (
            "image_auroc",
            "image_ap",
            "image_f1",
            "pixel_ap",
            "pixel_f1",
            "pixel_aupro",
        )

        for key in optional_metrics:
            if any(key in record for record in self.evaluation_results):
                metrics_map[key] = key
                
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
        return os.path.join(filepath, prepend + "params.pkl")

    def save_to_path(self, save_path, prepend=""):
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
        self.patchsize = patchsize
        self.stride = stride
        self.top_k = top_k

    def patchify(self, features, return_spatial_info=False):
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
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
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
