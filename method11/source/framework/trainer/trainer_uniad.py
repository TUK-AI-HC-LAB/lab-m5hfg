from loguru import logger
import pandas as pd
import wandb
import tqdm
import importlib

import torch
import torch.nn as nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import numpy as np

from trainer.trainer import Trainer
from .UniAD_lib.models.model_helper import ModelHelper
from .UniAD_lib.utils.criterion_helper import build_criterion

class Trainer_UniAD(Trainer):
    def initialize_model(self, **kwargs):
        cfg = [
            {
                "name": "backbone",
                "type": "UniAD_lib.models.backbones.efficientnet_b4",
                "frozen": True,
                "kwargs": {
                    "pretrained": True,
                    "outlayers": [1, 2, 3, 4],
                },
            },
            {
                "name": "neck",
                "prev": "backbone",
                "type": "UniAD_lib.models.necks.MFCN",
                "kwargs": {
                    "outstrides": [16],
                },
            },
            {
                "name": "reconstruction",
                "prev": "neck",
                "type": "UniAD_lib.models.reconstructions.UniAD",
                "kwargs": {
                    "pos_embed_type": "learned",
                    "hidden_dim": 256,
                    "nhead": 8,
                    "num_encoder_layers": 4,
                    "num_decoder_layers": 4,
                    "dim_feedforward": 1024,
                    "dropout": 0.1,
                    "activation": "relu",
                    "normalize_before": False,
                    "feature_jitter": {
                        "scale": 20.0,
                        "prob": 1.0,
                    },
                    "neighbor_mask": {
                        "neighbor_size": [7, 7],
                        "mask": [True, True, True],
                    },
                    "save_recon": None,#{
                        #"save_dir": "result_recon",
                    #},
                    "initializer": {
                        "method": "xavier_uniform",
                    },
                },
            },
        ]
        cfg = update_config(cfg)

        self.model = ModelHelper(cfg)
        self.model.cuda()

        frozen_layers = ["backbone"]
        active_layers = ["neck", "reconstruction"]
        parameters = [
            {"params": getattr(self.model, layer).parameters()} for layer in active_layers
        ]

        self.optimizer = torch.optim.AdamW(parameters, lr=self.args.lr, betas=self.args.betas, weight_decay=self.args.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size, gamma=self.args.gamma)
        self.criteron = torch.nn.MSELoss()

        self.trained = False

    def set_ea_modules(self):
        pass

    def train(self, training_data, val_data, test_data, dataset_name):
        # Initialize results list
        self.evaluation_results = []
        
        if self.args.uni and self.trained:
            # Record evaluation and return averaged results
            self.record_evaluation_epoch(test_data)
            if self.evaluation_results:
                avg_metrics = {key: np.mean([m[key] for m in self.evaluation_results]) for key in self.evaluation_results[0]}
            else:
                avg_metrics = self.record_evaluation_epoch(test_data)
            return avg_metrics["auroc"], avg_metrics["full_pixel_auroc"], avg_metrics["pro"], None, None, None, avg_metrics["saliency_cr_f1"]

        if self.args.epoch_test_mode:
            df = pd.DataFrame(columns=['IAUC', 'PAUC', 'SF1'])
        
        for i_mepoch in range(self.meta_epochs):
            logger.info(f"\n\n----- {i_mepoch} -----")
            self.model.train()

            for data_item in tqdm.tqdm(training_data, desc="Training Progress " + str(i_mepoch) + "/" + str(self.meta_epochs), unit="it"):
                image = data_item['image'].to(self.device)
                output = self.model(image)
                
                loss = self.criteron(output["feature_rec"], output["feature_align"])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # 에폭마다 시행
            self.lr_scheduler.step()
            
            if self.args.epoch_test_mode:
                # For monitoring: always perform evaluation
                scores, segmentations, features, labels_gt, masks_gt = self.predict(test_data)
                auroc, full_pixel_auroc, pro, saliency_cr_f1 = self._evaluate(
                    scores, segmentations, features, labels_gt, masks_gt
                )
                epoch_metrics = {
                    "auroc": auroc,
                    "full_pixel_auroc": full_pixel_auroc,
                    "pro": pro,
                    "saliency_cr_f1": saliency_cr_f1,
                }
                
                # Only add to evaluation_results if this is one of the last 5 epochs
                if i_mepoch >= self.meta_epochs - 5:
                    self.evaluation_results.append(epoch_metrics)
                
                a, f, s = epoch_metrics["auroc"], epoch_metrics["full_pixel_auroc"], epoch_metrics["saliency_cr_f1"]
                df.loc[i_mepoch] = [a, f, s]
                df.to_csv(f'lg_results/epoch_test_uniad_{dataset_name}_{training_data.dataset.root.split("/")[-1]}.csv', index=False)

        self.trained = True

        # Average results at the end
        if self.evaluation_results:
            avg_metrics = {key: np.mean([m[key] for m in self.evaluation_results]) for key in self.evaluation_results[0]}
        else:
            avg_metrics = self.record_evaluation_epoch(test_data)  # Fallback
        
        return avg_metrics["auroc"], avg_metrics["full_pixel_auroc"], avg_metrics["pro"], None, None, None, avg_metrics["saliency_cr_f1"]
    
    @torch.no_grad()
    def predict(self, datas):
        scores_img = []
        scores_img_i1 = []
        scores_img_i2 = []
        scores_img_i3 = []
        scores_maps = []

        gt_list = [] # gt 0, 1 image-det
        gt_mask_list = [] # gt [[0, 0, 0, ...], [0, 0, ...]] pixelmaps

        self.model.eval()
        with tqdm.tqdm(datas, desc="Inferring...", leave=False) as data_iterator:
            for data_item in data_iterator:
                gt_list += [data_item['is_anomaly']]
                gt_mask_list += [data_item['mask']]
                name = data_item['image_name']
                image = data_item['image'].to(self.device)
                
                output = self.model(image)
                pred = output["pred"].cpu().numpy()
                scores_maps += [pred]
                i1 = np.max(F.avg_pool2d(output["pred"], [16, 16], stride=1).cpu().numpy())
                i2 = pred.mean()
                i3 = pred.std()
                scores_img_i1 += [i1]
                scores_img_i2 += [i2]
                scores_img_i3 += [i3]
                scores_img += [(i1 + i2 + i3) / 3]

        gt_mask_list = np.asarray(gt_mask_list, dtype=int)
        scores_maps = np.asarray(scores_maps, dtype=float)
        return scores_img, scores_maps, None, gt_list, gt_mask_list

        per_pixel_rocauc = -1
        try:
            per_pixel_rocauc = roc_auc_score(gt_mask_list.flatten(), scores_maps.flatten())
        except:
            pass

        print(f'{name[0].split("/")[0]} image_auc_max : {roc_auc_score(gt_list, scores_img_i1)}')
        print(f'{name[0].split("/")[0]} image_auc_mean : {roc_auc_score(gt_list, scores_img_i2)}')
        print(f'{name[0].split("/")[0]} image_auc_std : {roc_auc_score(gt_list, scores_img_i3)}')
        result_dict = roc_auc_score(gt_list, scores_img), per_pixel_rocauc, -1, scores_maps, -1, -1

        return result_dict
    
def update_config(config):
    # update feature size
    input_size = [224,224]
    outstride = config[1]["kwargs"]["outstrides"][0]
    feature_size = [s // outstride for s in input_size]
    config[2]["kwargs"]["feature_size"] = feature_size

    # update planes & strides
    backbone_path, backbone_type = config[0]["type"].rsplit(".", 1)

    backbone_path = 'trainer.' + backbone_path
    module = importlib.import_module(backbone_path)
    backbone_info = getattr(module, "backbone_info")
    backbone = backbone_info[backbone_type]
    outblocks = []
    outstrides = []
    outplanes = []
    for layer in config[0]["kwargs"]["outlayers"]:
        if layer not in backbone["layers"]:
            raise ValueError(
                "only layer {} for backbone {} is allowed, but get {}!".format(
                    backbone["layers"], backbone_type, layer
                )
            )
        idx = backbone["layers"].index(layer)
        if "efficientnet" in backbone_type:
            outblocks.append(backbone["blocks"][idx])
        outstrides.append(backbone["strides"][idx])
        outplanes.append(backbone["planes"][idx])
    if "efficientnet" in backbone_type:
        config[0]["kwargs"].pop("outlayers")
        config[0]["kwargs"]["outblocks"] = outblocks
    config[0]["kwargs"]["outstrides"] = outstrides
    config[1]["kwargs"]["outplanes"] = [sum(outplanes)]

    return config