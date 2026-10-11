from loguru import logger
import pandas as pd
import wandb
import tqdm
import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import numpy as np
from scipy.ndimage import gaussian_filter

from trainer.trainer import Trainer
from trainer.RD_lib.resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from trainer.RD_lib.de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50, GroupNorm
from trainer.RD_lib.utils_train import MultiProjectionLayer, Revisit_RDLoss, loss_fucntion
from trainer.RD_lib.noise import Simplex_CLASS

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class Trainer_RD(Trainer):
    def initialize_model(self, **kwargs):
        self.encoder, self.bn = wide_resnet50_2(pretrained=True)
        self.encoder = self.encoder.to(self.device)
        self.bn = self.bn.to(self.device)
        self.encoder.eval()

        #decoder = de_wide_resnet50_2(pretrained=False)
        if self.args.rd_decoder_norm == 'groupnorm':
            norm_layer = GroupNorm # group 수를 32로 고정하고 사용하도록 customizing
        else:
            norm_layer = nn.BatchNorm2d

        self.decoder = de_wide_resnet50_2(pretrained=False, norm_layer=norm_layer)
        self.decoder = self.decoder.to(self.device)
        print(self.decoder)
        
        self.proj_layer =  MultiProjectionLayer(base=64).to(self.device)
        self.proj_loss = Revisit_RDLoss()
        self.optimizer_proj = torch.optim.Adam(list(self.proj_layer.parameters()), lr=self.args.proj_lr, betas=(0.5,0.999))
        self.optimizer_distill = torch.optim.Adam(list(self.decoder.parameters())+list(self.bn.parameters()), lr=self.args.distill_lr, betas=(0.5,0.999))

    def set_ea_modules(self):
        pass

    def train(self, training_data, val_data, test_data, dataset_name):
        # Initialize results list
        self.evaluation_results = []
        if self.args.checkpoint:
            prefix = "mvtec_"
            print(training_data.name)
            ckpt = self.args.checkpoint_path + '/' + training_data.name[len(prefix):] + '/' + 'wres50_' + training_data.name[len(prefix):] +'.pth'

            ckp = torch.load(ckpt, map_location='cpu')
            self.proj_layer.load_state_dict(ckp['proj'])
            self.bn.load_state_dict(ckp['bn'])
            self.decoder.load_state_dict(ckp['decoder'])
            return
        
        X = next(iter(training_data))['image']
        m = torch.asarray(IMAGENET_MEAN).view(1, 3, 1, 1)
        s = torch.asarray(IMAGENET_STD).view(1, 3, 1, 1)
        X = X * s + m

        train_data = MVTecDataset_train(X)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=self.args.bs, shuffle=True)

        if self.args.epoch_test_mode:
            df = pd.DataFrame(columns=['IAUC', 'PAUC', 'SF1'])

        for i_mepoch in range(self.meta_epochs):
            logger.info(f"\n\n----- {i_mepoch} -----")

            self.bn.train()
            self.proj_layer.train()
            self.decoder.train()

            for data_item in train_dataloader:
                img = data_item["image"].to(torch.float).to(self.device)
                img_noise = data_item["noise_image"].to(torch.float).to(self.device)

                inputs = self.encoder(img)
                inputs_noise = self.encoder(img_noise)

                (feature_space_noise, feature_space) = self.proj_layer(inputs, features_noise = inputs_noise)

                L_proj = self.proj_loss(inputs_noise, feature_space_noise, feature_space)

                outputs = self.decoder(self.bn(feature_space))#bn(inputs))
                L_distill = loss_fucntion(inputs, outputs)
                loss = L_distill + self.args.weight_proj * L_proj
                loss.backward()
                
                self.optimizer_proj.step()
                self.optimizer_distill.step()
                # Clear gradients
                self.optimizer_proj.zero_grad()
                self.optimizer_distill.zero_grad()

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
                df.to_csv('lg_results/epoch_test_rdpp_coreset.csv', index=False)
        
        # Average results at the end
        if self.evaluation_results:
            avg_metrics = {key: np.mean([m[key] for m in self.evaluation_results]) for key in self.evaluation_results[0]}
        else:
            avg_metrics = self.record_evaluation_epoch(test_data)  # Fallback
        
        return avg_metrics["auroc"], avg_metrics["full_pixel_auroc"], avg_metrics["pro"], None, None, None, avg_metrics["saliency_cr_f1"]

    def predict(self, datas):
        scores_img = []
        score_maps = []

        gt_list = [] # gt 0, 1 image-det
        gt_mask_list = [] # gt [[0, 0, 0, ...], [0, 0, ...]] pixelmaps

        self.encoder.eval()
        self.proj_layer.eval()
        self.bn.eval()
        self.decoder.eval()
        with tqdm.tqdm(datas, desc="Inferring...", leave=False) as data_iterator:
            with torch.no_grad():
                for data_item in data_iterator:
                    gt_list += [data_item['is_anomaly']]
                    gt_mask_list += [data_item['mask']]
                    name = data_item['image_name']
                    img = data_item['image'].to(self.device)
                    
                    inputs = self.encoder(img)
                    features = self.proj_layer(inputs)
                    outputs = self.decoder(self.bn(features))
                    anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
                    anomaly_map = gaussian_filter(anomaly_map, sigma=4)
                    
                    score_maps += [anomaly_map] # (H, W)
                    scores_img += [np.max(anomaly_map)]

        #logger.info(f"Average inference time: {np.sum(times) / np.sum(len_)}")
        #test_imgs, score_maps, gt_mask_list = specify_resolution(test_imgs, score_maps, gt_mask_list, resolution=(self.args.resolution, self.args.resolution))
        #result_dict = metric_cal_img(np.array(scores_img), gt_list, np.array(score_maps)), metric_cal_pix(np.array(score_maps), gt_mask_list), -1, score_maps, -1, -1
        result_dict = scores_img, score_maps, None, gt_list, gt_mask_list

        #return scores, masks, features, labels_gt, masks_gt

        return result_dict
    

class MVTecDataset_train(torch.utils.data.Dataset):
    def __init__(self, images):
        self.images = images
        self.simplexNoise = Simplex_CLASS()
        self.m = torch.asarray(IMAGENET_MEAN, dtype=torch.float).view(3, 1, 1)
        self.s = torch.asarray(IMAGENET_STD, dtype=torch.float).view(3, 1, 1)
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]

        ## simplex_noise
        size = 256
        h_noise = np.random.randint(10, int(size//8))
        w_noise = np.random.randint(10, int(size//8))
        start_h_noise = np.random.randint(1, size - h_noise)
        start_w_noise = np.random.randint(1, size - w_noise)
        noise_size = (h_noise, w_noise)
        simplex_noise = self.simplexNoise.rand_3d_octaves((3, *noise_size), 6, 0.6)
        init_zero = np.zeros((3, 256, 256))
        init_zero[:, start_h_noise: start_h_noise + h_noise, start_w_noise: start_w_noise+w_noise] = 0.2 * simplex_noise
        img_noise = img + init_zero

        img_noise = (img_noise - self.m) / self.s
        img = (img - self.m) / self.s

        return {"image": img, "noise_image": img_noise}

def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        #fs_norm = F.normalize(fs, p=2)
        #ft_norm = F.normalize(ft, p=2)
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list