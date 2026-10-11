import os
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import mahalanobis
from collections import OrderedDict
from loguru import logger
import pandas as pd
import wandb
from tqdm import tqdm
import importlib

import torch
import torch.nn as nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import numpy as np

from trainer.trainer import Trainer
from torchvision.models import wide_resnet50_2
from random import sample
from matplotlib import pyplot as plt
from .PaDiM_lib.functions import parse_args, plot_fig, denormalization, embedding_concat
import time, random, pickle
# set model's intermediate outputs
outputs = []

def hook(module, input, output):
    # 역할: `hook`에 해당하는 작업을 수행.
    # 매개변수: module, input, output.
    # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
    outputs.append(output)


class Trainer_PaDiM(Trainer):
    def initialize_model(self, **kwargs):
        # 역할: `initialize_model`에 해당하는 작업을 수행.
        # 매개변수: **kwargs.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        """
        args initialization
        Namespace(calc_mode='batch_einsum', dataset='lgdata', data_path='/home/robert.lim/main/config.json', save_path='./lg_result', arch='wide_resnet50_2')
        """
        args = self.args
        setattr(args, 'save_path', './lg_results/PaDiM_lib')
        setattr(args, 'arch', 'wide_resnet50_2')

        model = wide_resnet50_2(pretrained=True, progress=True)
        t_d = 1792
        d = 550

        model.to(self.device)
        model.eval()
        self.model = model

        self.idx = torch.tensor(sample(range(0, t_d), d))

        os.makedirs(os.path.join(args.save_path, 'temp_%s' %
                    args.arch), exist_ok=True)

        self.model.layer1[-1].register_forward_hook(hook)
        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)

    def set_ea_modules(self):
        # 역할: 객체의 설정 또는 내부 상태를 지정.
        # 매개변수: 없음.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        pass

    def _meta_train(self, training_data, val_data, test_data, dataset_name):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: training_data, val_data, test_data, dataset_name.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        args = self.args
        cov_time = 0

        train_outputs = OrderedDict(
            [('layer1', []), ('layer2', []), ('layer3', [])])

        # extract train set features
        train_feature_filepath = os.path.join(
            args.save_path, 'temp_%s' % args.arch, 'train_%s.pkl' % dataset_name)
        for data in tqdm(training_data):
            x = data['image']
            # model prediction
            with torch.no_grad():
                _ = self.model(x.to(self.device))
            # get intermediate layer outputs
            for k, v in zip(train_outputs.keys(), outputs):
                # the dimension of a vector is torch.Size([32, 256, 56, 56])
                train_outputs[k].append(v.cpu().detach())
            # initialize hook outputs
            outputs.clear()
        for k, v in train_outputs.items():
            # 예를 들어, 위에서 32개씩 모아 총 209 개를 torch.cat으로 합침
            train_outputs[k] = torch.cat(v, 0)
            """
            (Pdb) p train_outputs['layer1'].shape
            torch.Size([209, 256, 56, 56])
            (Pdb) p train_outputs['layer2'].shape
            torch.Size([209, 512, 28, 28])
            (Pdb) p train_outputs['layer3'].shape
            torch.Size([209, 1024, 14, 14])
            """

        # Embedding concat
        embedding_vectors = train_outputs['layer1']
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = embedding_concat(
                embedding_vectors, train_outputs[layer_name])

        # randomly select d dimension
        # embedding_vectors.shape torch.Size([209, 1792, 56, 56]) --> torch.Size([209, 550, 56, 56])
        embedding_vectors = torch.index_select(embedding_vectors, 1, self.idx)

        # calculate multivariate Gaussian distribution
        print('calculate mean vector and covariance matrix')
        B, C, H, W = embedding_vectors.size()
        # --> embedding_vectors.shape torch.Size([209, 550, 3136])
        embedding_vectors = embedding_vectors.view(B, C, H * W)
        # --> mean.shape (550, 3136)
        mean = torch.mean(embedding_vectors, dim=0).numpy()
        # --> cov.shape (550, 550, 3136)
        cov = torch.zeros(C, C, H * W).numpy()
        I = np.identity(C)  # --> I.shape (550, 550)
        for i in tqdm(range(H * W), desc='calculating covariance matrix of training data'):
            st = time.time()
            # cov[:, :, i] = LedoitWolf().fit( embedding_vectors[:, :, i].numpy()).covariance_
            cov[:, :, i] = np.cov(
                embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
            cov_time += time.time() - st

        # save learned distribution
        self.train_outputs = [mean, cov]
        with open(train_feature_filepath, 'wb') as f:
            pickle.dump(self.train_outputs, f)
    
    @torch.no_grad()
    def predict(self, test_dataloader):
        # 역할: 입력의 예측 또는 이상 탐지 점수와 map을 계산.
        # 매개변수: test_dataloader.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        inv_time = 0
        dist_time = 0
        args = self.args
        gt_list = []
        gt_mask_list = []
        test_imgs = []

        test_outputs = OrderedDict(
            [('layer1', []), ('layer2', []), ('layer3', [])])

        # extract test set features
        for data in tqdm(test_dataloader):
            x, y, mask = data['image'], data['is_anomaly'], data['mask']
            test_imgs.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            gt_mask_list.extend(mask.cpu().detach().numpy())
            # model prediction
            with torch.no_grad():
                _ = self.model(x.to(self.device))
            # get intermediate layer outputs
            for k, v in zip(test_outputs.keys(), outputs):
                test_outputs[k].append(v.cpu().detach())
            # initialize hook outputs
            outputs.clear()
        for k, v in test_outputs.items():
            test_outputs[k] = torch.cat(v, 0)

        # Embedding concat
        embedding_vectors = test_outputs['layer1']
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = embedding_concat(
                embedding_vectors, test_outputs[layer_name])

        # randomly select d dimension
        embedding_vectors = torch.index_select(embedding_vectors, 1, self.idx)

        # calculate distance matrix
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
        dist_list = []

        for i in tqdm(range(H * W), desc='calculating mahalanobis distance of test data'):
            mean = self.train_outputs[0][:, i]
            st = time.time()
            conv_inv = np.linalg.inv(self.train_outputs[1][:, :, i])
            inv_time += time.time() - st
            st = time.time()
            dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
            dist_time += time.time() - st
            dist_list.append(dist)

        dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

        # upsample
        dist_list = torch.tensor(dist_list)
        score_maps = F.interpolate(dist_list.unsqueeze(1), size=x.size(2), mode='bilinear',
                                  align_corners=False).squeeze().numpy()  # [83 (#test_images), 56, 56]   -->   [83, 224, 224]

        # apply gaussian smoothing on the score map
        for i in range(score_maps.shape[0]):
            # patch 간 경계를 부드럽게 하기 위해 gaussian smoothing 적용
            score_maps[i] = gaussian_filter(score_maps[i], sigma=4)

        scores_img = score_maps.reshape(score_maps.shape[0], -1).max(axis=1)

        return scores_img.tolist(), [el for el in score_maps], None, gt_list, gt_mask_list
# 한국어 코드 안내: 이 파일은 이상 탐지 방법의 학습·예측 제어을 담당합니다.
# 원본 실행 로직은 변경하지 않고, 초보자용 설명 주석만 추가했습니다.
