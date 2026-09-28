from loguru import logger
import wandb
import tqdm
import os

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from scipy.ndimage import gaussian_filter

from trainer.trainer import Trainer
from trainer.RD_lib.resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from trainer.RD_lib.de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50, GroupNorm
from trainer.RD_lib.utils_train import MultiProjectionLayer, Revisit_RDLoss, loss_fucntion
from trainer.RD_lib.noise import Simplex_CLASS
import pandas as pd

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class Trainer_RD_Orig(Trainer):
    def initialize_model(self, **kwargs):
        # 역할: `initialize_model`에 해당하는 작업을 수행.
        # 매개변수: **kwargs.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        encoder, bn = wide_resnet50_2(pretrained=True)
        self.encoder = encoder.to(self.device)
        self.bn = bn.to(self.device)
        encoder.eval()

        if self.args.rd_decoder_norm == 'groupnorm':
            norm_layer = GroupNorm
        else:
            norm_layer = nn.BatchNorm2d

        decoder = de_wide_resnet50_2(pretrained=False, norm_layer=norm_layer)
        self.decoder = decoder.to(self.device)

        self.optimizer = torch.optim.Adam(
            list(decoder.parameters()) + list(bn.parameters()),
            lr=0.005, betas=(0.5, 0.999))

    def set_ea_modules(self):
        # 역할: 객체의 설정 또는 내부 상태를 지정.
        # 매개변수: 없음.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        pass

    def train(self, training_data, val_data, test_data, dataset_name):
        # 역할: 학습 데이터를 사용해 모델 파라미터를 갱신.
        # 매개변수: training_data, val_data, test_data, dataset_name.
        # 반환값: 보통 없음(None)이며, 객체 상태가 변경됩니다..
        self.evaluation_results = []
        epochs = self.args.meta_epochs

        bn, encoder, decoder = self.bn, self.encoder, self.decoder

        with tqdm.tqdm(range(epochs), desc="Training...") as tt:
            for epoch in tt:
                bn.train()
                decoder.train()
                loss_list = []

                for data_item in training_data:
                    img = data_item['image'].to(self.device)
                    inputs = encoder(img)
                    outputs = decoder(bn(inputs))
                    loss = loss_fucntion(inputs, outputs)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    loss_list.append(loss.item())

                tt.set_postfix(loss=np.mean(loss_list))

        self.record_evaluation_epoch(test_data)

    def predict(self, test_data, prefix=""):
        # 역할: 입력의 예측 또는 이상 탐지 점수와 map을 계산.
        # 매개변수: test_data, prefix.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        encoder, bn, decoder = self.encoder, self.bn, self.decoder
        bn.eval()
        decoder.eval()

        scores = []
        score_maps = []
        features = []
        labels_gt = []
        masks_gt = []

        with tqdm.tqdm(test_data, desc="Inferring...", leave=False) as data_iterator:
            with torch.no_grad():
                for data_item in data_iterator:
                    labels_gt.extend(data_item['is_anomaly'].numpy().tolist())
                    masks_gt.extend(data_item['mask'].numpy().tolist())
                    img = data_item['image'].to(self.device)
                    inputs = encoder(img)
                    outputs = decoder(bn(inputs))
                    anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
                    anomaly_map = gaussian_filter(anomaly_map, sigma=4)
                    score_maps.append(anomaly_map)
                    scores.append(np.max(anomaly_map))

        return scores, score_maps, features, labels_gt, masks_gt


def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    # 역할: `cal_anomaly_map`에 해당하는 작업을 수행.
    # 매개변수: fs_list, ft_list, out_size, amap_mode.
    # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
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
# 한국어 코드 안내: 이 파일은 이상 탐지 방법의 학습·예측 제어을 담당합니다.
# 원본 실행 로직은 변경하지 않고, 초보자용 설명 주석만 추가했습니다.
