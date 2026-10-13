import random
import time
from random import sample
import argparse
import numpy as np
import os
import pickle
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.covariance import LedoitWolf
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.segmentation import mark_boundaries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, resnet18
from datasets import mvtec, lgdata


def _validate_vector(u, dtype=None):
    # XXX Is order='c' really necessary?
    u = np.asarray(u, dtype=dtype, order='c')
    if u.ndim == 1:
        return u
    raise ValueError("Input vector should be 1-D.")


def mahalanobis(u, v, VI, calc_mode):
    VI = np.atleast_2d(VI)
    delta = u - v
    if calc_mode == 'dot':
        m = np.dot(np.dot(delta, VI), delta)
    elif calc_mode == 'einsum':
        m = np.einsum('i,ij,j->', delta, VI, delta)
    elif calc_mode == 'batch_einsum':
        # m = np.einsum('nki,nkij,nkj->nk', delta, VI, delta)
        m = np.einsum('nki,kij,nkj->nk', delta, VI, delta)
    else:
        raise NotImplementedError

    return np.sqrt(m)


# hi

# device setup
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser('PaDiM')
    # parser.add_argument('--dataset', type=str, default='mvtec')
    parser.add_argument('--calc_mode', type=str, default='batch_einsum',
                        choices=['einsum', 'batch_einsum', 'dot'])
    parser.add_argument('--dataset', type=str, default='lgdata')
    # parser.add_argument('--data_path', type=str, default='/home/robert.lim/datasets/mvtec')
    parser.add_argument('--data_path', type=str,
                        default='/home/robert.lim/main/config.json')
    # parser.add_argument('--save_path', type=str, default='./mvtec_result')
    parser.add_argument('--save_path', type=str, default='./lg_result')
    parser.add_argument(
        '--arch', type=str, choices=['resnet18', 'wide_resnet50_2'], default='wide_resnet50_2')
    return parser.parse_args()


def plot_fig(test_img, scores, gts, threshold, save_dir, class_name):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[2].imshow(img, cmap='gray', interpolation='none')
        ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[2].title.set_text('Predicted heat map')
        ax_img[3].imshow(mask, cmap='gray')
        ax_img[3].title.set_text('Predicted mask')
        ax_img[4].imshow(vis_img)
        ax_img[4].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        fig_img.savefig(os.path.join(
            save_dir, class_name + '_{}'.format(i)), dpi=100)
        plt.close()


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)

    return x


def embedding_concat(x, y):
    """
    x, y 의 #channels, #height, #width 같게 만들어 concat
    """
    B, C1, H1, W1 = x.size()  # x.shape torch.Size([209, 256, 56, 56])
    _, C2, H2, W2 = y.size()  # y.shape torch.Size([209, 512, 28, 28])
    s = int(H1 / H2)
    # --> x.shape torch.Size([209, 1024, 784])
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    # --> x.shape torch.Size([209, 256, 4, 28, 28])
    x = x.view(B, C1, -1, H2, W2)
    # --> z.shape torch.Size([209, 768, 4, 28, 28])
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        # 4개 중 하나의 256 ch 의 x 마다, 512 ch 의 y 를 중복 concat
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)  # --> z.shape torch.Size([209, 3072, 784])
    # --> z.shape torch.Size([209, 768, 56, 56]), 즉, x 의 H1, W1 에 맞으면서 channel 단위로 병합된 vector 를 만들어냄
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z
