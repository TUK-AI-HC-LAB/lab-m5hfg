# ------------------------------------------------------------------
# SimpleNet: A Simple Network for Image Anomaly Detection and Localization (https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_SimpleNet_A_Simple_Network_for_Image_Anomaly_Detection_and_Localization_CVPR_2023_paper.pdf)
# Github source: https://github.com/DonaldRR/SimpleNet
# Licensed under the MIT License [see LICENSE for details]
# The script is based on the code of PatchCore (https://github.com/amazon-science/patchcore-inspection)
# ------------------------------------------------------------------

"""detection methods."""
import torch


def init_weight(m):

    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)


class Discriminator(torch.nn.Module):
    def __init__(self, in_planes, n_layers=1, hidden=None):
        super(Discriminator, self).__init__()

        _hidden = in_planes if hidden is None else hidden
        self.body = torch.nn.Sequential()
        for i in range(n_layers-1):
            _in = in_planes if i == 0 else _hidden
            _hidden = int(_hidden // 1.5) if hidden is None else hidden
            self.body.add_module('block%d' % (i+1),
                                 torch.nn.Sequential(
                                     torch.nn.Linear(_in, _hidden),
                                     torch.nn.BatchNorm1d(_hidden),
                                     torch.nn.LeakyReLU(0.2)
            ))
        self.tail = torch.nn.Linear(_hidden, 1, bias=False)
        self.apply(init_weight)

    def forward(self, x):
        x = self.body(x)
        out = self.tail(x)
        return out

class DiscriminatorMid(torch.nn.Module):
    def __init__(self, in_planes, out_planes, n_layers=1, hidden=None):
        super().__init__()

        _hidden = in_planes if hidden is None else hidden
        self.body = torch.nn.Sequential()
        for i in range(n_layers-1):
            _in = in_planes if i == 0 else _hidden
            _hidden = int(_hidden // 1.5) if hidden is None else hidden
            self.body.add_module('block%d' % (i+1),
                                 torch.nn.Sequential(
                                     torch.nn.Linear(_in, _hidden),
                                     torch.nn.BatchNorm1d(_hidden),
                                     torch.nn.LeakyReLU(0.2)
            ))
        self.mid = torch.nn.Linear(_hidden, out_planes, bias=False)
        self.tail = torch.nn.Linear(out_planes, 1, bias=False)
        self.apply(init_weight)

    def forward(self, x, ret_feat=False):
        x = self.body(x)
        x = self.mid(x)
        out = self.tail(x)
        if ret_feat:
            return out, x
        else:
            return out

class DiscriminatorInPlanes(torch.nn.Module):
    def __init__(self, in_planes, out_planes, n_layers=1, hidden=None):
        super().__init__()

        _hidden = in_planes if hidden is None else hidden
        self.body = torch.nn.Sequential()
        for i in range(n_layers-1):
            _in = in_planes if i == 0 else _hidden
            _hidden = int(_hidden // 1.5) if hidden is None else hidden
            self.body.add_module('block%d' % (i+1),
                                 torch.nn.Sequential(
                                     torch.nn.Linear(_in, _hidden),
                                     torch.nn.BatchNorm1d(_hidden),
                                     torch.nn.LeakyReLU(0.2)
            ))
        self.tail = torch.nn.Linear(_hidden, out_planes, bias=False)
        self.apply(init_weight)

    def forward(self, x):
        x = self.body(x)
        x = self.tail(x)
        return x


class Projection(torch.nn.Module):

    def __init__(self, in_planes, out_planes=None, n_layers=1, layer_type=0, conv=False):
        super(Projection, self).__init__()

        if out_planes is None:
            out_planes = in_planes
        self.layers = torch.nn.Sequential()
        _in = None
        _out = None
        for i in range(n_layers):
            _in = in_planes if i == 0 else _out
            _out = max(in_planes // (i+4),
                       out_planes) if i < n_layers - 1 else out_planes
            self.layers.add_module(f"{i}fc",
                                   torch.nn.Conv2d(_in, _out, 1) if conv else torch.nn.Linear(_in, _out))
            if i < n_layers - 1:
                # if layer_type > 0:
                #     self.layers.add_module(f"{i}bn",
                #                            torch.nn.BatchNorm1d(_out))
                if layer_type > 1:
                    self.layers.add_module(f"{i}relu",
                                           torch.nn.LeakyReLU(.2))
        self.apply(init_weight)

    def forward(self, x):
        # x = .1 * self.layers(x) + x
        x = self.layers(x)
        return x

import torch.nn as nn

class ProjectionBn(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=1, bn=True, bias=False):
        super(ProjectionBn, self).__init__()
        layers = []
        for i in range(n_layers):
            # 입력과 출력 차원 설정
            if i == 0:
                input_dim = in_dim
                output_dim = hidden_dim if n_layers > 1 else out_dim
            elif i == n_layers - 1:
                input_dim = hidden_dim
                output_dim = out_dim
            else:
                input_dim = hidden_dim
                output_dim = hidden_dim
            # 선형 레이어 추가
            layers.append(nn.Linear(input_dim, output_dim, bias=bias))
            # 마지막 레이어가 아닌 경우 배치 정규화와 활성화 함수 추가
            if i < n_layers - 1:
                if bn:
                    layers.append(nn.BatchNorm1d(output_dim))
                layers.append(nn.LeakyReLU(0.2))
        self.layers = nn.Sequential(*layers)
        self.apply(init_weight)

    def forward(self, x):
        x = self.layers(x)
        return x

