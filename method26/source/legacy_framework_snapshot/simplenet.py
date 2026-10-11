# ------------------------------------------------------------------
# SimpleNet: A Simple Network for Image Anomaly Detection and Localization (https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_SimpleNet_A_Simple_Network_for_Image_Anomaly_Detection_and_Localization_CVPR_2023_paper.pdf)
# Github source: https://github.com/DonaldRR/SimpleNet
# Licensed under the MIT License [see LICENSE for details]
# The script is based on the code of PatchCore (https://github.com/amazon-science/patchcore-inspection)
# ------------------------------------------------------------------

"""detection methods."""
import torch


def init_weight(m):

    # 역할: `init_weight`에 해당하는 작업을 수행.
    # 매개변수: m.
    # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
    # 상세 흐름: 주요 호출은 `isinstance`, `torch.nn.init.xavier_normal_`입니다.
    # 제어 흐름: 조건 분기 2개입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 0개입니다.
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)


class Discriminator(torch.nn.Module):
    def __init__(self, in_planes, n_layers=1, hidden=None):
        # 역할: 객체를 사용할 수 있도록 필요한 속성과 구성요소를 초기화.
        # 매개변수: in_planes, n_layers, hidden.
        # 반환값: 없음. 초기화가 끝난 객체를 Python이 생성합니다..
        # 상세 흐름: 주요 호출은 `super.__init__`, `torch.nn.Sequential`, `range`, `torch.nn.Linear`, `self.apply`입니다.
        # 제어 흐름: 반복문 1개입니다.
        # 상태 영향: `self.body`, `self.tail`; return 경로는 0개입니다.
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
        # 역할: 입력을 신경망 계층에 통과시켜 출력 tensor를 계산.
        # 매개변수: x.
        # 반환값: 모델이 계산한 tensor 또는 모델 출력입니다..
        # 상세 흐름: 주요 호출은 `self.body`, `self.tail`입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        x = self.body(x)
        out = self.tail(x)
        return out

class DiscriminatorMid(torch.nn.Module):
    def __init__(self, in_planes, out_planes, n_layers=1, hidden=None):
        # 역할: 객체를 사용할 수 있도록 필요한 속성과 구성요소를 초기화.
        # 매개변수: in_planes, out_planes, n_layers, hidden.
        # 반환값: 없음. 초기화가 끝난 객체를 Python이 생성합니다..
        # 상세 흐름: 주요 호출은 `super.__init__`, `torch.nn.Sequential`, `range`, `torch.nn.Linear`, `self.apply`입니다.
        # 제어 흐름: 반복문 1개입니다.
        # 상태 영향: `self.body`, `self.mid`, `self.tail`; return 경로는 0개입니다.
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
        # 역할: 입력을 신경망 계층에 통과시켜 출력 tensor를 계산.
        # 매개변수: x, ret_feat.
        # 반환값: 모델이 계산한 tensor 또는 모델 출력입니다..
        # 상세 흐름: 주요 호출은 `self.body`, `self.mid`, `self.tail`입니다.
        # 제어 흐름: 조건 분기 1개입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 2개입니다.
        x = self.body(x)
        x = self.mid(x)
        out = self.tail(x)
        if ret_feat:
            return out, x
        else:
            return out

class DiscriminatorInPlanes(torch.nn.Module):
    def __init__(self, in_planes, out_planes, n_layers=1, hidden=None):
        # 역할: 객체를 사용할 수 있도록 필요한 속성과 구성요소를 초기화.
        # 매개변수: in_planes, out_planes, n_layers, hidden.
        # 반환값: 없음. 초기화가 끝난 객체를 Python이 생성합니다..
        # 상세 흐름: 주요 호출은 `super.__init__`, `torch.nn.Sequential`, `range`, `torch.nn.Linear`, `self.apply`입니다.
        # 제어 흐름: 반복문 1개입니다.
        # 상태 영향: `self.body`, `self.tail`; return 경로는 0개입니다.
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
        # 역할: 입력을 신경망 계층에 통과시켜 출력 tensor를 계산.
        # 매개변수: x.
        # 반환값: 모델이 계산한 tensor 또는 모델 출력입니다..
        # 상세 흐름: 주요 호출은 `self.body`, `self.tail`입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        x = self.body(x)
        x = self.tail(x)
        return x


class Projection(torch.nn.Module):

    def __init__(self, in_planes, out_planes=None, n_layers=1, layer_type=0, conv=False):
        # 역할: 객체를 사용할 수 있도록 필요한 속성과 구성요소를 초기화.
        # 매개변수: in_planes, out_planes, n_layers, layer_type, conv.
        # 반환값: 없음. 초기화가 끝난 객체를 Python이 생성합니다..
        # 상세 흐름: 주요 호출은 `super.__init__`, `torch.nn.Sequential`, `range`, `self.apply`, `self.layers.add_module`입니다.
        # 제어 흐름: 반복문 1개, 조건 분기 3개입니다.
        # 상태 영향: `self.layers`; return 경로는 0개입니다.
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
        # 역할: 입력을 신경망 계층에 통과시켜 출력 tensor를 계산.
        # 매개변수: x.
        # 반환값: 모델이 계산한 tensor 또는 모델 출력입니다..
        # 상세 흐름: 주요 호출은 `self.layers`입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        x = self.layers(x)
        return x

import torch.nn as nn

class ProjectionBn(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=1, bn=True, bias=False):
        # 역할: 객체를 사용할 수 있도록 필요한 속성과 구성요소를 초기화.
        # 매개변수: in_dim, hidden_dim, out_dim, n_layers, bn, bias.
        # 반환값: 없음. 초기화가 끝난 객체를 Python이 생성합니다..
        # 상세 흐름: 주요 호출은 `super.__init__`, `range`, `nn.Sequential`, `self.apply`, `layers.append`입니다.
        # 제어 흐름: 반복문 1개, 조건 분기 4개입니다.
        # 상태 영향: `self.layers`; return 경로는 0개입니다.
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
        # 역할: 입력을 신경망 계층에 통과시켜 출력 tensor를 계산.
        # 매개변수: x.
        # 반환값: 모델이 계산한 tensor 또는 모델 출력입니다..
        # 상세 흐름: 주요 호출은 `self.layers`입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        x = self.layers(x)
        return x
# 한국어 코드 안내: 이 파일은 이 모듈에 포함된 기능의 구현을 담당합니다.
# 원본 실행 로직은 변경하지 않고, 초보자용 설명 주석만 추가했습니다.
