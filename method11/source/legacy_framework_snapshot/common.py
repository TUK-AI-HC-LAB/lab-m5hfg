import copy
from typing import List

import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F


class _BaseMerger:
    def __init__(self):
        # 역할: 객체를 사용할 수 있도록 필요한 속성과 구성요소를 초기화.
        # 매개변수: 없음.
        # 반환값: 없음. 초기화가 끝난 객체를 Python이 생성합니다..
        # 상세 흐름: 주요 호출은 직접 호출하는 다른 함수 없음입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 0개입니다.
        """Merges feature embedding by name."""

    def merge(self, features: list):
        # 역할: `merge`에 해당하는 작업을 수행.
        # 매개변수: features.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 상세 흐름: 주요 호출은 `np.concatenate`, `self._reduce`입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        features = [self._reduce(feature) for feature in features]
        return np.concatenate(features, axis=1)


class AverageMerger(_BaseMerger):
    @staticmethod
    def _reduce(features):
        # NxCxWxH -> NxC
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: features.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 상세 흐름: 주요 호출은 `features.reshape.mean`, `features.reshape`입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        return features.reshape([features.shape[0], features.shape[1], -1]).mean(
            axis=-1
        )


class ConcatMerger(_BaseMerger):
    @staticmethod
    def _reduce(features):
        # NxCxWxH -> NxCWH
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: features.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 상세 흐름: 주요 호출은 `features.reshape`, `len`입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        return features.reshape(len(features), -1)


class Preprocessing(torch.nn.Module):
    def __init__(self, input_dims, output_dim):
        # 역할: 객체를 사용할 수 있도록 필요한 속성과 구성요소를 초기화.
        # 매개변수: input_dims, output_dim.
        # 반환값: 없음. 초기화가 끝난 객체를 Python이 생성합니다..
        # 상세 흐름: 주요 호출은 `super.__init__`, `torch.nn.ModuleList`, `MeanMapper`, `self.preprocessing_modules.append`, `super`입니다.
        # 제어 흐름: 반복문 1개입니다.
        # 상태 영향: `self.input_dims`, `self.output_dim`, `self.preprocessing_modules`; return 경로는 0개입니다.
        super(Preprocessing, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim

        self.preprocessing_modules = torch.nn.ModuleList()
        for input_dim in input_dims:
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    def forward(self, features):
        # 역할: 입력을 신경망 계층에 통과시켜 출력 tensor를 계산.
        # 매개변수: features.
        # 반환값: 모델이 계산한 tensor 또는 모델 출력입니다..
        # 상세 흐름: 주요 호출은 `zip`, `torch.stack`, `_features.append`, `module`입니다.
        # 제어 흐름: 반복문 1개입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            """
            (Pdb) p feature.shape
                torch.Size([10368, 512, 3, 3])
            (Pdb) p module(feature).shape
                torch.Size([10368, 1536])
            """
            _features.append(module(feature))
        return torch.stack(_features, dim=1)


class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        # 역할: 객체를 사용할 수 있도록 필요한 속성과 구성요소를 초기화.
        # 매개변수: preprocessing_dim.
        # 반환값: 없음. 초기화가 끝난 객체를 Python이 생성합니다..
        # 상세 흐름: 주요 호출은 `super.__init__`, `super`입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: `self.preprocessing_dim`; return 경로는 0개입니다.
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        # 역할: 입력을 신경망 계층에 통과시켜 출력 tensor를 계산.
        # 매개변수: features.
        # 반환값: 모델이 계산한 tensor 또는 모델 출력입니다..
        # 상세 흐름: 주요 호출은 `features.reshape`, `F.adaptive_avg_pool1d.squeeze`, `len`, `F.adaptive_avg_pool1d`입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        features = features.reshape(len(features), 1, -1)
        # features.shape == torch.Size([10368, 1, 4608])
        # self.preprocessing_dim == 1536 NOTE 정해진 dim 으로 adaptive_avg_pool1d 를 적용
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)

class LayerwiseAggregator(torch.nn.Module):
    """
    (B,L,N,D) -> (B,N,D)

    Averaging feature maps along layer-axis.
    """
    def __init__(self):
        # 역할: 객체를 사용할 수 있도록 필요한 속성과 구성요소를 초기화.
        # 매개변수: 없음.
        # 반환값: 없음. 초기화가 끝난 객체를 Python이 생성합니다..
        # 상세 흐름: 주요 호출은 `super.__init__`, `super`입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 0개입니다.
        super(LayerwiseAggregator, self).__init__()

    def forward(self, features):
        # 역할: 입력을 신경망 계층에 통과시켜 출력 tensor를 계산.
        # 매개변수: features.
        # 반환값: 모델이 계산한 tensor 또는 모델 출력입니다..
        # 상세 흐름: 주요 호출은 `features.mean`입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        features = features.mean(axis=1)
        return features


class Aggregator(torch.nn.Module):
    def __init__(self, target_dim):
        # 역할: 객체를 사용할 수 있도록 필요한 속성과 구성요소를 초기화.
        # 매개변수: target_dim.
        # 반환값: 없음. 초기화가 끝난 객체를 Python이 생성합니다..
        # 상세 흐름: 주요 호출은 `super.__init__`, `super`입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: `self.target_dim`; return 경로는 0개입니다.
        super(Aggregator, self).__init__()
        self.target_dim = target_dim

    def forward(self, features):
        # 역할: 입력을 신경망 계층에 통과시켜 출력 tensor를 계산.
        # 매개변수: features.
        # 반환값: 모델이 계산한 tensor 또는 모델 출력입니다..
        # 상세 흐름: 주요 호출은 `features.reshape`, `F.adaptive_avg_pool1d`, `len`입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        """Returns reshaped and average pooled features."""
        # batchsize x number_of_layers x input_dim -> batchsize x target_dim
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)


class RescaleSegmentor:
    def __init__(self, device, target_size=224, smoothing=4):
        # 역할: 객체를 사용할 수 있도록 필요한 속성과 구성요소를 초기화.
        # 매개변수: device, target_size, smoothing.
        # 반환값: 없음. 초기화가 끝난 객체를 Python이 생성합니다..
        # 상세 흐름: 주요 호출은 직접 호출하는 다른 함수 없음입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: `self.device`, `self.target_size`, `self.smoothing`; return 경로는 0개입니다.
        self.device = device
        self.target_size = target_size
        self.smoothing = smoothing

    def convert_to_segmentation(self, patch_scores):
        # 역할: `convert_to_segmentation`에 해당하는 작업을 수행.
        # 매개변수: patch_scores.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 상세 흐름: 주요 호출은 `torch.no_grad`, `isinstance`, `patch_scores.to`, `_scores.unsqueeze`, `F.interpolate`입니다.
        # 제어 흐름: 조건 분기 2개, 자원/문맥 관리 1개입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 2개입니다.
        # patch 격자는 보통 원본 mask보다 작습니다. bilinear interpolation으로 같은 크기로 확대합니다.
        if patch_scores.shape[-1] != self.target_size[-1]: # NOTE target size 와 동일하면 굳이 처리할 필요없음
            with torch.no_grad():
                if isinstance(patch_scores, np.ndarray):
                    patch_scores = torch.from_numpy(patch_scores)
                _scores = patch_scores.to(self.device)
                _scores = _scores.unsqueeze(1)
                _scores = F.interpolate( _scores, size=self.target_size, mode="bilinear", align_corners=False)
                _scores = _scores.squeeze(1)
                patch_scores = _scores.cpu().numpy()

            # 확대 과정에서 생기는 작은 점 잡음을 Gaussian filter로 부드럽게 합니다.
            return [
                ndimage.gaussian_filter(patch_score, sigma=self.smoothing)
                for patch_score in patch_scores
            ]
        else:
            return [patch_score
                for patch_score in patch_scores
            ]

class RescaleSegmentorSimpleNet:
    def __init__(self, device, target_size=224):
        # 역할: 객체를 사용할 수 있도록 필요한 속성과 구성요소를 초기화.
        # 매개변수: device, target_size.
        # 반환값: 없음. 초기화가 끝난 객체를 Python이 생성합니다..
        # 상세 흐름: 주요 호출은 직접 호출하는 다른 함수 없음입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: `self.device`, `self.target_size`, `self.smoothing`; return 경로는 0개입니다.
        self.device = device
        self.target_size = target_size
        self.smoothing = 4

    def convert_to_segmentation(self, patch_scores, features=None):
        # 역할: `convert_to_segmentation`에 해당하는 작업을 수행.
        # 매개변수: patch_scores, features.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 상세 흐름: 주요 호출은 `torch.zeros`, `torch.no_grad`, `isinstance`, `patch_scores.to`, `_scores.unsqueeze`입니다.
        # 제어 흐름: 반복문 1개, 조건 분기 4개, 자원/문맥 관리 1개입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        if features is None:
            features = torch.zeros(*patch_scores.shape, 1536)

        with torch.no_grad():
            if isinstance(patch_scores, np.ndarray):
                patch_scores = torch.from_numpy(patch_scores)
            _scores = patch_scores.to(self.device)
            _scores = _scores.unsqueeze(1)
            _scores = F.interpolate(
                _scores, size=self.target_size, mode="bilinear", align_corners=False
            )
            _scores = _scores.squeeze(1)
            patch_scores = _scores.cpu().numpy()

            if isinstance(features, np.ndarray):
                features = torch.from_numpy(features)
            features = features.to(self.device).permute(0, 3, 1, 2)
            if self.target_size[0] * self.target_size[1] * features.shape[0] * features.shape[1] >= 2**31:
                subbatch_size = int(
                    (2**31-1) / (self.target_size[0] * self.target_size[1] * features.shape[1]))
                interpolated_features = []
                for i_subbatch in range(int(features.shape[0] / subbatch_size + 1)):
                    subfeatures = features[i_subbatch *
                                           subbatch_size:(i_subbatch+1)*subbatch_size]
                    subfeatures = subfeatures.unsuqeeze(0) if len(
                        subfeatures.shape) == 3 else subfeatures
                    subfeatures = F.interpolate(
                        subfeatures, size=self.target_size, mode="bilinear", align_corners=False
                    )
                    interpolated_features.append(subfeatures)
                features = torch.cat(interpolated_features, 0)
            else:
                features = F.interpolate(
                    features, size=self.target_size, mode="bilinear", align_corners=False
                )
            features = features.cpu().numpy()

        return [
            ndimage.gaussian_filter(patch_score, sigma=self.smoothing)
            for patch_score in patch_scores
        ], [
            feature
            for feature in features
        ]


class NetworkFeatureAggregator(torch.nn.Module):
    """Efficient extraction of network features."""

    def __init__(self, backbone, layers_to_extract_from, device, train_backbone=False):
        # 역할: 객체를 사용할 수 있도록 필요한 속성과 구성요소를 초기화.
        # 매개변수: backbone, layers_to_extract_from, device, train_backbone.
        # 반환값: 없음. 초기화가 끝난 객체를 Python이 생성합니다..
        # 상세 흐름: 주요 호출은 `super.__init__`, `self.to`, `hasattr`, `handle.remove`, `ForwardHook`입니다.
        # 제어 흐름: 반복문 2개, 조건 분기 4개입니다.
        # 상태 영향: `self.layers_to_extract_from`, `self.backbone`, `self.device`, `self.train_backbone`; return 경로는 0개입니다.
        super(NetworkFeatureAggregator, self).__init__()
        """Extraction of network features.

        Runs a network only to the last layer of the list of layers where
        network features should be extracted from.

        Args:
            backbone: torchvision.model
            layers_to_extract_from: [list of str]
        """
        self.layers_to_extract_from = layers_to_extract_from
        self.backbone = backbone
        self.device = device
        self.train_backbone = train_backbone
        if not hasattr(backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()
        self.outputs = {}

        # NOTE 틀린 부분.
        for extract_layer in layers_to_extract_from:
            forward_hook = ForwardHook(
                self.outputs, extract_layer, layers_to_extract_from[-1]
            )
            if "." in extract_layer:
                # NOTE GraphCore 의 backone 인 Visoin GNN 은 . 이 있으므로 이쪽이 사용됨
                extract_block, extract_idx = extract_layer.split(".")
                network_layer = backbone.__dict__["_modules"][extract_block]
                if extract_idx.isnumeric():
                    extract_idx = int(extract_idx)
                    network_layer = network_layer[extract_idx]
                else:
                    network_layer = network_layer.__dict__[
                        "_modules"][extract_idx]
            else:
                # NOTE resnet 은 .이 없으므로 이쪽이 사용됨
                network_layer = backbone.__dict__["_modules"][extract_layer]

            if isinstance(network_layer, torch.nn.Sequential):
                self.backbone.hook_handles.append(
                    network_layer[-1].register_forward_hook(forward_hook)
                )
            else:
                self.backbone.hook_handles.append(
                    network_layer.register_forward_hook(forward_hook)
                )
        self.to(self.device)

    def forward(self, images, eval=True):
        # 역할: 입력을 신경망 계층에 통과시켜 출력 tensor를 계산.
        # 매개변수: images, eval.
        # 반환값: 모델이 계산한 tensor 또는 모델 출력입니다..
        # 상세 흐름: 주요 호출은 `self.outputs.clear`, `self.backbone`, `torch.no_grad`입니다.
        # 제어 흐름: 조건 분기 1개, 자원/문맥 관리 1개입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        self.outputs.clear()
        if self.train_backbone and not eval:
            self.backbone(images)
        else:
            with torch.no_grad():
                # The backbone will throw an Exception once it reached the last
                # layer to compute features from. Computation will stop there.
                try:
                    _ = self.backbone(images)
                except LastLayerToExtractReachedException:
                    pass
        return self.outputs

    def feature_dimensions(self, input_shape):
        # 역할: `feature_dimensions`에 해당하는 작업을 수행.
        # 매개변수: input_shape.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 상세 흐름: 주요 호출은 `torch.ones.to`, `self`, `torch.ones`, `list`입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        """Computes the feature dimensions for all layers given input_shape."""
        _input = torch.ones([1] + list(input_shape)).to(self.device)
        _output = self(_input)
        return [_output[layer].shape[1] for layer in self.layers_to_extract_from]


class ForwardHook:
    def __init__(self, hook_dict, layer_name: str, last_layer_to_extract: str):
        # 역할: 객체를 사용할 수 있도록 필요한 속성과 구성요소를 초기화.
        # 매개변수: hook_dict, layer_name, last_layer_to_extract.
        # 반환값: 없음. 초기화가 끝난 객체를 Python이 생성합니다..
        # 상세 흐름: 주요 호출은 `copy.deepcopy`입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: `self.hook_dict`, `self.layer_name`, `self.raise_exception_to_break`; return 경로는 0개입니다.
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.raise_exception_to_break = copy.deepcopy(
            layer_name == last_layer_to_extract
        )

    def __call__(self, module, input, output):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: module, input, output.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 상세 흐름: 주요 호출은 직접 호출하는 다른 함수 없음입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        self.hook_dict[self.layer_name] = output
        # if self.raise_exception_to_break:
        #     raise LastLayerToExtractReachedException()
        return None


class LastLayerToExtractReachedException(Exception):
    pass
# 한국어 코드 안내: 이 파일은 여러 모듈이 함께 쓰는 보조 기능을 담당합니다.
# 원본 실행 로직은 변경하지 않고, 초보자용 설명 주석만 추가했습니다.
