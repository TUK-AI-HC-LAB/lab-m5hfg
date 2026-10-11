import os
import math
import pickle

import scipy.ndimage as ndimage
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import time

import COAD.COAD as COAD
from COAD.common import NetworkFeatureAggregator, RescaleSegmentor
from COAD.sampler import WeightedGreedyCoresetSampler
import COAD.encoder
import common
from trainer.trainer_patchcore import FaissNN, ApproximateGreedyCoresetSampler, NearestNeighbourScorer
from trainer.trainer import Trainer
#import COAD.fft
from loguru import logger
import matplotlib.pyplot as plt

import torch.nn as nn
from sklearn.neighbors import LocalOutlierFactor
import open_clip

def contrastive_loss(device, image_embeds, text_embeds, labels):
    # 역할: `contrastive_loss`에 해당하는 작업을 수행.
    # 매개변수: device, image_embeds, text_embeds, labels.
    # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
    # 상세 흐름: 주요 호출은 `F.normalize`, `.to`, `labels.float`, `torch.cdist`, `torch.pow`입니다.
    # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
    import torch.nn.functional as F
    image_embeds = F.normalize(image_embeds, dim=1)
    text_embeds = F.normalize(text_embeds, dim=1)

    labels = (labels.unsqueeze(1) == labels.unsqueeze(0)).to(device)
    labels = labels.float()

    distances = torch.cdist(image_embeds, text_embeds, p=2)

    positive_pairs = labels * torch.pow(distances, 2)
    negative_pairs = (1 - labels) * torch.pow(torch.clamp(1 - distances, min=0.0), 2)

    loss = torch.sum(positive_pairs + negative_pairs) / len(image_embeds)

    return loss

class Net(nn.Module):
    def __init__(self):
        # 역할: 객체를 사용할 수 있도록 필요한 속성과 구성요소를 초기화.
        # 매개변수: 없음.
        # 반환값: 없음. 초기화가 끝난 객체를 Python이 생성합니다..
        # 상세 흐름: 주요 호출은 `super.__init__`, `nn.Linear`, `nn.Dropout`, `super`입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: `self.fc1`, `self.fc2`, `self.fc3`, `self.fc4`; return 경로는 0개입니다.
        super(Net, self).__init__()
        self.fc1 = nn.Linear(512, 384)
        self.fc2 = nn.Linear(384, 384)
        self.fc3 = nn.Linear(384, 384)
        self.fc4 = nn.Linear(384, 384)
        
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # 역할: 입력을 신경망 계층에 통과시켜 출력 tensor를 계산.
        # 매개변수: x.
        # 반환값: 모델이 계산한 tensor 또는 모델 출력입니다..
        # 상세 흐름: 주요 호출은 `self.fc1`, `self.fc2`, `self.dropout`, `self.fc3`, `self.fc4`입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        x = self.fc1(x)
        x = self.fc2(F.relu(x))
        x = self.dropout(x)
        x = self.fc3(F.relu(x))
        x = self.dropout(x)
        x = self.fc4(F.relu(x))
        return x

class COADNetwork(Trainer):
    def __init__(self, device, params):
        # 역할: 객체를 사용할 수 있도록 필요한 속성과 구성요소를 초기화.
        # 매개변수: device, params.
        # 반환값: 없음. 초기화가 끝난 객체를 Python이 생성합니다..
        # 상세 흐름: 주요 호출은 `super.__init__`, `super`입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: `self.params`; return 경로는 0개입니다.
        """COAD anomaly detection class."""
        super().__init__(device)
        self.params = params
    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        bpm=False,  # Modified to False since 'bpm' is not passed via args
        patchsize=8,
        anomaly_scorer_num_nn=5,
        local_nn_method=FaissNN(False, 8),
        topk=0.05,
        lmda=0.0,
        thres=0.1,
        temp=0.0,
        return_topk_index=False,
        agg_type='layer',
        **kwargs,
    ):
        # 역할: 필요한 모델·가중치·설정을 준비.
        # 매개변수: backbone, layers_to_extract_from, device, input_shape, bpm, patchsize, anomaly_scorer_num_nn, local_nn_method, topk, lmda, thres, temp, return_topk_index, agg_type, **kwargs.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 상세 흐름: 주요 호출은 `FaissNN`, `backbone.to`, `PatchScorer`, `AttentionMask`, `torch.nn.ModuleDict`입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: `self.backbone`, `self.layers_to_extract_from`, `self.input_shape`, `self.bpm`; return 경로는 0개입니다.
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape
        self.bpm = bpm
        self.args = kwargs['args']

        self.device = device
        
        assert topk > 0.0, "Top k-ratio should be greater than zero. Please retry."
        
        self.patch_scorer = PatchScorer(topk, return_topk_index)
        """
        (Pdb) p topk
            0.05
        (Pdb) p return_topk_index
            False
        """
        
        self.attention_mask = AttentionMask(self.backbone, self.layers_to_extract_from[-1], patchsize, rollout='sup' in self.backbone.name)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        """
        (Pdb) p self.layers_to_extract_from
            ['2', '3', '5', '6', '7', '8', '9']
        """

        self.forward_modules["feature_aggregator"] = feature_aggregator

        self.agg_type = agg_type
        layer_aggregator = common.LayerwiseAggregator()
        _ = layer_aggregator.to(self.device)

        self.forward_modules["layer_aggregator"] = layer_aggregator

        self.anomaly_scorer = NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_scorer_num_nn, nn_method=local_nn_method
        )

        self.anomaly_segmentor = RescaleSegmentor(
            device=self.device, target_size=(self.args.masksize, self.args.masksize)
        )

        self.lmda = lmda
        self.thres = thres
        self.topk_index = return_topk_index

        #self.dog_filter = COAD.fft.DifferenceOfGaussian(3, 1.0, 2.0).to(self.device)
        
        ## FFT ## 
        self.net = None
        
        ## CLIP ##
        self.clip = None

        
        ############ SoftPatch ############
        self.featuresampler = WeightedGreedyCoresetSampler(self.args.coreset_ratio, self.device, num_coreset_samples=self.args.patchcore_coreset_num)
        self.patch_weight = None
        self.feature_shape = []
        self.lof_k = 5
        self.threshold = 0.15
        self.coreset_weight = None
        self.weight_method = 'lof'
        self.soft_weight_flag = True
        
    def embed(self, data):
        # 역할: `embed`에 해당하는 작업을 수행.
        # 매개변수: data.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 상세 흐름: 주요 호출은 `isinstance`, `self._embed`, `torch.no_grad`, `image.to.to`, `features.append`입니다.
        # 제어 흐름: 반복문 1개, 조건 분기 2개, 자원/문맥 관리 1개입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 2개입니다.
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image, pretrain=True))
            return features
        return self._embed(data)

    def _embed(self, images, texts, detach=False, provide_patch_shapes=False, pretrain=True):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: images, texts, detach, provide_patch_shapes, pretrain.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 상세 흐름: 주요 호출은 `.eval`, `_detach`, `torch.no_grad`, `torch.cat`, `self.clip.encode_text`입니다.
        # 제어 흐름: 조건 분기 3개, 자원/문맥 관리 2개입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 4개입니다.
        """Returns feature embeddings for images."""

        def _detach(features):
            # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
            # 매개변수: features.
            # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
            # 상세 흐름: 주요 호출은 `features.detach.cpu.numpy`, `features.detach.cpu`, `features.detach`입니다.
            # 제어 흐름: 조건 분기 1개입니다.
            # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 2개입니다.
            if detach:
                return features.detach().cpu().numpy()
            return features

        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)
        features = [features[layer] for layer in self.layers_to_extract_from] 

        if "vit" in self.backbone.name:
            patch_shapes = [(int(x.shape[1]**0.5), int(x.shape[1]**0.5)) for x in features]

            features = [x.unsqueeze(1) for x in features]
            
            features = torch.cat(features, dim=1)

            features = self.forward_modules["layer_aggregator"](features)
        
        with torch.no_grad():
            text_features = self.clip.encode_text(texts.long())
            text_features = F.normalize(self.net(text_features), dim=1)
            features = torch.cat((features, text_features.unsqueeze(1).repeat(1,785,1)), dim=-1)
        
        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)

    def fit(self, training_data):
        # 역할: 주어진 데이터로 통계량 또는 모델을 학습.
        # 매개변수: training_data.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 상세 흐름: 주요 호출은 `self._fill_memory_bank`입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 0개입니다.
        """COAD training.

        This function computes the embeddings of the training data and fills the
        memory bank of COAD.
        """
        self._fill_memory_bank(training_data)

    def _fill_memory_bank(self, input_data):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: input_data.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 상세 흐름: 주요 호출은 `self.forward_modules.eval`, `torch.cat`, `features.reshape`, `self.anomaly_scorer.fit`, `tqdm.tqdm`입니다.
        # 제어 흐름: 반복문 1개, 조건 분기 1개, 자원/문맥 관리 3개입니다.
        # 상태 영향: `self.feature_shape`, `self.weight_map`, `self.weight_map`, `self.patch_weight`; return 경로는 1개입니다.
        """Computes and sets the support features for COAD."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image, input_text):
            # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
            # 매개변수: input_image, input_text.
            # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
            # 상세 흐름: 주요 호출은 `torch.no_grad`, `input_image.to.to`, `input_text.to.to`, `self._embed`, `input_image.to`입니다.
            # 제어 흐름: 자원/문맥 관리 1개입니다.
            # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                input_text = input_text.to(torch.float).to(self.device)
                return self._embed(input_image, input_text)

        features = []
        
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    text = self._transform_text_to_tokens(image)
                    image = image["image"]
                _features = _image_to_features(image, text)[:, 1:, :]
                    
                features.append(_features)
                torch.cuda.empty_cache()
                
        features = torch.cat(features)
        features = features.reshape(-1, features.shape[-1])
        
        ############ SoftPatch ############
        with torch.no_grad():
            self.feature_shape = self._embed(image.to(torch.float).to(self.device), text.to(torch.float).to(self.device), provide_patch_shapes=True)[1][0]
            patch_weight = self._compute_patch_weight(features) # torch.Size([784000, 384])
            self.weight_map = torch.mean(patch_weight, dim=0)
            self.weight_map = self.weight_map.unsqueeze(1).unsqueeze(0)
            
            patch_weight = patch_weight.reshape(-1)
            threshold = torch.quantile(patch_weight, 1 - self.threshold)
            sampling_weight = torch.where(patch_weight > threshold, 0, 1)
            self.featuresampler.set_sampling_weight(sampling_weight)
            self.patch_weight = patch_weight.clamp(min=0)

            sample_features, sample_indices = self.featuresampler.run(features)
            features = sample_features
            self.coreset_weight = self.patch_weight[sample_indices].cpu().numpy()

        self.anomaly_scorer.fit(detection_features=[features.cpu()])
        
    ############ SoftPatch ############
    def _compute_patch_weight(self, features: np.ndarray):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: features.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 상세 흐름: 주요 호출은 `isinstance`, `self.featuresampler._reduce_features`, `reduced_features.reshape`, `patch_features.reshape`, `patch_features.permute`입니다.
        # 제어 흐름: 조건 분기 6개, 예외 발생 경로 1개입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)

        reduced_features = self.featuresampler._reduce_features(features)

        patch_features = \
            reduced_features.reshape(-1, reduced_features.shape[-1], self.feature_shape[0], self.feature_shape[1]) # torch.Size([1000, 784, 128])
        
        patch_features = patch_features.reshape(-1, self.feature_shape[0]*self.feature_shape[1], reduced_features.shape[-1])
        
        patch_features = patch_features.permute(1, 0, 2) # torch.Size([784, 1000, 128])

        if self.weight_method == "lof":
            patch_weight = self._compute_lof(self.lof_k, patch_features).transpose(-1, -2)
        elif self.weight_method == "lof_gpu":
            patch_weight = self._compute_lof_gpu(self.lof_k, patch_features).transpose(-1, -2)
        elif self.weight_method == "nearest":
            patch_weight = self._compute_nearest_distance(patch_features).transpose(-1, -2)
            patch_weight = patch_weight + 1
        elif self.weight_method == "gaussian":
            gaussian = COAD.multi_variate_gaussian.MultiVariateGaussian(patch_features.shape[2], patch_features.shape[0])
            stats = gaussian.fit(patch_features)
            patch_weight = self._compute_distance_with_gaussian(patch_features, stats).transpose(-1, -2)
            patch_weight = patch_weight + 1
        elif self.weight_method == "attention":
            patch_weight = self._compute_attention(self.lof_k, patch_features).transpose(-1, -2)
        else:
            raise ValueError("Unexpected weight method")

        return patch_weight

    def _attention(self, Q, K, V):
        # Query와 Key 사이의 유사도를 계산 (여기서는 내적을 사용)
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: Q, K, V.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 상세 흐름: 주요 호출은 `F.softmax`, `torch.matmul`, `K.transpose`, `Q.size`입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
        # Softmax를 통해 가중치로 변환
        attention_weights = F.softmax(scores, dim=-1)
        # Value에 가중치를 곱해서 최종 결과 계산
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def _compute_attention(self, k, embedding: torch.Tensor) -> torch.Tensor:
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: k, embedding.
        # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
        # 상세 흐름: 주요 호출은 `torch.zeros`, `range`, `self._attention`, `torch.mean`입니다.
        # 제어 흐름: 반복문 1개입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        patch, batch, _ = embedding.shape   # torch.size([784, 1000, 128])
        
        scores = torch.zeros(size=(patch, batch), device=embedding.device) # torch.Size([784, 1000])
        for i in range(patch):
            Q, K, V = embedding[i],embedding[i],embedding[i]
            output, attention_weights = self._attention(Q,K,V)
            scores[i] = torch.mean(attention_weights, dim=0)
        return scores 

    def _compute_lof(self, k, embedding: torch.Tensor) -> torch.Tensor:
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: k, embedding.
        # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
        # 상세 흐름: 주요 호출은 `LocalOutlierFactor`, `torch.zeros`, `range`, `clf.fit`, `torch.Tensor`입니다.
        # 제어 흐름: 반복문 1개입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        patch, batch, _ = embedding.shape   # torch.size([784, 1000, 128])
        clf = LocalOutlierFactor(n_neighbors=int(k), metric='l2')
        scores = torch.zeros(size=(patch, batch), device=embedding.device) # torch.Size([784, 1000])
        for i in range(patch):
            clf.fit(embedding[i].cpu()) # 배치마다 각 자리의 patch끼리 비교
            scores[i] = torch.Tensor(- clf.negative_outlier_factor_)
        return scores # len(scores)->784, scores[0]->torch.size([1000]) 각 자리별 score

    def _transform_text_to_tokens(self, data_):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: data_.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 상세 흐름: 주요 호출은 `range`, `self.tokenizer`, `len`입니다.
        # 제어 흐름: 반복문 1개입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        text_list = data_['text']
        for i in range(len(text_list)):
            text_list[i] = f'This photo is {text_list[i]}.'
            
        tokens = self.tokenizer(text_list)
        return tokens

    def _train_clip(self, training_data):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: training_data.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 상세 흐름: 주요 호출은 `open_clip.create_model_and_transforms`, `self.clip.to`, `Net.to`, `open_clip.get_tokenizer`, `optim.Adam`입니다.
        # 제어 흐름: 반복문 2개, 조건 분기 1개입니다.
        # 상태 영향: `self.clip`, `self.net`, `self.tokenizer`; return 경로는 0개입니다.
        import torch.optim as optim
        import tqdm

        device = self.device

        # Initialize the CLIP model and coupling network
        import open_clip
        self.clip, _, _ = open_clip.create_model_and_transforms(
            model_name='ViT-B-16',
            pretrained='openai',
            force_quick_gelu=True,
            device=device
        )
        self.clip = self.clip.to(device)
        self.net = Net().to(device)
        self.tokenizer = open_clip.get_tokenizer(model_name='ViT-B-32')

        # Set up the optimizer
        optimizer = optim.Adam([
            {'params': self.clip.parameters(), 'lr': 0.00001, 'betas': (0.9, 0.98), 'eps': 1e-6},
            {'params': self.net.parameters(), 'lr': 0.0001} # NOTE 주어진 yaml 참고함
        ])

        self.clip.train()
        self.net.train()
        for epoch in range(30): # NOTE 주어진 yaml 참고함
            total_loss = 0
            tmp = 0

            progress_bar = tqdm.tqdm(training_data, desc=f"Epoch {epoch + 1}")
            for data in progress_bar:
                images = data['image'].to(device)
                labels = data["is_anomaly"].to(device)
                text = self._transform_text_to_tokens(data).to(device)

                image_features = self.clip.encode_image(images)
                text_features = self.clip.encode_text(text)

                image_features = self.net(image_features)
                text_features = self.net(text_features)

                loss = contrastive_loss(device, image_features, text_features, labels=labels)
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                progress_bar.set_postfix(loss=loss.item() / (progress_bar.n + 1))
                tmp += 1

            logger.info(f'Epoch {epoch+1}: {total_loss/tmp}')

            if  self.args.subtest == 1:
                break  # NOTE 중요. early stopping 해야 함.

        self.clip.eval()
        self.net.eval()


    def train(self, training_data, val_data, test_data, dataset_name):
        # 역할: 학습 데이터를 사용해 모델 파라미터를 갱신.
        # 매개변수: training_data, val_data, test_data, dataset_name.
        # 반환값: 보통 없음(None)이며, 객체 상태가 변경됩니다..
        # 상세 흐름: 주요 호출은 `self._meta_train`입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: `self.evaluation_results`; return 경로는 1개입니다.
        self.evaluation_results = []
        return self._meta_train(training_data, val_data, test_data, dataset_name)

    def _meta_train(self, training_data, val_data, test_data, dataset_name):

        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: training_data, val_data, test_data, dataset_name.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 상세 흐름: 주요 호출은 `torch.utils.data.DataLoader`, `self._train_clip`, `self.fit`, `self.predict`, `self._save_fault_images`입니다.
        # 제어 흐름: 조건 분기 1개입니다.
        # 상태 영향: `self.dataset_name`; return 경로는 1개입니다.
        self.dataset_name = dataset_name
        state_dict = {}
        ckpt_path = None
        # NOTE saved model 을 불러와서 infer 하는 부분. 나중에 SimpleNet 에서 가져오자

        best_record = None

        # CLIP dataloader (batch_size 만 변경함)
        clip_dataloader = torch.utils.data.DataLoader(
            training_data.dataset,
            batch_size=16,
            shuffle=True,
            num_workers=2,
            prefetch_factor=2,
            pin_memory=True,
        )
        self._train_clip(clip_dataloader)
        self.fit(clip_dataloader)


        if ckpt_path is not None:
            torch.save(state_dict, ckpt_path)

        scores, segmentations, _, labels_gt, masks_gt = self.predict(test_data)
        self._save_fault_images(test_data, scores)
        auroc, full_pixel_auroc, pro, saliency_cr_f1 = self._evaluate(scores, segmentations, None, labels_gt, masks_gt)

        self.evaluation_results.append(
            {
                "auroc": auroc,
                "full_pixel_auroc": full_pixel_auroc,
                "pro": pro,
                "saliency_cr_f1": saliency_cr_f1,
            }
        )

        best_record = [auroc, full_pixel_auroc,
                       pro, segmentations, labels_gt, scores, saliency_cr_f1]

        return best_record


    def predict(self, data):
        # 역할: 입력의 예측 또는 이상 탐지 점수와 map을 계산.
        # 매개변수: data.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 상세 흐름: 주요 호출은 `isinstance`, `self._predict`, `self._predict_dataloader`입니다.
        # 제어 흐름: 조건 분기 1개입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 2개입니다.
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)

    def _predict_dataloader(self, dataloader):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: dataloader.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 상세 흐름: 주요 호출은 `self.forward_modules.eval`, `time.time`, `print`, `tqdm.tqdm`, `enumerate`입니다.
        # 제어 흐름: 반복문 2개, 조건 분기 4개, 자원/문맥 관리 1개입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 2개입니다.
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        scores = []
        masks = []
        labels_gt = []
        masks_gt = []

        if self.topk_index:
            topk_features = []

        # Measure FPS
        start_time = time.time()
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=True) as data_iterator:
            for i, image in enumerate(data_iterator):
                if isinstance(image, dict):
                    labels_gt.extend(image["is_anomaly"].numpy().tolist())
                    masks_gt.extend(image["mask"].numpy().tolist())
                    text = self._transform_text_to_tokens(image)
                    image = image["image"]
                                    
                if self.topk_index:
                    _scores, _masks, _topk_features = self._predict(image, text)
                    topk_features.append(_topk_features)
                else:
                    _scores, _masks = self._predict(image, text)

                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)
        end_time = time.time()
        print("FPS:", len(dataloader)/(end_time-start_time))
        
        if self.topk_index:
            return scores, masks, topk_features, labels_gt, masks_gt
        
        return scores, masks, [], labels_gt, masks_gt

    def _predict(self, images, text):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: images, text.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 상세 흐름: 주요 호출은 `images.to.to`, `text.to.to`, `self.forward_modules.eval`, `torch.no_grad`, `self._embed`입니다.
        # 제어 흐름: 조건 분기 2개, 자원/문맥 관리 1개입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 2개입니다.
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device)
        text = text.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        with torch.no_grad():
            features, patch_shapes = self._embed(images, text, provide_patch_shapes=True, pretrain=True)

            features = features[:, 1:, :] # exclude global tokens
            
            pred = self.anomaly_scorer.predict(features.cpu().numpy())
            patch_scores = image_scores = pred[0] # [784,]

            image_scores = self.patch_scorer.unpatch_scores(
                image_scores, batchsize=batchsize
            )

            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_scorer.score(image_scores)

            if self.topk_index:
                topk_index = image_scores[1]
                image_scores = image_scores[0]
                topk_features = np.take_along_axis(features, topk_index, axis=1)

            # Scale the score 
            scores = image_scores

            patch_scores = self.patch_scorer.unpatch_scores(
                patch_scores, batchsize=batchsize
            ) # patch_scores for segmentation
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

            if self.topk_index:
               return [score for score in scores], [mask for mask in masks], topk_features 

        return [score for score in scores], [mask for mask in masks]
    
    @staticmethod
    def _params_file(filepath, prepend=""):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: filepath, prepend.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 상세 흐름: 주요 호출은 `os.path.join`입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        return os.path.join(filepath, prepend + "COAD_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        # 역할: `save_to_path`에 해당하는 작업을 수행.
        # 매개변수: save_path, prepend.
        # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
        # 상세 흐름: 주요 호출은 `logger.info`, `self.anomaly_scorer.save`, `open`, `pickle.dump`, `self._params_file`입니다.
        # 제어 흐름: 조건 분기 1개, 자원/문맥 관리 1개입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 0개입니다.
        logger.info("Saving COAD data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        if self.params:
            COAD_params = self.params
            COAD_params['input_shape'] = self.input_shape

            try:
                COAD_params['backbone.name'] = COAD_params['backbone_name']
                del COAD_params['backbone_name']
            except KeyError:
                pass
        else:
            COAD_params = {
                "backbone.name": self.backbone.name,
                "layers_to_extract_from": self.layers_to_extract_from,
                "input_shape": self.input_shape,
                "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
            }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(COAD_params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        local_nn_method: FaissNN,
        topk: float=0.01,
        bpm: bool=True,
        thres: float=0.1,
        anomaly_scorer_nn: int=1,
        patchsize: int=7,
        prepend: str = "",
    ) -> None:
        # 역할: `load_from_path`에 해당하는 작업을 수행.
        # 매개변수: load_path, device, local_nn_method, topk, bpm, thres, anomaly_scorer_nn, patchsize, prepend.
        # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
        # 상세 흐름: 주요 호출은 `logger.info`, `COAD.encoder.load`, `self.load`, `self.anomaly_scorer.load`, `open`입니다.
        # 제어 흐름: 자원/문맥 관리 1개입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 0개입니다.
        logger.info("Loading and initializing COAD.")
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            COAD_params = pickle.load(load_file)
        
        COAD_params["backbone"] = COAD.encoder.load(
            COAD_params["backbone.name"]
        )
        COAD_params["backbone"].name = COAD_params["backbone.name"]
        del COAD_params["backbone.name"]
        
        COAD_params["topk"] = topk
        COAD_params["bpm"] = bpm
        COAD_params["thres"] = thres
        COAD_params["anomaly_scorer_num_nn"] = anomaly_scorer_nn
        COAD_params["patchsize"] = patchsize
        
        self.load(**COAD_params, device=device, local_nn_method=local_nn_method)

        self.anomaly_scorer.load(load_path, prepend)

class PatchScorer:
    def __init__(self, k, return_index):
        # 역할: 객체를 사용할 수 있도록 필요한 속성과 구성요소를 초기화.
        # 매개변수: k, return_index.
        # 반환값: 없음. 초기화가 끝난 객체를 Python이 생성합니다..
        # 상세 흐름: 주요 호출은 직접 호출하는 다른 함수 없음입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: `self.k`, `self.return_index`; return 경로는 0개입니다.
        self.k = k
        self.return_index = return_index
    
    def unpatch_scores(self, x, batchsize):
        # 역할: `unpatch_scores`에 해당하는 작업을 수행.
        # 매개변수: x, batchsize.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 상세 흐름: 주요 호출은 `x.reshape`입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x, shape=None):
        # x.shape == (1, 65536, 1) == (batchsize, patchsize, 1)
        # 역할: `score`에 해당하는 작업을 수행.
        # 매개변수: x, shape.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 상세 흐름: 주요 호출은 `isinstance`, `int`, `torch.topk`, `torch.mean.reshape`, `torch.from_numpy`입니다.
        # 제어 흐름: 조건 분기 4개입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 4개입니다.
        was_numpy = False

        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)

        #x = self.apply_central_weighting(x, (math.sqrt(x.shape[1]), math.sqrt(x.shape[1])) )
        
        k_num = int(round(self.k*x.shape[1]))
        topk = torch.topk(x, min(x.shape[1], k_num), dim=1)
        x = torch.mean(topk.values, dim=1).reshape(-1)

        if self.return_index:
            if was_numpy:
                return x.numpy(), topk.indices.numpy()
            return x, topk.indices

        if was_numpy:
            return x.numpy()
        return x

    def apply_central_weighting(self, x, patch_size):
        # 역할: `apply_central_weighting`에 해당하는 작업을 수행.
        # 매개변수: x, patch_size.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 상세 흐름: 주요 호출은 `print`, `torch.linspace`, `torch.meshgrid`, `torch.stack`, `torch.tensor`입니다.
        # 제어 흐름: 조건 분기 1개, 예외 발생 경로 1개입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        """
        Apply central weighting to anomaly scores based on proximity to the center of the patch grid.

        Args:
            x (torch.Tensor): Input anomaly scores of shape (batchsize, patchsize, 1).
            patch_size (tuple): (H, W) number of patches along the height and width.

        Returns:
            torch.Tensor: Weighted anomaly scores of the same shape as input.
        """
        print(">>>>>>>>>>>>>>>> Central weighting is applied.")

        if not isinstance(x, torch.Tensor):
            raise TypeError("Input x must be a torch.Tensor")
        
        batchsize, patchsize, _ = x.shape
        H_patches, W_patches = patch_size
        H_patches, W_patches = int(H_patches), int(W_patches)

        # Generate grid of patch centers
        y_coords = torch.linspace(0, 1, H_patches)
        x_coords = torch.linspace(0, 1, W_patches)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        grid = torch.stack([yy, xx], dim=-1)  # Shape: (H_patches, W_patches, 2)

        # Calculate distance from the center of the patch grid
        center = torch.tensor([0.5, 0.5])
        distances = torch.norm(grid - center, dim=-1)  # Shape: (H_patches, W_patches)

        # min_max normalize distances to [0, 1]
        distances = (distances - distances.min()) / (distances.max() - distances.min())
        distances *= 0.41  # Scale to [0, n] 이 n이 1에 가까울수록 중심에 가까운 patch는 높은 가중치를 받음

        # Convert distances to weights (closer to center = higher weight)
        weights = 1 - distances  # Prevent division by zero

        # Flatten weights to match the order of patches in x
        weights = weights.flatten().unsqueeze(0).unsqueeze(-1)  # Shape: (1, patchsize, 1)

        # Repeat weights for batch size and apply to x
        weights = weights.repeat(batchsize, 1, 1)  # Shape: (batchsize, patchsize, 1)
        weighted_x = x * weights

        return weighted_x

class PatchScorerBlur:
    def __init__(self, k, return_index):
        # 역할: 객체를 사용할 수 있도록 필요한 속성과 구성요소를 초기화.
        # 매개변수: k, return_index.
        # 반환값: 없음. 초기화가 끝난 객체를 Python이 생성합니다..
        # 상세 흐름: 주요 호출은 직접 호출하는 다른 함수 없음입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: `self.k`, `self.return_index`; return 경로는 0개입니다.
        self.k = k
        self.return_index = return_index
    
    def unpatch_scores(self, x, batchsize):
        # 역할: `unpatch_scores`에 해당하는 작업을 수행.
        # 매개변수: x, batchsize.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 상세 흐름: 주요 호출은 `x.reshape`입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x, shape=None):
        # x.shape == (1, 65536, 1) == (batchsize, patchsize, 1)
        # 역할: `score`에 해당하는 작업을 수행.
        # 매개변수: x, shape.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 상세 흐름: 주요 호출은 `torch.from_numpy`, `x.reshape`, `save_tensor_as_image`, `torch.nn.functional.adaptive_avg_pool2d`, `torch.nn.functional.interpolate`입니다.
        # 제어 흐름: 조건 분기 4개입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 4개입니다.
        was_numpy = False

        target_dim = (64, 64)
        from utils import save_tensor_as_image
        # save x as png
        x = torch.from_numpy(x)
        x = x.reshape(1, 1, *shape)
        save_tensor_as_image(x.squeeze(0), 'lg_results/before_pool.png')
        x = torch.nn.functional.adaptive_avg_pool2d(x, target_dim)
        x = torch.nn.functional.interpolate(x, shape, mode='nearest')
        x = ndimage.gaussian_filter(x.reshape(*shape), sigma=2.0)
        x = torch.from_numpy(x.reshape(1, 1, *shape))
        save_tensor_as_image(x.squeeze(0), 'lg_results/after_pool.png')
        x = x.reshape(1, np.prod(shape), 1)

        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        
        k_num = int(round(self.k*x.shape[1]))
        topk = torch.topk(x, min(x.shape[1], k_num), dim=1)
        x = torch.mean(topk.values, dim=1).reshape(-1)

        if self.return_index:
            if was_numpy:
                return x.numpy(), topk.indices.numpy()
            return x, topk.indices

        if was_numpy:
            return x.numpy()
        return x


class AttentionMask:
    """
    1) 2-D realignment of attention vectors
    2) average pooling with fixed resolution
    3) masking to score
    """
    def __init__(self, model, layer, kernel_size=5, stride=1, rollout=False):
        # 역할: 객체를 사용할 수 있도록 필요한 속성과 구성요소를 초기화.
        # 매개변수: model, layer, kernel_size, stride, rollout.
        # 반환값: 없음. 초기화가 끝난 객체를 Python이 생성합니다..
        # 상세 흐름: 주요 호출은 `isinstance`, `int`, `list`, `TypeError`, `map`입니다.
        # 제어 흐름: 조건 분기 2개, 예외 발생 경로 1개입니다.
        # 상태 영향: `self.model`, `self.kernel_size`, `self.stride`, `self.padding`; return 경로는 0개입니다.
        self.model = model
        if isinstance(layer, str):
            self.layer = int(layer)
        elif isinstance(layer, list):
            self.layer = list(map(int, layer))
        else:
            raise TypeError("Layer should be a single scalar or a list of scalars.")

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2

        self.rollout=rollout

    def get_attention(self, images):
        # 역할: 요청한 내부 정보 또는 계산 결과를 가져옴.
        # 매개변수: images.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 상세 흐름: 주요 호출은 `self.model.get_all_selfattention`, `torch.stack`, `torch.mean`, `isinstance`, `int`입니다.
        # 제어 흐름: 반복문 1개, 조건 분기 3개입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        b = images.shape[0]

        attentions = self.model.get_all_selfattention(images)
        attentions = torch.stack(attentions)
        # Averaging across heads
        attentions = torch.mean(attentions, dim=2)

        if self.rollout:
            res_attn = torch.eye(attentions.size(-1)).cuda()

            aug_att_mat = attentions + res_attn

            joint_attentions = torch.zeros(aug_att_mat.size()).cuda()
            joint_attentions[0] = aug_att_mat[0]

            for i in range(1, len(attentions)):
                joint_attentions[i] = torch.matmul(aug_att_mat[i],joint_attentions[i-1])
            
            attentions = joint_attentions

        if isinstance(self.layer, list):
            attentions = torch.mean(attentions[self.layer], dim=0)
        else:
            attentions = attentions[self.layer]
        
        # get an attention score of the global token
        if int(np.sqrt(attentions.shape[-1])) == np.sqrt(attentions.shape[-1]): # when global token not available
            attentions = torch.mean(attentions, dim=1)
        else:        
            attentions = attentions[:, 0, 1:]

        # realign attention score to a 2-D array
        grid_size = int(np.sqrt(attentions.shape[-1]))
        attentions = attentions.reshape(b, grid_size, grid_size)

        pooled = self.pool(attentions).flatten(start_dim=1)

        return pooled

    def pool(self, attn):
        # 역할: `pool`에 해당하는 작업을 수행.
        # 매개변수: attn.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 상세 흐름: 주요 호출은 `torch.nn.AvgPool2d`, `pooling`, `pooled.max`입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        """
        Apply pooling to 2d-attention map
        
        - Args
            attn (torch.Tensor): 2-d attention map

        - Returns
            pooled (torch.Tensor): 2-d normalized & pooled attention map
        """
        pooling = torch.nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        pooled = pooling(attn)

        return pooled / pooled.max()
# 한국어 코드 안내: 이 파일은 이 모듈에 포함된 기능의 구현을 담당합니다.
# 원본 실행 로직은 변경하지 않고, 초보자용 설명 주석만 추가했습니다.
