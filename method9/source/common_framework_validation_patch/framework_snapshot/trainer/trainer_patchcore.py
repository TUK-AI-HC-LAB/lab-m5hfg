import os
import abc
import pickle
from loguru import logger
from typing import List
from typing import Union

import faiss
import tqdm
import numpy as np
import torch

from trainer.trainer import Trainer
import utils
import common


class Trainer_PatchCore(Trainer):
    def initialize_model(self, **kwargs):
        # 역할: `initialize_model`에 해당하는 작업을 수행.
        # 매개변수: **kwargs.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        self.coreset_ratio = self.args.coreset_ratio
        self.anomaly_scorer = NearestNeighbourScorer(
            n_nearest_neighbours=1, nn_method=FaissNN(False, 8))
        self.featuresampler = ApproximateGreedyCoresetSampler( self.coreset_ratio, self.device, num_coreset_samples=self.args.patchcore_coreset_num)
        #setattr(self, 'anomaly_segmentor', common.RescaleSegmentorSimpleNet(device=self.device, target_size=(self.args.masksize, self.args.masksize)))

    def set_ea_modules(self):
        # 역할: 객체의 설정 또는 내부 상태를 지정.
        # 매개변수: 없음.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        pass

    def _pretrain_model(self, training_data, val_data, test_data, dataset_name):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: training_data, val_data, test_data, dataset_name.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 역할: `_preprocessing_memory_bank`에 해당하는 작업을 수행합니다.
        # 매개변수: _image, _features.
        # 반환값: 구현에서 계산한 결과 또는 None입니다.
        """
        Ours mask 버전 구현을 위한 module 화
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: _features.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        """
        return

    def _preprocessing_features_predict(self, _features):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: _image, _features.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        """
        Ours mask 버전 구현을 위한 module 화
        """
        return _features

    def _preprocessing_memory_bank(self, _image, _features):
        # 역할: `_preprocessing_memory_bank`에 해당하는 작업을 수행합니다.
        # 매개변수: _image, _features.
        # 반환값: 구현에서 계산한 결과 또는 None입니다.
        """
        Uniformaly, ours 구현을 위한 module 화
        """
        return _features

    def _meta_train(self, training_data, val_data, test_data, dataset_name):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: training_data, val_data, test_data, dataset_name.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        """Overrides the base training loop with PatchCore's single-pass logic."""
        self._pretrain_model(training_data, val_data, test_data, dataset_name)
        self._fill_memory_bank(training_data)
        
        # Record the evaluation result once the memory bank is filled
        self.record_evaluation_epoch(test_data)
        
        # Additional visualizations
        self._plot_tsne_features(training_data, val_data, test_data)
        
        # This method does not need to return anything.
        # The base train() method will handle averaging the single recorded result.
    
    def _plot_tsne_features(self, input_data, val_data, test_data):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: input_data, val_data, test_data.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        if not self.args.plot_tsne:
            return

        tr_tsne_features = []
        tr_tsne_labels = []
        te_tsne_features = []
        te_tsne_labels = []

        features = self._fill_memory_bank(input_data, fit=False)
        tr_tsne_features.append(features)
        tr_tsne_labels.append([0]*len(features))

        for i, data in enumerate(test_data):
            if i > 2: # 2개 test image 에 대해서만 plot
                break
            images = data["image"].to(torch.float).to(self.device)
            masks = data["mask"].to(torch.float).to(self.device)
            # save masks into jpeg file
            #utils.save_mask_as_png(masks.squeeze(0).squeeze(0).cpu().numpy(), f'lg_results/mask_{i}.png')
            
            assert len(images) == 1, "Only one image can be processed at a time."
            utils.save_image_and_mask_as_png(data['image_path'][0], masks.squeeze(0).squeeze(0).cpu().numpy(), f'lg_results/mask_{i}.png')

            with torch.no_grad():
                features, patch_shapes = self._embed(images)
                features = self._preprocessing_features_predict(features)
                features = features.cpu().numpy()

            patch_masks = utils.mask_to_patch(masks, patch_shapes[0])
            te_tsne_features.append(features)
            te_tsne_labels.append(patch_masks.reshape(-1).cpu().tolist())
        
        tr_tsne_features = np.concatenate(tr_tsne_features, axis=0)
        tr_tsne_labels = np.concatenate(tr_tsne_labels, axis=0)
        te_tsne_features = np.concatenate(te_tsne_features, axis=0)
        te_tsne_labels = np.concatenate(te_tsne_labels, axis=0)

        features = np.concatenate([tr_tsne_features, te_tsne_features], axis=0)
        labels = np.concatenate([tr_tsne_labels, te_tsne_labels], axis=0)
        memory_bank_indices = list(range(len(tr_tsne_labels)))
        test_abnormal_indices = labels.nonzero()[0] # normal 은 어차피 0 이라 무시됨

        # plot tsne
        name = f'lg_results/tsne_prediction_{self.args.mainmodel}_{self.args.few_cluster_alg}.png'
        utils.plot_tsne(features, name, color_indices=[memory_bank_indices, test_abnormal_indices], shape_indices=[test_abnormal_indices], sizes_indices={10: memory_bank_indices, 5: test_abnormal_indices}, legend_labels={(0, -1): 'memory_bank', (-1,-1): 'test_normal', (1,0): 'test_abnormal'}) # NOTE memory bank 는 색을 입히고, test 는 black 인데, test 중 anomaly 는 모양을 다르게 함

    def _use_featuresampler(self, features):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: features.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        return self.featuresampler.run(features)

    def _fill_memory_bank(self, input_data, fit=True):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: input_data, fit.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
            # 매개변수: input_image.
            # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)[0]

        # 정상 train image의 모든 patch embedding을 모아 memory bank 후보를 만듭니다.
        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                feat = _image_to_features(image)
                feat = self._preprocessing_memory_bank(image, feat)
                features.append(feat.cpu())

        features = np.concatenate(features, axis=0)
        # 모든 patch를 저장하면 매우 크므로, 정상 분포를 대표하는 coreset만 남깁니다.
        features = self._use_featuresampler(features)

        # fit=True일 때만 FAISS 최근접 이웃 index를 구축합니다.
        if fit:
            self.anomaly_scorer.fit(detection_features=[features])
        return features

    #def _predict(self, images):
    #    """Infer score and mask for a batch of images."""
    #    images = images.to(torch.float).to(self.device)
    #    _ = self.forward_modules.eval()

    #    batchsize = images.shape[0]
    #    with torch.no_grad():
    #        features, patch_shapes = self._embed(images)
    #        features = self._preprocessing_features_predict(features)
    #        features = np.asarray(features.cpu())
    #        patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
    #        image_scores = self.patch_maker.unpatch_scores( image_scores, batchsize=batchsize)
    #        image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
    #        image_scores = self._preprocessing_predict(images, image_scores)
    #        image_scores = self._score(image_scores)
    #        patch_scores = self.patch_maker.unpatch_scores( patch_scores, batchsize=batchsize) # patch_scores.shape==(784,) --> (1, 784)
    #        scales = patch_shapes[0]
    #        patch_scores = patch_scores.reshape( batchsize, scales[0], scales[1]) # patch_scores.shape==(1, 784) --> (1, 28, 28)
    #        features = features.reshape(batchsize, scales[0], scales[1], -1)
    #        masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores, features)

    #    # return [score for score in image_scores], [mask for mask in masks]
    #    return [score for score in image_scores], [-1], [-1], [score for score in patch_scores]

    def _get_pred_scores(self, features):
        # 역할: `_get_pred_scores`에 해당하는 작업을 수행합니다.
        # 매개변수: features.
        # 반환값: 구현에서 계산한 결과 또는 None입니다.
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: features.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # test patch와 memory bank의 최근접 정상 patch 사이 거리를 반환합니다.
        # 거리가 클수록 학습 때 본 정상 패턴에서 멀다는 뜻입니다.
        return self.anomaly_scorer.predict([features.cpu().numpy()])[0]


class NearestNeighbourScorer(object):
    def __init__(self, n_nearest_neighbours: int, nn_method) -> None:
        # 역할: 객체를 사용할 수 있도록 필요한 속성과 구성요소를 초기화.
        # 매개변수: n_nearest_neighbours, nn_method.
        # 반환값: 없음. 초기화가 끝난 객체를 Python이 생성합니다..
        """
        Neearest-Neighbourhood Anomaly Scorer class.

        Args:
            n_nearest_neighbours: [int] Number of nearest neighbours used to
                determine anomalous pixels.
            nn_method: Nearest neighbour search method.
        """
        self.feature_merger = ConcatMerger()

        self.n_nearest_neighbours = n_nearest_neighbours
        self.nn_method = nn_method

        self.imagelevel_nn = lambda query: self.nn_method.run(
            n_nearest_neighbours, query
        )
        self.pixelwise_nn = lambda query, index: self.nn_method.run(
            1, query, index)

    def fit(self, detection_features: List[np.ndarray]) -> None:
        # 역할: 주어진 데이터로 통계량 또는 모델을 학습.
        # 매개변수: detection_features.
        # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
        """Calls the fit function of the nearest neighbour method.

        Args:
            detection_features: [list of np.arrays]
                [[bs x d_i] for i in n] Contains a list of
                np.arrays for all training images corresponding to respective
                features VECTORS (or maps, but will be resized) produced by
                some backbone network which should be used for image-level
                anomaly detection.
        """
        self.detection_features = self.feature_merger.merge(
            detection_features,
        )
        self.nn_method.fit(self.detection_features)

    def predict(
        self, query_features: List[np.ndarray]
    ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        # 역할: 입력의 예측 또는 이상 탐지 점수와 map을 계산.
        # 매개변수: query_features.
        # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
        """Predicts anomaly score.

        Searches for nearest neighbours of test images in all
        support training images.

        Args:
             detection_query_features: [dict of np.arrays] List of np.arrays
                 corresponding to the test features generated by
                 some backbone network.
        """
        query_features = self.feature_merger.merge(
            query_features,
        )
        query_distances, query_nns = self.imagelevel_nn(query_features)
        anomaly_scores = np.mean(query_distances, axis=-1)
        return anomaly_scores, query_distances, query_nns

    @staticmethod
    def _detection_file(folder, prepend=""):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: folder, prepend.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        return os.path.join(folder, prepend + "nnscorer_features.pkl")

    @staticmethod
    def _index_file(folder, prepend=""):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: folder, prepend.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        return os.path.join(folder, prepend + "nnscorer_search_index.faiss")

    @staticmethod
    def _save(filename, features):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: filename, features.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        if features is None:
            return
        with open(filename, "wb") as save_file:
            pickle.dump(features, save_file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _load(filename: str):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: filename.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        with open(filename, "rb") as load_file:
            return pickle.load(load_file)

    def save(
        self,
        save_folder: str,
        save_features_separately: bool = False,
        prepend: str = "",
    ) -> None:
        # 역할: 현재 상태 또는 결과를 파일에 저장.
        # 매개변수: save_folder, save_features_separately, prepend.
        # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
        self.nn_method.save(self._index_file(save_folder, prepend))
        if save_features_separately:
            self._save(
                self._detection_file(
                    save_folder, prepend), self.detection_features
            )

    def save_and_reset(self, save_folder: str) -> None:
        # 역할: `save_and_reset`에 해당하는 작업을 수행.
        # 매개변수: save_folder.
        # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
        self.save(save_folder)
        self.nn_method.reset_index()

    def load(self, load_folder: str, prepend: str = "") -> None:
        # 역할: 필요한 모델·가중치·설정을 준비.
        # 매개변수: load_folder, prepend.
        # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
        self.nn_method.load(self._index_file(load_folder, prepend))
        if os.path.exists(self._detection_file(load_folder, prepend)):
            self.detection_features = self._load(
                self._detection_file(load_folder, prepend)
            )


class FaissNN(object):
    def __init__(self, on_gpu: bool = False, num_workers: int = 4) -> None:
        # 역할: 객체를 사용할 수 있도록 필요한 속성과 구성요소를 초기화.
        # 매개변수: on_gpu, num_workers.
        # 반환값: 없음. 초기화가 끝난 객체를 Python이 생성합니다..
        """FAISS Nearest neighbourhood search.

        Args:
            on_gpu: If set true, nearest neighbour searches are done on GPU.
            num_workers: Number of workers to use with FAISS for similarity search.
        """
        faiss.omp_set_num_threads(num_workers)
        self.on_gpu = on_gpu
        self.search_index = None

    def _gpu_cloner_options(self):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: 없음.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        return faiss.GpuClonerOptions()

    def _index_to_gpu(self, index):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: index.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        if self.on_gpu:
            # For the non-gpu faiss python package, there is no GpuClonerOptions
            # so we can not make a default in the function header.
            return faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 0, index, self._gpu_cloner_options()
            )
        return index

    def _index_to_cpu(self, index):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: index.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        if self.on_gpu:
            return faiss.index_gpu_to_cpu(index)
        return index

    def _create_index(self, dimension):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: dimension.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        logger.info("현재 faiss 는 IndexFlatL2 를 쓰는 중")
        if self.on_gpu:
            return faiss.GpuIndexFlatL2(
                faiss.StandardGpuResources(), dimension, faiss.GpuIndexFlatConfig()
            )
        return faiss.IndexFlatL2(dimension)

    def fit(self, features: np.ndarray) -> None:
        # 역할: 주어진 데이터로 통계량 또는 모델을 학습.
        # 매개변수: features.
        # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
        """
        Adds features to the FAISS search index.

        Args:
            features: Array of size NxD.
        """
        if self.search_index:
            self.reset_index()
        self.search_index = self._create_index(features.shape[-1])
        self._train(self.search_index, features)
        self.search_index.add(features)

    def _train(self, _index, _features):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: _index, _features.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        pass

    def run(
        self,
        n_nearest_neighbours,
        query_features: np.ndarray,
        index_features: np.ndarray = None,
    ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        # 역할: `run`에 해당하는 작업을 수행.
        # 매개변수: n_nearest_neighbours, query_features, index_features.
        # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
        """
        Returns distances and indices of nearest neighbour search.

        Args:
            query_features: Features to retrieve.
            index_features: [optional] Index features to search in.
        """
        if index_features is None:
            return self.search_index.search(query_features, n_nearest_neighbours)

        # Build a search index just for this search.
        search_index = self._create_index(index_features.shape[-1])
        self._train(search_index, index_features)
        search_index.add(index_features)
        return search_index.search(query_features, n_nearest_neighbours)

    def save(self, filename: str) -> None:
        # 역할: 현재 상태 또는 결과를 파일에 저장.
        # 매개변수: filename.
        # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
        faiss.write_index(self._index_to_cpu(self.search_index), filename)

    def load(self, filename: str) -> None:
        # 역할: 필요한 모델·가중치·설정을 준비.
        # 매개변수: filename.
        # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
        self.search_index = self._index_to_gpu(faiss.read_index(filename))

    def reset_index(self):
        # 역할: `reset_index`에 해당하는 작업을 수행.
        # 매개변수: 없음.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        if self.search_index:
            self.search_index.reset()
            self.search_index = None


class _BaseMerger:
    def __init__(self):
        # 역할: 객체를 사용할 수 있도록 필요한 속성과 구성요소를 초기화.
        # 매개변수: 없음.
        # 반환값: 없음. 초기화가 끝난 객체를 Python이 생성합니다..
        """Merges feature embedding by name."""

    def merge(self, features: list):
        # 역할: `merge`에 해당하는 작업을 수행.
        # 매개변수: features.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        features = [self._reduce(feature) for feature in features]
        return np.concatenate(features, axis=1)


class ConcatMerger(_BaseMerger):
    @staticmethod
    def _reduce(features):
        # NxCxWxH -> NxCWH
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: features.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        return features.reshape(len(features), -1)


class BaseSampler(abc.ABC):
    def __init__(self, percentage: float):
        # 역할: 객체를 사용할 수 있도록 필요한 속성과 구성요소를 초기화.
        # 매개변수: percentage.
        # 반환값: 없음. 초기화가 끝난 객체를 Python이 생성합니다..
        if not 0 < percentage <= 1:
            raise ValueError("Percentage value not in (0, 1).")
        self.percentage = percentage

    @abc.abstractmethod
    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        # 역할: `run`에 해당하는 작업을 수행.
        # 매개변수: features.
        # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
        pass

    def _store_type(self, features: Union[torch.Tensor, np.ndarray]) -> None:
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: features.
        # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
        self.features_is_numpy = isinstance(features, np.ndarray)
        if not self.features_is_numpy:
            self.features_device = features.device

    def _restore_type(self, features: torch.Tensor) -> Union[torch.Tensor, np.ndarray]:
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: features.
        # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
        if self.features_is_numpy:
            return features.cpu().numpy()
        return features.to(self.features_device)


class GreedyCoresetSampler(BaseSampler):
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        dimension_to_project_features_to=128,
    ):
        # 역할: 객체를 사용할 수 있도록 필요한 속성과 구성요소를 초기화.
        # 매개변수: percentage, device, dimension_to_project_features_to.
        # 반환값: 없음. 초기화가 끝난 객체를 Python이 생성합니다..
        """Greedy Coreset sampling base class."""
        super().__init__(percentage)

        self.device = device
        self.dimension_to_project_features_to = dimension_to_project_features_to

    def _reduce_features(self, features):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: features.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        if features.shape[1] == self.dimension_to_project_features_to:
            return features
        mapper = torch.nn.Linear(
            features.shape[1], self.dimension_to_project_features_to, bias=False
        )
        try:
            _ = mapper.to(self.device)
            features = features.to(self.device)
        except:
            mapper = mapper.cpu()
            features = features.cpu()
        return mapper(features)

    def run(
        self, features: Union[torch.Tensor, np.ndarray], return_indices=False
    ) -> Union[torch.Tensor, np.ndarray]:
        # 역할: `run`에 해당하는 작업을 수행.
        # 매개변수: features, return_indices.
        # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
        """Subsamples features using Greedy Coreset.

        Args:
            features: [N x D]
        """
        self._store_type(features)
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        try:
            reduced_features = self._reduce_features(features)
            sample_indices = self._compute_greedy_coreset_indices(reduced_features)
        except:
            reduced_features = self._reduce_features(features.cpu())
            sample_indices = self._compute_greedy_coreset_indices(reduced_features.cpu())

        features = features[sample_indices]
        if return_indices:
            return self._restore_type(features), sample_indices
        return self._restore_type(features)

    @staticmethod
    def _compute_batchwise_differences(
        matrix_a: torch.Tensor, matrix_b: torch.Tensor
    ) -> torch.Tensor:
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: matrix_a, matrix_b.
        # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
        """Computes batchwise Euclidean distances using PyTorch."""
        a_times_a = matrix_a.unsqueeze(1).bmm(
            matrix_a.unsqueeze(2)).reshape(-1, 1)
        b_times_b = matrix_b.unsqueeze(1).bmm(
            matrix_b.unsqueeze(2)).reshape(1, -1)
        a_times_b = matrix_a.mm(matrix_b.T)

        return (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt()

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: features.
        # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
        """Runs iterative greedy coreset selection.

        Args:
            features: [NxD] input feature bank to sample.
        """
        distance_matrix = self._compute_batchwise_differences(
            features, features)
        coreset_anchor_distances = torch.norm(distance_matrix, dim=1)

        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)

        for _ in range(num_coreset_samples):
            select_idx = torch.argmax(coreset_anchor_distances).item()
            coreset_indices.append(select_idx)

            coreset_select_distance = distance_matrix[
                :, select_idx: select_idx + 1  # noqa E203
            ]
            coreset_anchor_distances = torch.cat(
                [coreset_anchor_distances.unsqueeze(-1), coreset_select_distance], dim=1
            )
            coreset_anchor_distances = torch.min(
                coreset_anchor_distances, dim=1).values

        return np.array(coreset_indices)


class ApproximateGreedyCoresetSampler(GreedyCoresetSampler):
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        number_of_starting_points: int = 10,
        dimension_to_project_features_to: int = 128,
        num_coreset_samples: int = None,
    ):
        # 역할: 객체를 사용할 수 있도록 필요한 속성과 구성요소를 초기화.
        # 매개변수: percentage, device, number_of_starting_points, dimension_to_project_features_to, num_coreset_samples.
        # 반환값: 없음. 초기화가 끝난 객체를 Python이 생성합니다..
        """Approximate Greedy Coreset sampling base class."""
        self.number_of_starting_points = number_of_starting_points
        self.num_coreset_samples = num_coreset_samples
        super().__init__(percentage, device, dimension_to_project_features_to)

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: features.
        # 반환값: 함수 선언에 적힌 타입의 계산 결과입니다..
        """Runs approximate iterative greedy coreset selection.

        This greedy coreset implementation does not require computation of the
        full N x N distance matrix and thus requires a lot less memory, however
        at the cost of increased sampling times.

        Args:
            features: [NxD] input feature bank to sample.
        """
        number_of_starting_points = np.clip(
            self.number_of_starting_points, None, len(features)
        )  # --> 10
        start_points = np.random.choice(
            len(features), number_of_starting_points, replace=False
        ).tolist()  # --> 10 개 indices

        approximate_distance_matrix = self._compute_batchwise_differences(
            features, features[start_points]
        )  # --> #features x 10 matrix 연산. e.g., torch.Size([458640, 10])

        approximate_coreset_anchor_distances = torch.mean(
            approximate_distance_matrix, axis=-1
        ).reshape(-1, 1)  # --> torch.Size([458640, 1])
        coreset_indices = []

        num_coreset_samples = int(len(features) * self.percentage)
        if self.num_coreset_samples is None:
            pass
        else:
            num_coreset_samples = min(num_coreset_samples, int(self.num_coreset_samples))

        with torch.no_grad():
            for _ in tqdm.tqdm(range(num_coreset_samples), desc="Subsampling..."):
                select_idx = torch.argmax(
                    approximate_coreset_anchor_distances).item()  # 가장 큰 값의 index 1개
                coreset_indices.append(select_idx)
                coreset_select_distance = self._compute_batchwise_differences(
                    features, features[select_idx: select_idx + 1]  # noqa: E203
                )  # 방금 추출한 coresot index 와의 거리 계산
                approximate_coreset_anchor_distances = torch.cat(
                    [approximate_coreset_anchor_distances, coreset_select_distance],
                    dim=-1,
                )  # --> torch.Size([458640, 2])
                approximate_coreset_anchor_distances = torch.min(
                    approximate_coreset_anchor_distances, dim=1
                ).values.reshape(-1, 1)  # --> torch.Size([458640, 1]) 둘 중에 작은 값으로 업데이트

        return np.array(coreset_indices)
# 한국어 코드 안내: 이 파일은 이상 탐지 방법의 학습·예측 제어을 담당합니다.
# 원본 실행 로직은 변경하지 않고, 초보자용 설명 주석만 추가했습니다.
