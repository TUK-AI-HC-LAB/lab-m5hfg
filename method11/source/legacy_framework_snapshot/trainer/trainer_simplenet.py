from loguru import logger
import wandb
import torch
import tqdm
from utils import plot_segmentation_images
import torch
from trainer.trainer import Trainer
from simplenet import Discriminator, Projection
import common

class Trainer_SimpleNet(Trainer):
    def initialize_model(self, dsc_layers, dsc_hidden, pre_proj, proj_layer_type, meta_epochs, aed_meta_epochs, gan_epochs, dsc_margin, dsc_lr, lr, **kwargs):
        # 역할: `initialize_model`에 해당하는 작업을 수행.
        # 매개변수: dsc_layers, dsc_hidden, pre_proj, proj_layer_type, meta_epochs, aed_meta_epochs, gan_epochs, dsc_margin, dsc_lr, lr, **kwargs.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        self.discriminator = Discriminator(
            self.target_embed_dimension, n_layers=dsc_layers, hidden=dsc_hidden)
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: features.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 역할: `_get_pred_scores`에 해당하는 작업을 수행합니다.
        # 매개변수: features.
        # 반환값: 구현에서 계산한 결과 또는 None입니다.
        """
        (Pdb) p dsc_layers
        2
        (Pdb) p dsc_margin
        0.5
        (Pdb) p dsc_hidden
        1024
        (Pdb) p self.target_embed_dimension
        1536
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: features.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 역할: `_preprocessing_features_predict`에 해당하는 작업을 수행합니다.
        # 매개변수: features.
        # 반환값: 구현에서 계산한 결과 또는 None입니다.
        """
        if self.pre_proj > 0:
            self.pre_projection = Projection(
                self.target_embed_dimension, self.target_embed_dimension, pre_proj, proj_layer_type)

        self.discriminator.to(self.device)
        # self.elapsed_time_test(self.discriminator)
        self.dsc_opt = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.dsc_lr, weight_decay=1e-5)
        self.dsc_schl = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.dsc_opt, (meta_epochs - aed_meta_epochs) * gan_epochs, self.dsc_lr*.4)
        self.dsc_margin = dsc_margin
        if self.pre_proj > 0:
            self.pre_projection.to(self.device)
            self.proj_opt = torch.optim.AdamW(
                self.pre_projection.parameters(), lr*.1)
            
        #setattr(self, 'anomaly_segmentor', common.RescaleSegmentorSimpleNet(device=self.device, target_size=(self.args.masksize, self.args.masksize)))

    def _preprocessing_train_disc(self, features):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: features.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        """
        Ours mask 버전 구현을 위한 module 화
        """
        return features

    def _get_pred_scores(self, features):
        # 역할: `_get_pred_scores`에 해당하는 작업을 수행합니다.
        # 매개변수: features.
        # 반환값: 구현에서 계산한 결과 또는 None입니다.
        """
        trainer_ours_gans 는 다른 함수 필요
        """
        # discriminator는 정상/합성 이상 feature를 구분하도록 학습됩니다.
        # 이 구현은 부호를 반대로 하여 큰 값이 이상 쪽을 의미하게 맞춥니다.
        return -self.discriminator(features)

    def _preprocessing_features_predict(self, features):
        # 역할: `_preprocessing_features_predict`에 해당하는 작업을 수행합니다.
        # 매개변수: features.
        # 반환값: 구현에서 계산한 결과 또는 None입니다.
        """
        Ours mask 버전 구현을 위한 module 화
        """
        if self.pre_proj > 0:
            features = self.pre_projection(features)
        return features

    #def _predict(self, images):
    #    """Infer score and mask for a batch of images."""
    #    images = images.to(torch.float).to(self.device)
    #    _ = self.forward_modules.eval()

    #    batchsize = images.shape[0]
    #    if self.pre_proj > 0:
    #        self.pre_projection.eval()
    #    self.discriminator.eval()
    #    self.elapsed_timer.reset()
    #    with torch.no_grad():
    #        features, patch_shapes = self._embed(images,
    #                                             evaluation=True)
    #        self.elapsed_timer.elapsed(
    #            f"{self.dataset_name}_feature extraction", batchsize)
    #        features = self._preprocessing_features_predict(features)

    #        # features = features.cpu().numpy()
    #        # features = np.ascontiguousarray(features.cpu().numpy())
    #        patch_scores = image_scores = self._get_pred_scores(features)
    #        self.elapsed_timer.elapsed(
    #            f"{self.dataset_name}_discriminator", batchsize)
    #        patch_scores = patch_scores.cpu().numpy()
    #        image_scores = image_scores.cpu().numpy()

    #        image_scores = self.patch_maker.unpatch_scores(
    #            image_scores, batchsize=batchsize
    #        )
    #        image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
    #        image_scores = self.patch_maker.score(image_scores)

    #        masks, features = self._get_segmentations(
    #            patch_scores, batchsize, patch_shapes)

    #    # list(masks), list(features), list(patch_scores)
    #    return list(image_scores), [-1], [-1], [-1]

    #def _get_segmentations(self, patch_scores, batchsize, patch_shapes):
    #    return None, None
    #    # patch_scores = self.patch_maker.unpatch_scores(
    #    #    patch_scores, batchsize=batchsize
    #    # )
    #    # scales = patch_shapes[0]
    #    # patch_scores = patch_scores.reshape(
    #    #    batchsize, scales[0], scales[1])
    #    # features = features.reshape(batchsize, scales[0], scales[1], -1)
    #    # masks, features = self.anomaly_segmentor.convert_to_segmentation(
    #    #    patch_scores, features)
    #    # self.elapsed_timer.elapsed(
    #    #    f"{self.dataset_name}_anomaly map generation", batchsize)
    #    # return masks, features
# 한국어 코드 안내: 이 파일은 이상 탐지 방법의 학습·예측 제어을 담당합니다.
# 원본 실행 로직은 변경하지 않고, 초보자용 설명 주석만 추가했습니다.
