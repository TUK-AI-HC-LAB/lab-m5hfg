"""
Mixin classes extracted from Trainer to keep the base class focused on core functionality.

DiscriminatorMixin: GAN discriminator training loop + loss/optimizer helpers
VisualizationMixin: Segmentation image saving + fault image saving
"""

import os
import shutil
import pickle
import re
import PIL
import numpy as np
import torch
import tqdm
import pandas as pd
from loguru import logger
from time import time

import common
import metrics_gpu
from utils import plot_segmentation_images


class DiscriminatorMixin:
    """GAN-based discriminator training loop and related helpers.
    Used by: SimpleNet, Glass, SAM-CutPaste-PatchAug"""

    def _get_len_true_fake(self, input_):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: input_.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        return len(input_) // 2, len(input_) // 2

    def _optim_zero_grad(self):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: 없음.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        if self.pre_proj > 0:
            self.proj_opt.zero_grad()
        if self.train_backbone:
            self.backbone_opt.zero_grad()
        self.dsc_opt.zero_grad()

    def _optim_step(self):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: 없음.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        if self.pre_proj > 0:
            self.proj_opt.step()
        if self.train_backbone:
            self.backbone_opt.step()
        self.dsc_opt.step()

    def _preprocessing_input_data(self, input_data):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: input_data.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        return input_data

    def _preprocessing_image(self, dict_):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: dict_.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        img = dict_["image"]
        img = img.to(torch.float).to(self.device)
        true_feats = self._embed(img, evaluation=False)[0]
        if self.pre_proj > 0:
            true_feats = self.pre_projection(true_feats)
        return true_feats

    def _preprocessing_train_disc(self, features):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: features.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        raise NotImplementedError

    def _postprocessing_train_disc(self, true_feats, fake_feats):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: true_feats, fake_feats.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        return torch.cat([true_feats, fake_feats])

    def apply_augment(self, true_feats):
        # 역할: `apply_augment`에 해당하는 작업을 수행.
        # 매개변수: true_feats.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        noise_idxs = torch.randint(0, self.mix_noise, torch.Size([true_feats.shape[0]]))
        noise_one_hot = torch.nn.functional.one_hot(noise_idxs, num_classes=self.mix_noise).to(self.device)
        noise = torch.stack([
            torch.normal(0, self.noise_std * 1.1 ** k, true_feats.shape)
            for k in range(self.mix_noise)
        ], dim=1).to(self.device)
        noise = (noise * noise_one_hot.unsqueeze(-1)).sum(1)
        fake_feats = true_feats + noise
        return true_feats, fake_feats

    def _loss_function(self, input_, true_feats_size, fake_feats_size):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: input_, true_feats_size, fake_feats_size.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        return self._loss_score(input_, true_feats_size, fake_feats_size)

    def _loss_score(self, input_, true_feats_size, fake_feats_size):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: input_, true_feats_size, fake_feats_size.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        scores = self.discriminator(input_)
        true_scores = scores[:true_feats_size]
        fake_scores = scores[fake_feats_size:]
        th = self.dsc_margin
        p_true = (true_scores.detach() >= th).sum() / len(true_scores)
        p_fake = (fake_scores.detach() < -th).sum() / len(fake_scores)
        true_loss = torch.clip(-true_scores + th, min=0)
        fake_loss = torch.clip(fake_scores + th, min=0)
        loss = true_loss.mean() + fake_loss.mean()
        return loss, p_true, p_fake

    def _loss_backward(self, loss):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: loss.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        if isinstance(loss, list):
            loss[0].backward()
        else:
            loss.backward()

    def _loss_aggregate(self, all_loss, loss):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: all_loss, loss.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        if isinstance(loss, list):
            all_loss.append([l.detach().cpu().item() for l in loss])
        else:
            all_loss.append(loss.detach().cpu().item())

    def _loss_str(self, all_loss, ret=False):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: all_loss, ret.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        if isinstance(all_loss[0], list):
            total = [0] * len(all_loss[0])
            for loss in all_loss:
                for i, l in enumerate(loss):
                    total[i] += l
            total = [round(t / len(all_loss), 5) for t in total]
            if ret:
                total = total[0]
        else:
            total = round(sum(all_loss) / len(all_loss), 5)
        return total

    def _train_discriminator(self, input_data):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: input_data.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        _ = self.forward_modules.eval()
        if self.pre_proj > 0:
            self.pre_projection.train()
        self.discriminator.train()
        i_iter = 0
        logger.info("Training discriminator...")

        input_data = self._preprocessing_input_data(input_data)
        with tqdm.tqdm(total=self.gan_epochs) as pbar:
            for i_epoch in range(self.gan_epochs):
                all_loss = []
                all_p_true = []
                all_p_fake = []
                all_p_interp = []
                embeddings_list = []
                for data_item in input_data:
                    self._optim_zero_grad()
                    i_iter += 1
                    out = self._preprocessing_image(data_item)
                    out = self._preprocessing_train_disc(out)
                    true_feats, fake_feats = self.apply_augment(out)
                    input_ = self._postprocessing_train_disc(true_feats, fake_feats)
                    len_true, len_fake = self._get_len_true_fake(input_)
                    loss, p_true, p_fake = self._loss_function(input_, len_true, len_fake)
                    self._loss_backward(loss)
                    self._optim_step()
                    self._loss_aggregate(all_loss, loss)
                    all_p_true.append(p_true.cpu().item())
                    all_p_fake.append(p_fake.cpu().item())

                if len(embeddings_list) > 0:
                    self.auto_noise[1] = torch.cat(embeddings_list).std(0).mean(-1)
                if self.cos_lr:
                    self.dsc_schl.step()

                all_p_true = sum(all_p_true) / len(input_data)
                all_p_fake = sum(all_p_fake) / len(input_data)
                cur_lr = self.dsc_opt.state_dict()['param_groups'][0]['lr']
                pbar_str = f"epoch:{i_epoch} loss:{self._loss_str(all_loss)} "
                pbar_str += f"lr:{round(cur_lr, 6)}"
                pbar_str += f" p_true:{round(all_p_true, 3)} p_fake:{round(all_p_fake, 3)}"
                if len(all_p_interp) > 0:
                    pbar_str += f" p_interp:{round(sum(all_p_interp) / len(input_data), 3)}"
                pbar.set_description_str(pbar_str)
                pbar.update(1)

                if self.args.epoch_test_mode:
                    epoch = self.i_mepoch * self.gan_epochs + i_epoch
                    scores, segmentations, features, labels_gt, masks_gt = self.predict(self.test_data)
                    a, f, p, s = self._evaluate(scores, segmentations, features, labels_gt, masks_gt)
                    self.df.loc[epoch] = [a, f, s]
                    self.df.to_csv(f'lg_results/epoch_test_{self.args.mainmodel}_coreset.csv', index=False)

        return self._loss_str(all_loss, ret=True)


class VisualizationMixin:
    """Segmentation and fault image saving.
    Used by trainers that need --save_segmentation_images or --save_fault_images."""

    def _save_segmentation_per_image_path(self, image_path, mask_path, segmentation, save_path, image_index):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: image_path, mask_path, segmentation, save_path, image_index.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        if mask_path:
            mask = Image.open(mask_path).convert("RGB")
        else:
            mask = Image.new('RGB', image.size)

        if isinstance(segmentation, np.ndarray):
            segmentation_normalized = (segmentation - segmentation.min()) / (segmentation.ptp() + 1e-8)
            segmentation_image = Image.fromarray((segmentation_normalized * 255).astype(np.uint8)).convert("RGB")
        else:
            segmentation_image = segmentation

        target_size = (224, 224)
        image = image.resize(target_size)
        mask = mask.resize(target_size)
        segmentation_image = segmentation_image.resize(target_size)

        combined_width = target_size[0] * 3
        combined_image = Image.new('RGB', (combined_width, target_size[1]))
        combined_image.paste(image, (0, 0))
        combined_image.paste(mask, (target_size[0], 0))
        combined_image.paste(segmentation_image, (2 * target_size[0], 0))

        os.makedirs(save_path, exist_ok=True)
        combined_image.save(os.path.join(save_path, f'segmentation_{image_index}.png'), format='PNG')

    def _save_segmentation_images(self, data, segmentations, scores, image_labels, save_path):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: data, segmentations, scores, image_labels, save_path.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        image_paths = [x[2] for x in data.dataset.data_to_iterate]
        mask_paths = [x[3] for x in data.dataset.data_to_iterate]

        segmentations = np.stack(segmentations)
        min_score = segmentations.min()
        max_score = segmentations.max()
        segmentations = (segmentations - min_score) / (max_score - min_score)

        def image_transform(image):
            # 역할: `image_transform`에 해당하는 작업을 수행.
            # 매개변수: image.
            # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
            in_std = np.array(data.dataset.transform_std).reshape(-1, 1, 1)
            in_mean = np.array(data.dataset.transform_mean).reshape(-1, 1, 1)
            image = data.dataset.transform_img(image)
            return np.clip((image.numpy() * in_std + in_mean) * 255, 0, 255).astype(np.uint8)

        def mask_transform(mask):
            # 역할: `mask_transform`에 해당하는 작업을 수행.
            # 매개변수: mask.
            # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
            return data.dataset.transform_mask(mask).numpy()

        plot_segmentation_images(
            save_path, image_paths, segmentations, scores,
            mask_paths, image_labels=image_labels,
            image_transform=image_transform, mask_transform=mask_transform,
        )

    def _save_fault_images(self, data, scores):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: data, scores.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        if not self.args.save_fault_images:
            return

        assert 'good' in [d[1] for d in data.dataset.data_to_iterate]

        dataset_name = re.sub(r"[\s,\[\]\']", "", str(data.dataset.classnames_to_use))
        save_path = f'./saved_fault_images/{self.args.mainmodel}/{dataset_name}'
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)

        scores = torch.tensor(scores)
        labels = torch.tensor([0 if d[1] == 'good' else 1 for d in data.dataset.data_to_iterate])
        ok_scores = scores[labels == 0]
        ng_scores = scores[labels == 1]
        ok_paths = [d[2] for d in data.dataset.data_to_iterate if d[1] == 'good']
        ng_paths = [d[2] for d in data.dataset.data_to_iterate if d[1] != 'good']

        for id in torch.argsort(ok_scores, descending=True)[:5]:
            PIL.Image.open(ok_paths[id]).save(
                os.path.join(save_path, f'okfault_p{ok_scores[id]:.2f}_id{id}_ngfrom{ng_scores.min():.2f}to{ng_scores.max():.2f}.png'))
        for id in torch.argsort(ok_scores)[:5]:
            PIL.Image.open(ok_paths[id]).save(
                os.path.join(save_path, f'okgood_p{ok_scores[id]:.2f}_id{id}_ngfrom{ng_scores.min():.2f}to{ng_scores.max():.2f}.png'))
        for id in torch.argsort(ng_scores)[:5]:
            PIL.Image.open(ng_paths[id]).save(
                os.path.join(save_path, f'ngfault_p{ng_scores[id]:.2f}_id{id}_okfrom{ok_scores.min():.2f}to{ok_scores.max():.2f}.png'))
        for id in torch.argsort(ng_scores, descending=True)[:5]:
            PIL.Image.open(ng_paths[id]).save(
                os.path.join(save_path, f'nggood_p{ng_scores[id]:.2f}_id{id}_okfrom{ok_scores.min():.2f}to{ok_scores.max():.2f}.png'))
# 한국어 코드 안내: 이 파일은 여러 모듈이 함께 쓰는 보조 기능을 담당합니다.
# 원본 실행 로직은 변경하지 않고, 초보자용 설명 주석만 추가했습니다.
