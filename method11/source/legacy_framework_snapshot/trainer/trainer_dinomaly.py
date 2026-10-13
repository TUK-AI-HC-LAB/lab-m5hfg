from loguru import logger
import pandas as pd
import wandb
import tqdm
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

from trainer.trainer import Trainer
from trainer.Dinomaly_lib.models import vit_encoder
from trainer.Dinomaly_lib.models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from trainer.Dinomaly_lib.models.uad import ViTill
from trainer.Dinomaly_lib.decoder import DECODER_REGISTRY
from trainer.Dinomaly_lib.dinov1.utils import trunc_normal_
from trainer.Dinomaly_lib.optimizers import StableAdamW
from trainer.Dinomaly_lib.utils import global_cosine_hm_percent, WarmCosineScheduler, get_gaussian_kernel, cal_anomaly_maps

class Trainer_Dinomaly(Trainer):
    def initialize_model(self, **kwargs):
        # encoder_name = 'dinov2reg_vit_small_14'
        # 역할: `initialize_model`에 해당하는 작업을 수행.
        # 매개변수: **kwargs.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        encoder_name = 'dinov2reg_vit_base_14'
        # encoder_name = 'dinov2reg_vit_large_14'
        
        if 'small' in encoder_name:
            embed_dim, num_heads = 384, 6
        elif 'base' in encoder_name:
            embed_dim, num_heads = 768, 12
        elif 'large' in encoder_name:
            embed_dim, num_heads = 1024, 16
            target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
        else:
            raise "Architecture not in small, base, large."
        
        encoder = vit_encoder.load(encoder_name)
        
        target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
        fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        
        bottleneck = nn.ModuleList([bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2)])

        decoder_type = getattr(self.args, 'decoder_type', 'default')
        DecoderCls = DECODER_REGISTRY[decoder_type]
        decoder = DecoderCls(
            embed_dim=embed_dim, num_heads=num_heads, num_layers=8,
            pruning_ratio=getattr(self.args, 'pruning_ratio', 0.5),
            pruning_after_layer=getattr(self.args, 'pruning_after_layer', 2),
        )

        self.Dinomaly = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers,
                    mask_neighbor_size=0, fuse_layer_encoder=fuse_layer_encoder, fuse_layer_decoder=fuse_layer_decoder)
        self.Dinomaly = self.Dinomaly.to(self.device)
        self.trainable = nn.ModuleList([bottleneck, decoder])
        
        for m in self.trainable.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
                
        self.optimizer = StableAdamW([{'params': self.trainable.parameters()}],
                            lr=self.args.lr, betas=self.args.betas, weight_decay=self.args.weight_decay, amsgrad=True, eps=1e-8)
        self.lr_scheduler = WarmCosineScheduler(self.optimizer, base_value=2e-3, final_value=2e-4, total_iters=self.args.total_iter,
                                       warmup_iters=100)
        
        self.trained = False

    def set_ea_modules(self):
        # 역할: 객체의 설정 또는 내부 상태를 지정.
        # 매개변수: 없음.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        pass

    def train(self, training_data, val_data, test_data, dataset_name):
        # Initialize results list
        # 역할: 학습 데이터를 사용해 모델 파라미터를 갱신.
        # 매개변수: training_data, val_data, test_data, dataset_name.
        # 반환값: 보통 없음(None)이며, 객체 상태가 변경됩니다..
        self.evaluation_results = []
        
        if self.args.uni and self.trained:
            # Evaluate and return
            self.record_evaluation_epoch(test_data)
            if self.evaluation_results:
                avg_metrics = {key: np.mean([m[key] for m in self.evaluation_results]) for key in self.evaluation_results[0]}
            else:
                avg_metrics = self.record_evaluation_epoch(test_data)
            return avg_metrics["auroc"], avg_metrics["full_pixel_auroc"], avg_metrics["pro"], None, None, None, avg_metrics["saliency_cr_f1"]
        
        it = 0
        total_epoch = int(np.ceil(self.args.total_iter / len(training_data)))
        
        if self.args.epoch_test_mode:
            df = pd.DataFrame(columns=['IAUC', 'PAUC', 'SF1'])
        for i_mepoch in range(total_epoch):
            logger.info(f"\n\n----- {i_mepoch} -----")
            self.Dinomaly.train()

            for data_item in tqdm.tqdm(training_data, desc="Training Progress " + str(i_mepoch) + "/" + str(total_epoch), unit="it"):
                image = data_item['image'].to(self.device)
               
                en, de = self.Dinomaly(image)

                p_final = 0.9
                p = min(p_final * it / 1000, p_final)
                loss = global_cosine_hm_percent(en, de, p=p, factor=0.1)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.trainable.parameters(), max_norm=0.1)

                self.optimizer.step()
                self.lr_scheduler.step()
                
                it += 1
                if it == self.args.total_iter:
                    break
        
            if self.args.epoch_test_mode:
                # For monitoring: always perform evaluation
                scores, segmentations, features, labels_gt, masks_gt = self.predict(test_data)
                auroc, full_pixel_auroc, pro, saliency_cr_f1 = self._evaluate(
                    scores, segmentations, features, labels_gt, masks_gt
                )
                epoch_metrics = {
                    "auroc": auroc,
                    "full_pixel_auroc": full_pixel_auroc,
                    "pro": pro,
                    "saliency_cr_f1": saliency_cr_f1,
                }
                
                # Only add to evaluation_results if this is one of the last 5 epochs
                if i_mepoch >= total_epoch - 5:
                    self.evaluation_results.append(epoch_metrics)
                
                a, f, s = epoch_metrics["auroc"], epoch_metrics["full_pixel_auroc"], epoch_metrics["saliency_cr_f1"]
                df.loc[i_mepoch] = [a, f, s]
                df.to_csv(f'lg_results/epoch_test_dinomaly_{dataset_name}.csv', index=False)
        self.trained = True
        
        # Average results at the end
        if self.evaluation_results:
            avg_metrics = {key: np.mean([m[key] for m in self.evaluation_results]) for key in self.evaluation_results[0]}
        else:
            avg_metrics = self.record_evaluation_epoch(test_data)  # Fallback
        
        return avg_metrics["auroc"], avg_metrics["full_pixel_auroc"], avg_metrics["pro"], None, None, None, avg_metrics["saliency_cr_f1"]

    @torch.no_grad()
    def predict(self, datas):
        # 역할: 입력의 예측 또는 이상 탐지 점수와 map을 계산.
        # 매개변수: datas.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        scores_img = []
        scores_maps = []

        gt_list = []
        gt_mask_list = []
        
        self.Dinomaly.eval()
        gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(self.device)
        with tqdm.tqdm(datas, desc="Inferring...", leave=False) as data_iterator:
            for data_item in data_iterator:
                gt_list += [data_item['is_anomaly']]
                name = data_item['image_name']
                image = data_item['image'].to(self.device)
                
                output = self.Dinomaly(image)
                en, de = output[0], output[1]
                # encoder와 decoder token feature의 차이를 위치별 score map으로 바꿉니다.
                score_map, _ = cal_anomaly_maps(en, de, image.shape[-1])
                
                score_map = F.interpolate(score_map, size=256, mode='bilinear', align_corners=False)
                gt = F.interpolate(data_item['mask'], size=256, mode='nearest')
                gt_mask_list += [gt]
                
                score_map = gaussian_kernel(score_map).cpu()
                
                # 한 픽셀의 우연한 spike 대신 높은 score 상위 1%의 평균으로 image score를 만듭니다.
                anomaly_map = score_map.flatten(1)
                score_img = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :int(anomaly_map.shape[1] * 0.01)]
                score_img = score_img.mean(dim=1)
                
                scores_maps += score_map
                scores_img += score_img

        gt_mask_list = np.asarray(gt_mask_list, dtype=int)
        scores_maps = np.asarray(scores_maps, dtype=float)

        segmentations = [torch.tensor(ma).squeeze() for ma in scores_maps]
        masks_gt = [torch.tensor(ma) for ma in gt_mask_list]
        scores = [sc.cpu().item() for sc in scores_img]
        labels_gt = [gt.cpu().item() for gt in gt_list]

        return scores, segmentations, None, labels_gt, masks_gt

        per_pixel_rocauc = -1
        try:
            per_pixel_rocauc = roc_auc_score(gt_mask_list.flatten(), scores_maps.flatten())
        except:
            pass
        result_dict = roc_auc_score(gt_list, scores_img), per_pixel_rocauc, -1, scores_maps, -1, -1

        return result_dict
# 한국어 코드 안내: 이 파일은 이상 탐지 방법의 학습·예측 제어을 담당합니다.
# 원본 실행 로직은 변경하지 않고, 초보자용 설명 주석만 추가했습니다.
