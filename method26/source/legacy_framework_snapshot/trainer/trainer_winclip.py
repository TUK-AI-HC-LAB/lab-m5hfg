from loguru import logger
import wandb
import tqdm
import time

import torch
import torch.nn as nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import numpy as np

from trainer.trainer import Trainer
from .WinCLIP_lib import open_clip
from .WinCLIP_lib.open_clip import tokenizer

class Trainer_WinCLIP(Trainer):
    def initialize_model(self, **kwargs):
        # 역할: `initialize_model`에 해당하는 작업을 수행.
        # 매개변수: **kwargs.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        self.model = CLIP_AD()
        self.model.to(self.device)
        self.model.eval()

    def set_ea_modules(self):
        # 역할: 객체의 설정 또는 내부 상태를 지정.
        # 매개변수: 없음.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        pass
    
    @torch.no_grad()
    def _meta_train(self, training_data, val_data, test_data, dataset_name):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: training_data, val_data, test_data, dataset_name.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        obj = next(iter(training_data))['classname']
        self.Mermory_avg_normal_text_features, self.Mermory_avg_abnormal_text_features = prepare_text_future(self.model, obj)

        setattr(self.args, 'k_shot', len(training_data.dataset))
        self.few = False if self.args.k_shot <= 0 else True
        
        if self.few:
            for i_mepoch in range(self.meta_epochs):
                logger.info(f"\n\n----- {i_mepoch} -----")

                self.mid_memory, self.large_memory, self.patch_memory = initialize_memory(obj)
                i = 0
                for data_item in training_data:
                    image = data_item['image'].to(self.device)
                    patch_size = 16
                    large_scale_tokens, mid_scale_tokens, patch_tokens, class_tokens, large_scale, mid_scale = self.model.encode_image(image, patch_size)

                    for class_name, tokens in zip(obj, large_scale_tokens):
                        self.large_memory[class_name].append(tokens)
                    for class_name, tokens in zip(obj, mid_scale_tokens):
                        self.mid_memory[class_name].append(tokens)
                    for class_name, tokens in zip(obj, patch_tokens):
                        self.patch_memory[class_name].append(tokens)

                    i += 1
                    if i >= self.args.k_shot: break
                for class_name in obj:
                    self.large_memory[class_name] = torch.cat(self.large_memory[class_name]).to(self.device)
                    self.mid_memory[class_name] = torch.cat(self.mid_memory[class_name]).to(self.device)
                    self.patch_memory[class_name] = torch.cat(self.patch_memory[class_name]).to(self.device)

    
    @torch.no_grad()
    def predict(self, datas):
        # 역할: 입력의 예측 또는 이상 탐지 점수와 map을 계산.
        # 매개변수: datas.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        scores_img = []
        scores_maps = []

        gt_list = [] # gt 0, 1 image-det
        gt_mask_list = [] # gt [[0, 0, 0, ...], [0, 0, ...]] pixelmaps

        patch_size = 16
        with tqdm.tqdm(datas, desc="Inferring...", leave=False) as data_iterator:
            for data_item in data_iterator:
                gt_list += [data_item['is_anomaly']]
                gt_mask_list += [data_item['mask']]
                cls_name = data_item['classname']
                image = data_item['image'].to(self.device)
                b, c, h, w = image.shape
                
                average_normal_features = self.Mermory_avg_normal_text_features
                average_anomaly_features = self.Mermory_avg_abnormal_text_features
                large_scale_tokens, mid_scale_tokens, patch_tokens, class_tokens, large_scale, mid_scale = self.model.encode_image(image, patch_size)
                
                if self.few:
                    m_l = few_shot(self.large_memory, large_scale_tokens, cls_name)
                    m_m = few_shot(self.mid_memory, mid_scale_tokens, cls_name)
                    m_p = few_shot(self.patch_memory, patch_tokens, cls_name)

                    m_l  =  harmonic_aggregation((b, h//patch_size, w//patch_size) ,m_l, large_scale).to(self.device)
                    m_m  =  harmonic_aggregation((b, h//patch_size, w//patch_size) ,m_m, mid_scale).to(self.device)
                    m_p  =  m_p.reshape((b, h//patch_size, w//patch_size)).to(self.device)

                    few_shot_score = torch.nan_to_num((m_l + m_m + m_p)/3.0, nan=0.0, posinf=0.0, neginf=0.0)
                zscore = compute_score(class_tokens, torch.cat((average_normal_features, average_anomaly_features), dim = 1).permute(0, 2, 1))
                z0score = zscore[:,0,1]

                large_scale_simmarity = compute_sim(large_scale_tokens, torch.cat((average_normal_features, average_anomaly_features), dim = 1).permute(0, 2, 1))[:,:,1]
                mid_scale_simmarity = compute_sim(mid_scale_tokens, torch.cat((average_normal_features, average_anomaly_features), dim = 1).permute(0, 2, 1))[:,:,1]
                large_scale_score = harmonic_aggregation((b, h//patch_size, w//patch_size) ,large_scale_simmarity, large_scale)
                mid_scale_score  = harmonic_aggregation((b, h//patch_size, w//patch_size), mid_scale_simmarity, mid_scale)

                multiscale_score = mid_scale_score
                multiscale_score = torch.nan_to_num(3.0/(1.0/large_scale_score + 1.0/mid_scale_score + 1.0/z0score.unsqueeze(1).unsqueeze(1)), nan=0.0, posinf=0.0, neginf=0.0)
                multiscale_score = multiscale_score.to(self.device).unsqueeze(1)  # Add batch and channel dimensions

                if self.few:
                    multiscale_score = multiscale_score + few_shot_score.to(self.device).unsqueeze(1)
                    z0score = (z0score+ torch.max(torch.max(few_shot_score, dim = 1)[0],dim = 1)[0])/2.0
                
                multiscale_score = F.interpolate(multiscale_score, size=(h, w), mode='bilinear')
                multiscale_score = multiscale_score.squeeze()

                scores_maps += [multiscale_score.cpu()]
                scores_img += [z0score.cpu().item()]

        gt_mask_list = np.asarray(gt_mask_list, dtype=int)
        scores_maps = np.asarray(scores_maps, dtype=float)

        per_pixel_rocauc = -1
        try:
            per_pixel_rocauc = roc_auc_score(gt_mask_list.flatten(), scores_maps.flatten())
        except:
            pass
        #result_dict = roc_auc_score(gt_list, scores_img), per_pixel_rocauc, -1, scores_maps, -1, -1, -1
        return scores_img, scores_maps, None, gt_list, gt_mask_list

        #return scores, masks, features, labels_gt, masks_gt

        return result_dict
    

class patch_scale():
    def __init__(self, image_size):
        # 역할: 객체를 사용할 수 있도록 필요한 속성과 구성요소를 초기화.
        # 매개변수: image_size.
        # 반환값: 없음. 초기화가 끝난 객체를 Python이 생성합니다..
        self.h, self.w = image_size
 
    def make_mask(self, patch_size = 16, kernel_size = 16, stride_size = 16): 
        # 역할: `make_mask`에 해당하는 작업을 수행.
        # 매개변수: patch_size, kernel_size, stride_size.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        self.patch_size = patch_size
        self.patch_num_h = self.h//self.patch_size
        self.patch_num_w = self.w//self.patch_size
        ###################################################### patch_level
        self.kernel_size = kernel_size//patch_size
        self.stride_size = stride_size//patch_size
        self.idx_board = torch.arange(1, self.patch_num_h * self.patch_num_w + 1, dtype = torch.float32).reshape((1,1,self.patch_num_h, self.patch_num_w))
        patchfy = torch.nn.functional.unfold(self.idx_board, kernel_size=self.kernel_size, stride=self.stride_size)
        return patchfy

class CLIP_AD(nn.Module):
    def __init__(self, model_name = 'ViT-B-16-plus-240'):
        # 역할: 객체를 사용할 수 있도록 필요한 속성과 구성요소를 초기화.
        # 매개변수: model_name.
        # 반환값: 없음. 초기화가 끝난 객체를 Python이 생성합니다..
        super(CLIP_AD, self).__init__()
        self.model, _, self.preprocess = open_clip.create_customer_model_and_transforms(model_name, pretrained='laion400m_e31')
        self.mask = patch_scale((240,240))
    def multiscale(self):
        # 역할: `multiscale`에 해당하는 작업을 수행.
        # 매개변수: 없음.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        pass
    
    def encode_text(self, text):
        # 역할: `encode_text`에 해당하는 작업을 수행.
        # 매개변수: text.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        return self.model.encode_text(text)
    def encode_image(self, image, patch_size, mask=True):
        # 역할: `encode_image`에 해당하는 작업을 수행.
        # 매개변수: image, patch_size, mask.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        if mask:
            b, _, _, _ = image.shape
            large_scale = self.mask.make_mask(kernel_size=48, patch_size=patch_size).squeeze().cuda()
            mid_scale = self.mask.make_mask(kernel_size=32, patch_size=patch_size).squeeze().cuda()
            tokens_list, class_tokens, patch_tokens = self.model.encode_image(image, [large_scale,mid_scale], proj = False)
            large_scale_tokens, mid_scale_tokens = tokens_list[0], tokens_list[1]
            return large_scale_tokens, mid_scale_tokens, patch_tokens.unsqueeze(2), class_tokens, large_scale, mid_scale

class prompt_order():
    def __init__(self) -> None:
        # 역할: 객체를 사용할 수 있도록 필요한 속성과 구성요소를 초기화.
        # 매개변수: 없음.
        # 반환값: 없음. 초기화가 끝난 객체를 Python이 생성합니다..
        super().__init__()
        self.state_normal_list = [
            "{}",
            "flawless {}",
            "perfect {}",
            "unblemished {}",
            "{} without flaw",
            "{} without defect",
            "{} without damage"
        ]

        self.state_anomaly_list = [
            "damaged {}",
            "{} with flaw",
            "{} with defect",
            "{} with damage"
        ]

        self.template_list =[
        "a cropped photo of the {}.",
        "a close-up photo of a {}.",
        "a close-up photo of the {}.",
        "a bright photo of a {}.",
        "a bright photo of the {}.",
        "a dark photo of the {}.",
        "a dark photo of a {}.",
        "a jpeg corrupted photo of the {}.",
        "a jpeg corrupted photo of the {}.",
        "a blurry photo of the {}.",
        "a blurry photo of a {}.",
        "a photo of a {}.",
        "a photo of the {}.",
        "a photo of a small {}.",
        "a photo of the small {}.",
        "a photo of a large {}.",
        "a photo of the large {}.",
        "a photo of the {} for visual inspection.",
        "a photo of a {} for visual inspection.",
        "a photo of the {} for anomaly detection.",
        "a photo of a {} for anomaly detection."
        ]
    def prompt(self, class_name):
        # 역할: `prompt`에 해당하는 작업을 수행.
        # 매개변수: class_name.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        class_state = [ele.format(class_name) for ele in self.state_normal_list]
        normal_ensemble_template = [class_template.format(ele) for ele in class_state for class_template in self.template_list]
    
        class_state = [ele.format(class_name) for ele in self.state_anomaly_list]
        anomaly_ensemble_template = [class_template.format(ele) for ele in class_state for class_template in self.template_list]
        return normal_ensemble_template, anomaly_ensemble_template

def prepare_text_future(model, obj_list):
    # 역할: `prepare_text_future`에 해당하는 작업을 수행.
    # 매개변수: model, obj_list.
    # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
    Mermory_avg_normal_text_features = []
    Mermory_avg_abnormal_text_features = []
    text_generator = prompt_order()

    for i in obj_list:

        normal_description, abnormal_description = text_generator.prompt(i)

        normal_tokens = tokenizer.tokenize(normal_description)
        abnormal_tokens = tokenizer.tokenize(abnormal_description)
        normal_text_features = model.encode_text(normal_tokens.cuda()).float()
        abnormal_text_features = model.encode_text(abnormal_tokens.cuda()).float()

        avg_normal_text_features = torch.mean(normal_text_features, dim = 0, keepdim= True) 
        avg_abnormal_text_features = torch.mean(abnormal_text_features, dim = 0, keepdim= True)
        Mermory_avg_normal_text_features.append(avg_normal_text_features)
        Mermory_avg_abnormal_text_features.append(avg_abnormal_text_features)
    Mermory_avg_normal_text_features = torch.stack(Mermory_avg_normal_text_features)
    Mermory_avg_abnormal_text_features = torch.stack(Mermory_avg_abnormal_text_features)
    return Mermory_avg_normal_text_features, Mermory_avg_abnormal_text_features

def compute_score(image_features, text_features):
    # 역할: 이름에 해당하는 수치 또는 지표를 계산.
    # 매개변수: image_features, text_features.
    # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
    image_features /= image_features.norm(dim=1, keepdim=True)
    text_features /= text_features.norm(dim=1, keepdim=True)
    text_probs = (torch.bmm(image_features.unsqueeze(1), text_features)/0.07).softmax(dim=-1)

    return text_probs

def compute_sim(image_features, text_features):
    # 역할: 이름에 해당하는 수치 또는 지표를 계산.
    # 매개변수: image_features, text_features.
    # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=1, keepdim=True)
    simmarity = (torch.bmm(image_features.squeeze(2), text_features)/0.07).softmax(dim=-1)
    return simmarity

def harmonic_aggregation(score_size, simmarity, mask):
    # 역할: `harmonic_aggregation`에 해당하는 작업을 수행.
    # 매개변수: score_size, simmarity, mask.
    # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
    b, h, w = score_size
    simmarity = simmarity.double()
    score = torch.zeros((b, h*w)).to(simmarity).double()
    mask = mask.T
    for idx in range(h*w):
        patch_idx = torch.isin(mask, idx+1).sum(dim=1) == 1
        sum_num = sum(patch_idx)
        harmonic_sum = torch.sum(1.0 / simmarity[:, patch_idx], dim = -1)
        score[:, idx] =sum_num /harmonic_sum

    score = score.reshape(b, h, w)
    return score


from collections import OrderedDict
def initialize_memory(obj_list):
    # 역할: `initialize_memory`에 해당하는 작업을 수행.
    # 매개변수: obj_list.
    # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
    mid = []
    large = []
    patch = []
    for x in obj_list:
        mid.append((x, []))
        large.append((x, []))
        patch.append((x, []))
    mid_memory   = OrderedDict(mid)
    large_memory = OrderedDict(large)
    patch_memory = OrderedDict(patch)
    return mid_memory, large_memory, patch_memory

def few_shot(memory, token, class_name):
    # 역할: `few_shot`에 해당하는 작업을 수행.
    # 매개변수: memory, token, class_name.
    # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
    retrive = []
    for i in class_name:
        L, N, D = memory[i].shape
        retrive.append(memory[i].permute(2, 1, 0).reshape(D,-1)) # D NL
    retrive = torch.stack(retrive)# B D NL
     #B D L 
    M = 1/2 * torch.min(1.0 - torch.bmm(F.normalize(token.squeeze(2), dim = -1), F.normalize(retrive, dim = 1)), dim = -1)[0]
    return M
# 한국어 코드 안내: 이 파일은 이상 탐지 방법의 학습·예측 제어을 담당합니다.
# 원본 실행 로직은 변경하지 않고, 초보자용 설명 주석만 추가했습니다.
