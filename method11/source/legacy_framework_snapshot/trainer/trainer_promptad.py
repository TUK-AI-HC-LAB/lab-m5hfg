"""
PromptAD 를 torch 로 학습한 이후에 prediction 만 onnx 로 사용하기 위해 코드 수정 필요.
1. task == 'seg' 모드는 사용하지 않으니 삭제.
2. 학습은 다른 파일에서 할거니까, 학습 코드도 삭제.
3. 아래 코드를 참고해서 ONNX 변환 코드를 작성할 것. 마지막에 torch 와 onnx 모델의 output 을 검증하는 코드가 있어야 함.

<코드>
def convert_to_onnx(
    model: nn.Module,
    onnx_out_path: str,
    input_shape=(3, 640, 640),
    max_text_length=20,
    exemplar_path=None,
    exemplar_box=None,
):
    #기존 성공적으로 변환해온 onnx 변환 함수를 참고하여 작성한 예시

    #Args:
    #    model: PyTorch model
    #    onnx_out_path: 변환된 onnx 파일 경로
    #    input_shape: (C, H, W)
    #    max_text_length: 텍스트 최대 길이 (dummy)
    #    exemplar_path: exemplar 이미지 경로 (dummy)
    #    exemplar_box: exemplar bounding box (dummy)
    #Returns:
    #    onnx_model (onnx.ModelProto)
    model.eval()
    device = next(model.parameters()).device

    # (B, C, H, W) 더미 입력
    dummy_input = torch.randn(
        1, input_shape[0], input_shape[1], input_shape[2],
        dtype=torch.float32, device=device
    )

    # GroundingDINO의 forward는 여러 인자를 요구하므로,
    # 현재는 간단히 이미지 텐서만 넣어도 동작하도록 wrapper를 만든다.(이후 exemplar, caption 등 추가 처리가 필요할 수 있음)

    class OnnxWrapper(nn.Module):
        def __init__(self, gmodel):
            super().__init__()
            self.gmodel = gmodel

        def forward(self, x):
            nested = nested_tensor_from_tensor_list(x)
            exemp = [torch.zeros((0, 4), dtype=torch.float32, device=x.device)]
            labels = [torch.tensor([0], dtype=torch.long, device=x.device)]

            # 구분 부호를 넣어주면 bertwarper.py에서 정상적으로 토큰 구분이 이루어져 cate_to_token_mask_list가 비지 않게 되어 에러가 사라짐
            outputs = self.gmodel(
                samples=nested,
                exemplars=exemp,
                labels=labels,
                captions=["dummy caption ."]  # 뒤에 마침표 추가
            )
            return outputs["pred_logits"], outputs["pred_boxes"]


    wrapper = OnnxWrapper(model).to(device)
    out_logits, out_boxes = wrapper(dummy_input)

    input_names = ["input_image"]
    output_names = ["pred_logits", "pred_boxes"]

    dynamic_axes = {
        "input_image": {0: "batch_size", 2: "height", 3: "width"},
        "pred_logits": {0: "batch_size"},
        "pred_boxes": {0: "batch_size"},
    }

    torch.onnx.export(
        wrapper,
        dummy_input,
        onnx_out_path,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=16,
        do_constant_folding=True
    )

    # 모델 로드 및 검사
    onnx_model = onnx.load(onnx_out_path)
    onnx.checker.check_model(onnx_model)

    # 추론 테스트 - GPU 사용
    ort_session = onnxruntime.InferenceSession(
        onnx_out_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    # 동일한 입력으로 ONNX 모델 실행
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    pred_logits_onnx = torch.tensor(ort_outs[0]).to(device)
    pred_boxes_onnx = torch.tensor(ort_outs[1]).to(device)

    # PyTorch 모델과 ONNX 모델 출력 비교
    torch_out_logits = out_logits
    torch_out_boxes = out_boxes

    # 비교 (GPU 텐서 간 직접 비교)
    # Calculate max absolute and relative differences
    logits_abs_diff = torch.max(torch.abs(torch_out_logits - pred_logits_onnx))
    boxes_abs_diff = torch.max(torch.abs(torch_out_boxes - pred_boxes_onnx))
    
    # Calculate relative differences where values are not close to zero
    logits_rel_diff = torch.max(torch.abs((torch_out_logits - pred_logits_onnx) / (torch_out_logits + 1e-7)))
    boxes_rel_diff = torch.max(torch.abs((torch_out_boxes - pred_boxes_onnx) / (torch_out_boxes + 1e-7)))
    
    logits_eq = torch.allclose(torch_out_logits, pred_logits_onnx, rtol=1e-02, atol=1e-04)
    boxes_eq = torch.allclose(torch_out_boxes, pred_boxes_onnx, rtol=1e-02, atol=1e-04)
    
    if logits_eq and boxes_eq:
        print(f"[INFO] ONNX export success. Model saved to {onnx_out_path}")
    else:
        print("[WARNING] ONNX export finished but outputs differ beyond threshold.")
        print(f"Logits max abs diff: {logits_abs_diff.item():.6f}, max rel diff: {logits_rel_diff.item():.6f}")
        print(f"Boxes max abs diff: {boxes_abs_diff.item():.6f}, max rel diff: {boxes_rel_diff.item():.6f}")

    return onnx_model

"""
# Inference time 2.3ms 매우 빠름. pixel 까지 포함했을때 성능임.

from loguru import logger
import time
import numpy as np
from PIL import Image
import wandb
import tqdm
import matplotlib
matplotlib.use("Agg")
import os

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim.lr_scheduler
from scipy.ndimage import gaussian_filter

from . import CLIPAD
from .CLIPAD.prompt_utils.csv_utils import *
from .CLIPAD.prompt_utils.metrics import *
from .CLIPAD.prompt_utils.training_utils import *
from .CLIPAD.prompt_utils.eval_utils import *
from .CLIPAD.ad_prompts import *
from trainer.trainer import Trainer

class Trainer_PromptAD(Trainer):
    def initialize_model(self, **kwargs):
        # 역할: `initialize_model`에 해당하는 작업을 수행.
        # 매개변수: **kwargs.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        pass


    def set_ea_modules(self):
        # 역할: 객체의 설정 또는 내부 상태를 지정.
        # 매개변수: 없음.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        pass

    def _feature_extract(self, training_data):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: training_data.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        features1 = []
        features2 = []
        for data_item in training_data:
            data = data_item['image'][:self.args.k_shot].to(self.device)
            self.promptAD.set_shot(len(data))
            _, _, feature_map1, feature_map2 = self.promptAD.encode_image(data)
            features1.append(feature_map1)
            features2.append(feature_map2)
        
        features1 = torch.cat(features1, dim=0)
        features2 = torch.cat(features2, dim=0)
        self.promptAD.build_image_feature_gallery(features1, features2)

    def _meta_train(self, training_data, val_data, test_data, dataset_name):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: training_data, val_data, test_data, dataset_name.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        if self.args.full_shot:
            setattr(self.args, 'k_shot', len(training_data.dataset))

        promptAD = PromptAD(device=self.device, **vars(self.args))
        self.promptAD = promptAD.to(self.device)
        self.promptAD.eval()
        self.optimizer = torch.optim.SGD(self.promptAD.prompt_learner.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.meta_epochs, eta_min=1e-5)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.criterion_tip = TripletLoss(margin=0.0)
        self._feature_extract(training_data=training_data)

        for i_mepoch in range(self.meta_epochs):
            logger.info(f"\n\n----- {i_mepoch} -----")
            self.promptAD.train()

            for data_item in training_data:
                if self.args.full_shot:
                    data = data_item['image'].to(self.device)
                else:
                    data = data_item['image'][:self.args.k_shot].to(self.device)

                normal_text_prompt, abnormal_text_prompt_handle, abnormal_text_prompt_learned = self.promptAD.prompt_learner()

                self.optimizer.zero_grad()

                normal_text_features = self.promptAD.encode_text_embedding(normal_text_prompt, self.promptAD.tokenized_normal_prompts)

                abnormal_text_features_handle = self.promptAD.encode_text_embedding(abnormal_text_prompt_handle, self.promptAD.tokenized_abnormal_prompts_handle)
                abnormal_text_features_learned = self.promptAD.encode_text_embedding(abnormal_text_prompt_learned, self.promptAD.tokenized_abnormal_prompts_learned)
                abnormal_text_features = torch.cat([abnormal_text_features_handle, abnormal_text_features_learned], dim=0)

                # compute mean
                mean_ad_handle = torch.mean(F.normalize(abnormal_text_features_handle, dim=-1), dim=0)
                mean_ad_learned = torch.mean(F.normalize(abnormal_text_features_learned, dim=-1), dim=0)

                loss_match_abnormal = (mean_ad_handle - mean_ad_learned).norm(dim=0) ** 2.0

                cls_feature, feature_map, _, _ = self.promptAD.encode_image(data)

                # compute v2t loss and triplet loss
                normal_text_features_ahchor = normal_text_features.mean(dim=0).unsqueeze(0)
                normal_text_features_ahchor = normal_text_features_ahchor / normal_text_features_ahchor.norm(dim=-1, keepdim=True)

                abnormal_text_features_ahchor = abnormal_text_features.mean(dim=0).unsqueeze(0)
                abnormal_text_features_ahchor = abnormal_text_features_ahchor / abnormal_text_features_ahchor.norm(dim=-1, keepdim=True)
                abnormal_text_features = abnormal_text_features / abnormal_text_features.norm(dim=-1, keepdim=True)

                l_pos = torch.einsum('nc,cm->nm', cls_feature, normal_text_features_ahchor.transpose(0, 1))
                l_neg_v2t = torch.einsum('nc,cm->nm', cls_feature, abnormal_text_features.transpose(0, 1))

                if self.promptAD.precision == 'fp16':
                    logit_scale = self.promptAD.model.logit_scale.half()
                else:
                    logit_scale = self.promptAD.model.logit_scalef

                logits_v2t = torch.cat([l_pos, l_neg_v2t], dim=-1) * logit_scale

                target_v2t = torch.zeros([logits_v2t.shape[0]], dtype=torch.long).to(self.device)

                loss_v2t = self.criterion(logits_v2t, target_v2t)

                trip_loss = self.criterion_tip(cls_feature, normal_text_features_ahchor, abnormal_text_features_ahchor)
                loss = loss_v2t + trip_loss + loss_match_abnormal * self.args.lambda1

                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
            self.promptAD.build_text_feature_gallery()

            self.promptAD.eval()

    def predict(self, datas):
        # 역할: 입력의 예측 또는 이상 탐지 점수와 map을 계산.
        # 매개변수: datas.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        scores_img = []
        score_maps = []
        test_imgs = []
        gt_list = []
        gt_mask_list = []
        names = []
        i = 0

        times = []
        len_ = []
        with tqdm.tqdm(datas, desc="Inferring...", leave=False) as data_iterator:
            for data_item in data_iterator:
                name = data_item['image_name']
                label = data_item['is_anomaly']
                mask = data_item['mask'].squeeze(1)
                data = data_item['image'].to(self.device)
                for d, n, l, m in zip(data, name, label, mask):
                    test_imgs += [denormalization(d.cpu().numpy())]
                    l = l.numpy().item()
                    m = m.numpy()
                    m[m > 0] = 1

                    names += [n]
                    gt_list += [l]
                    gt_mask_list += [m]
                
                st = time.time()
                score_img, score_map = self.promptAD(data, 'cls')
                #score_map = self.promptAD(data, 'seg')
                times.append(time.time() - st)
                len_.append(data.shape[0])
                # [1], [1, 240, 240]
                score_maps += score_map
                scores_img += score_img

        logger.info(f"Average inference time: {np.sum(times) / np.sum(len_)}")
        #test_imgs, score_maps, gt_mask_list = specify_resolution(test_imgs, score_maps, gt_mask_list, resolution=(self.args.resolution, self.args.resolution))
        #result_dict = metric_cal_img(np.array(scores_img), gt_list, np.array(score_maps)), metric_cal_pix(np.array(score_maps), gt_mask_list), -1, score_maps, -1, -1
        scores_img = [a for a in np.array(score_maps).reshape(np.array(score_maps).shape[0], -1).max(axis=1)] # metric_cal_img 참고함
        result_dict = scores_img, score_maps, None, gt_list, gt_mask_list

        #return scores, masks, features, labels_gt, masks_gt

        return result_dict
    
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        # 역할: 객체를 사용할 수 있도록 필요한 속성과 구성요소를 초기화.
        # 매개변수: margin.
        # 반환값: 없음. 초기화가 끝난 객체를 Python이 생성합니다..
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):

        # 역할: 입력을 신경망 계층에 통과시켜 출력 tensor를 계산.
        # 매개변수: anchor, positive, negative.
        # 반환값: 모델이 계산한 tensor 또는 모델 출력입니다..
        pos_distance = torch.sum((anchor - positive).pow(2), dim=1)

        neg_distance = torch.sum((anchor - negative).pow(2), dim=1)

        loss = torch.relu(pos_distance - neg_distance + self.margin)

        return torch.mean(loss)

valid_backbones = ['ViT-B-16-plus-240', "ViT-B-16"]
valid_pretrained_datasets = ['laion400m_e32']

mean_train = [0.48145466, 0.4578275, 0.40821073]
std_train = [0.26862954, 0.26130258, 0.27577711]

def _convert_to_rgb(image):
    # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
    # 매개변수: image.
    # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
    return image.convert('RGB')

def denormalization(x):
    # 역할: `denormalization`에 해당하는 작업을 수행.
    # 매개변수: x.
    # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
    x = (((x.transpose(1, 2, 0) * std_train) + mean_train) * 255.).astype(np.uint8)
    return x

class PromptLearner(nn.Module):
    def __init__(self, n_ctx, n_pro, n_ctx_ab, n_pro_ab, classname, clip_model, pre):
        # 역할: 객체를 사용할 수 있도록 필요한 속성과 구성요소를 초기화.
        # 매개변수: n_ctx, n_pro, n_ctx_ab, n_pro_ab, classname, clip_model, pre.
        # 반환값: 없음. 초기화가 끝난 객체를 Python이 생성합니다..
        super().__init__()
        classname = classname.replace("_visa", "")

        if pre == 'fp16':
            dtype = torch.float16
        else:
            dtype = torch.float32

        state_anomaly1 = state_anomaly + class_state_abnormal[classname]

        if classname in class_mapping:
            classname = class_mapping[classname]

        ctx_dim = clip_model.ln_final.weight.shape[0]

        # random initialization
        normal_ctx_vectors = torch.empty(n_pro, n_ctx, ctx_dim, dtype=dtype)
        abnormal_ctx_vectors = torch.empty(n_pro_ab, n_ctx_ab, ctx_dim, dtype=dtype)

        nn.init.normal_(normal_ctx_vectors, std=0.02)
        nn.init.normal_(abnormal_ctx_vectors, std=0.02)

        normal_prompt_prefix = " ".join(["N"] * n_ctx)
        abnormal_prompt_prefix = " ".join(["A"] * n_ctx_ab)

        self.normal_ctx = nn.Parameter(normal_ctx_vectors)  # to be optimized
        self.abnormal_ctx = nn.Parameter(abnormal_ctx_vectors)  # to be optimized

        # normal prompt
        normal_prompts = [normal_prompt_prefix + " " + classname + "." for _ in range(n_pro)]

        # abnormal prompt
        self.n_ab_handle = len(state_anomaly1)
        abnormal_prompts_handle = [normal_prompt_prefix + " " + state.format(classname) + "." for state in state_anomaly1 for _ in range(n_pro)]
        abnormal_prompts_learned = [normal_prompt_prefix + " " + abnormal_prompt_prefix + " " + classname + "." for _ in range(n_pro_ab) for _ in range(n_pro)]

        # abnormal_prompts = abnormal_prompts_learned + abnormal_prompts_handle

        tokenized_normal_prompts = CLIPAD.tokenize(normal_prompts)
        tokenized_abnormal_prompts_handle = torch.cat([CLIPAD.tokenize(p) for p in abnormal_prompts_handle])
        tokenized_abnormal_prompts_learned = torch.cat([CLIPAD.tokenize(p) for p in abnormal_prompts_learned])

        with torch.no_grad():
            normal_embedding = clip_model.token_embedding(tokenized_normal_prompts).type(dtype)
            abnormal_embedding_handle = clip_model.token_embedding(tokenized_abnormal_prompts_handle).type(dtype)
            abnormal_embedding_learned = clip_model.token_embedding(tokenized_abnormal_prompts_learned).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("normal_token_prefix", normal_embedding[:, :1, :])  # SOS
        self.register_buffer("normal_token_suffix", normal_embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.register_buffer("abnormal_token_prefix_handle", abnormal_embedding_handle[:, :1, :])  # SOS
        self.register_buffer("abnormal_token_suffix_handle", abnormal_embedding_handle[:, 1 + n_ctx:, :])  # CLS, EOS

        self.register_buffer("abnormal_token_prefix_learned", abnormal_embedding_learned[:, :1, :])  # SOS
        self.register_buffer("abnormal_token_suffix_learned", abnormal_embedding_learned[:, 1 + n_ctx + n_ctx_ab:, :])  # CLS, EOS

        self.n_pro = n_pro
        self.n_ctx = n_ctx
        self.n_pro_ab = n_pro_ab
        self.n_ctx_ab = n_ctx_ab
        self.tokenized_normal_prompts = tokenized_normal_prompts  # torch.Tensor
        self.tokenized_abnormal_prompts_handle = tokenized_abnormal_prompts_handle  # torch.Tensor
        self.tokenized_abnormal_prompts_learned = tokenized_abnormal_prompts_learned  # torch.Tensor
        # self.tokenized_abnormal_prompts = torch.cat([tokenized_abnormal_prompts_handle, tokenized_abnormal_prompts_learned], dim=0)
        # self.tokenized_abnormal_prompts = tokenized_abnormal_prompts_handle
        # self.name_lens = name_lens

    def forward(self):

        # learned normal prompt
        # 역할: 입력을 신경망 계층에 통과시켜 출력 tensor를 계산.
        # 매개변수: 없음.
        # 반환값: 모델이 계산한 tensor 또는 모델 출력입니다..
        normal_ctx = self.normal_ctx

        normal_prefix = self.normal_token_prefix
        normal_suffix = self.normal_token_suffix

        normal_prompts = torch.cat(
            [
                normal_prefix,  # (n_pro, 1, dim)
                normal_ctx,     # (n_pro, n_ctx, dim)
                normal_suffix,  # (n_pro, *, dim)
            ],
            dim=1,
        )

        # handle abnormal prompt
        n_ab_handle = self.n_ab_handle

        n_pro, n_ctx, dim = normal_ctx.shape
        normal_ctx1 = normal_ctx.unsqueeze(0).expand(n_ab_handle, -1, -1, -1).reshape(-1, n_ctx, dim)

        abnormal_prefix_handle = self.abnormal_token_prefix_handle
        abnormal_suffix_handle = self.abnormal_token_suffix_handle

        abnormal_prompts_handle = torch.cat(
            [
                abnormal_prefix_handle,     # (n_pro * n_ab_handle, 1, dim)
                normal_ctx1,                # (n_pro * n_ab_handle, n_ctx, dim)
                abnormal_suffix_handle,     # (n_pro * n_ab_handle, *, dim)
            ],
            dim=1,
        )

        # learned abnormal prompt
        abnormal_prefix_learned = self.abnormal_token_prefix_learned
        abnormal_suffix_learned = self.abnormal_token_suffix_learned
        abnormal_ctx = self.abnormal_ctx
        n_pro_ad, n_ctx_ad, dim_ad = abnormal_ctx.shape
        normal_ctx2 = normal_ctx.unsqueeze(0).expand(self.n_pro_ab, -1, -1, -1).reshape(-1, n_ctx, dim)
        abnormal_ctx = abnormal_ctx.unsqueeze(0).expand(self.n_pro, -1, -1, -1).reshape(-1, n_ctx_ad, dim_ad)

        abnormal_prompts_learned = torch.cat(
            [
                abnormal_prefix_learned,        # (n_pro * n_pro_ab, 1, dim)
                normal_ctx2,                    # (n_pro * n_pro_ab, n_ctx, dim)
                abnormal_ctx,                   # (n_pro * n_pro_ab, n_ctx_ab, dim)
                abnormal_suffix_learned,        # (n_pro * n_pro_ab, *, dim)
            ],
            dim=1,
        )

        # abnormal_prompts = torch.cat([abnormal_prompts_handle, abnormal_prompts_learned], dim=0)
        # abnormal_prompts = abnormal_prompts_handle

        return normal_prompts, abnormal_prompts_handle, abnormal_prompts_learned

class PromptAD(torch.nn.Module):
    def __init__(self, device, backbone, pretrained_dataset, n_ctx, n_pro, n_ctx_ab, n_pro_ab, subdatasets, **kwargs):
        # 역할: 객체를 사용할 수 있도록 필요한 속성과 구성요소를 초기화.
        # 매개변수: device, backbone, pretrained_dataset, n_ctx, n_pro, n_ctx_ab, n_pro_ab, subdatasets, **kwargs.
        # 반환값: 없음. 초기화가 끝난 객체를 Python이 생성합니다..
        super(PromptAD, self).__init__()

        self.shot = kwargs['k_shot']

        self.out_size_h = kwargs['resolution']
        self.out_size_w = kwargs['resolution']
        self.precision = 'fp16' #precision  -40% GPU memory (2.8G->1.6G) with slight performance drop

        self.device = device
        self.get_model(n_ctx, n_pro, n_ctx_ab, n_pro_ab, subdatasets[0], backbone, pretrained_dataset)
        self.phrase_form = '{}'
        self.device = device

        # version v1: no norm for each of linguistic embedding
        # version v1:    norm for each of linguistic embedding
        self.version = 'V1' # V1:
        # visual textual, textual_visual

        """self.transform = transforms.Compose([
            transforms.Resize((kwargs['resize'], kwargs['resize']), Image.BICUBIC),
            transforms.CenterCrop(kwargs['img_cropsize']),
            _convert_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_train, std=std_train)])

        self.gt_transform = transforms.Compose([
            transforms.Resize((kwargs['resize'], kwargs['resize']), Image.NEAREST),
            transforms.CenterCrop(kwargs['img_cropsize']),
            transforms.ToTensor()])"""
        
    def set_shot(self, shot):
        # 역할: 객체의 설정 또는 내부 상태를 지정.
        # 매개변수: shot.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        self.shot = shot

    def get_model(self, n_ctx, n_pro, n_ctx_ab, n_pro_ab, class_name, backbone, pretrained_dataset):

        # 역할: 요청한 내부 정보 또는 계산 결과를 가져옴.
        # 매개변수: n_ctx, n_pro, n_ctx_ab, n_pro_ab, class_name, backbone, pretrained_dataset.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        assert backbone in valid_backbones
        assert pretrained_dataset in valid_pretrained_datasets

        model, _, _ = CLIPAD.create_model_and_transforms(model_name=backbone, pretrained=pretrained_dataset, precision = self.precision)
        tokenizer = CLIPAD.get_tokenizer(backbone)
        model.eval()

        self.prompt_learner = PromptLearner(n_ctx, n_pro, n_ctx_ab, n_pro_ab, class_name, model, self.precision)
        self.model = model.to(self.device)

        self.tokenizer = tokenizer
        self.normal_text_features = None
        self.abnormal_text_features = None
        self.grid_size = model.visual.grid_size
        self.visual_gallery = None

        visual_gallery1 = torch.zeros((self.shot*self.grid_size[0]*self.grid_size[1], self.model.visual.embed_dim))
        self.register_buffer("feature_gallery1", visual_gallery1)

        visual_gallery2 = torch.zeros((self.shot*self.grid_size[0]*self.grid_size[1], self.model.visual.embed_dim))
        self.register_buffer("feature_gallery2", visual_gallery2)

        text_features = torch.zeros((2, self.model.visual.output_dim))
        self.register_buffer("text_features", text_features)

        if self.precision == 'fp16':
            self.feature_gallery1  = self.feature_gallery1.half()
            self.feature_gallery2  = self.feature_gallery2.half()
            self.text_features  = text_features.half()

        # # for testing
        # p1, p2 = self.prompt_learner()
        self.tokenized_normal_prompts = self.prompt_learner.tokenized_normal_prompts
        self.tokenized_abnormal_prompts_handle = self.prompt_learner.tokenized_abnormal_prompts_handle
        self.tokenized_abnormal_prompts_learned = self.prompt_learner.tokenized_abnormal_prompts_learned
        self.tokenized_abnormal_prompts = torch.cat([self.tokenized_abnormal_prompts_handle, self.tokenized_abnormal_prompts_learned], dim=0)

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor):

        # 역할: `encode_image`에 해당하는 작업을 수행.
        # 매개변수: image.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        if self.precision == "fp16":
            image = image.half()
        image_features = self.model.encode_image(image)
        return [f / f.norm(dim=-1, keepdim=True) for f in image_features]

    @torch.no_grad()
    def encode_text(self, text: torch.Tensor):
        # 역할: `encode_text`에 해당하는 작업을 수행.
        # 매개변수: text.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        text_features = self.model.encode_text(text)
        # return [f / f.norm(dim=-1, keepdim=True) for f in text_features]
        return text_features

    def encode_text_embedding(self, text_embedding, original_tokens):
        # 역할: `encode_text_embedding`에 해당하는 작업을 수행.
        # 매개변수: text_embedding, original_tokens.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        text_features = self.model.encode_text_embeddings(text_embedding, original_tokens)
        return text_features

    @torch.no_grad()
    def build_text_feature_gallery(self):
        # 역할: `build_text_feature_gallery`에 해당하는 작업을 수행.
        # 매개변수: 없음.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        normal_text_embeddings, abnormal_text_embeddings_handle, abnormal_text_embeddings_learned = self.prompt_learner()
        abnormal_text_embeddings = torch.cat([abnormal_text_embeddings_handle, abnormal_text_embeddings_learned], dim=0)

        if self.version == "V1":
            normal_text_features = self.encode_text_embedding(normal_text_embeddings, self.tokenized_normal_prompts)
            abnormal_text_features = self.encode_text_embedding(abnormal_text_embeddings, self.tokenized_abnormal_prompts)
        elif self.version == "V2":
            normal_text_features = []
            for phrase_id in range(normal_text_embeddings.size()[0]):
                normal_text_feature = self.encode_text_embedding(normal_text_embeddings[phrase_id].unsqueeze(0), self.tokenized_normal_prompts)
                normal_text_feature = normal_text_feature/normal_text_feature.norm(dim=-1, keepdim=True)
                normal_text_features.append(normal_text_feature)
            normal_text_features = torch.cat(normal_text_features, 0).half()
            abnormal_text_features = []
            for phrase_id in range(abnormal_text_embeddings.size()[0]):
                abnormal_text_feature = self.encode_text_embedding(abnormal_text_embeddings[phrase_id].unsqueeze(0), self.tokenized_abnormal_prompts)
                abnormal_text_feature = abnormal_text_feature/abnormal_text_feature.norm(dim=-1, keepdim=True)
                abnormal_text_features.append(abnormal_text_feature)
            abnormal_text_features = torch.cat(abnormal_text_features, 0).half()
        else:
            raise NotImplementedError

        avr_normal_text_features = torch.mean(normal_text_features, dim=0, keepdim=True)
        avr_abnormal_text_features = torch.mean(abnormal_text_features, dim=0, keepdim=True)

        text_features_all = torch.cat([normal_text_features, abnormal_text_features], dim=0)
        text_features_all /= text_features_all.norm(dim=-1, keepdim=True)

        avr_normal_text_features = avr_normal_text_features
        avr_abnormal_text_features = avr_abnormal_text_features
        text_features = torch.cat([avr_normal_text_features, avr_abnormal_text_features], dim=0)
        self.text_features.copy_(text_features / text_features.norm(dim=-1, keepdim=True))

    def build_image_feature_gallery(self, features1, features2):
        # 역할: `build_image_feature_gallery`에 해당하는 작업을 수행.
        # 매개변수: features1, features2.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        b1, n1, d1 = features1.shape
        self.feature_gallery1.copy_(F.normalize(features1.reshape(-1, d1), dim=-1))

        b2, n2, d2 = features2.shape
        self.feature_gallery2.copy_(F.normalize(features2.reshape(-1, d2), dim=-1))

    def calculate_textual_anomaly_score(self, visual_features, task):
        # t = 100
        # 역할: `calculate_textual_anomaly_score`에 해당하는 작업을 수행.
        # 매개변수: visual_features, task.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        t = self.model.logit_scale
        # t = self.t
        # 역할: `calculate_visual_anomaly_score`에 해당하는 작업을 수행.
        # 매개변수: visual_features.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 역할: `calculate_visual_anomaly_score`에 해당하는 작업을 수행합니다.
        # 매개변수: visual_features.
        # 반환값: 구현에서 계산한 결과 또는 None입니다.
        N = visual_features[1].shape[0]

        if task == 'seg':
            # ############################################## local tokens scores ############################
            # token_features = self.cross_attention(visual_features[1])
            token_features = visual_features[1]
            local_normality_and_abnormality_score = (t * token_features @ self.text_features.T).softmax(dim=-1)

            local_abnormality_score = local_normality_and_abnormality_score[:, :, 1]

            local_abnormality_score = torch.zeros((N, self.grid_size[0] * self.grid_size[1])) + local_abnormality_score.cpu()
            local_abnormality_score = local_abnormality_score.reshape((N, self.grid_size[0], self.grid_size[1])).unsqueeze(1)

            return local_abnormality_score.detach()

        elif task == 'cls':
            # ################################################ global cls token scores ##########################
            # global_feature = self.cross_attention(visual_features[0].unsqueeze(dim=1)).squeeze(dim=1)
            global_feature = visual_features[0]
            global_normality_and_abnormality_score = (t * global_feature @ self.text_features.T).softmax(dim=-1)

            global_abnormality_score = global_normality_and_abnormality_score[:, 1]

            global_abnormality_score = global_abnormality_score.cpu()

            return global_abnormality_score.detach().numpy()

        else:
            assert 'task error'

    def calculate_visual_anomaly_score(self, visual_features):
        # 역할: `calculate_visual_anomaly_score`에 해당하는 작업을 수행합니다.
        # 매개변수: visual_features.
        # 반환값: 구현에서 계산한 결과 또는 None입니다.
        N = visual_features[1].shape[0]

        score1, _ = (1.0 - visual_features[2] @ self.feature_gallery1.t()).min(dim=-1)
        score1 /= 2.0

        score2, _ = (1.0 - visual_features[3] @ self.feature_gallery2.t()).min(dim=-1)
        score2 /= 2.0

        score = torch.zeros((N, self.grid_size[0] * self.grid_size[1])) + 0.5 * (score1 + score2).cpu()

        return score.reshape((N, self.grid_size[0], self.grid_size[1])).unsqueeze(1)

    def forward(self, images, task):

        # 역할: 입력을 신경망 계층에 통과시켜 출력 tensor를 계산.
        # 매개변수: images, task.
        # 반환값: 모델이 계산한 tensor 또는 모델 출력입니다..
        visual_features = self.encode_image(images)
        if task == 'seg':
            textual_anomaly_map = self.calculate_textual_anomaly_score(visual_features, 'seg')

            visual_anomaly_map = self.calculate_visual_anomaly_score(visual_features)
            #
            anomaly_map = 1. / (1. / textual_anomaly_map + 1. / visual_anomaly_map)
            # anomaly_map = 0.5 * (textual_anomaly_map + visual_anomaly_map)
            # anomaly_map = visual_anomaly_map
            # anomaly_map = textual_anomaly_map

            anomaly_map = F.interpolate(anomaly_map, size=(self.out_size_h, self.out_size_w), mode='bilinear', align_corners=False)

            am_pix = anomaly_map.squeeze(1).numpy()

            am_pix_list = []

            for i in range(am_pix.shape[0]):
                am_pix[i] = gaussian_filter(am_pix[i], sigma=4)
                am_pix_list.append(am_pix[i])

            return am_pix_list

        elif task == 'cls':
            textual_anomaly = self.calculate_textual_anomaly_score(visual_features, 'cls')

            visual_anomaly_map = self.calculate_visual_anomaly_score(visual_features)

            anomaly_map = F.interpolate(visual_anomaly_map, size=(self.out_size_h, self.out_size_w), mode='bilinear',
                                        align_corners=False)

            am_pix = anomaly_map.squeeze(1).numpy()

            am_pix_list = []

            for i in range(am_pix.shape[0]):
                am_pix_list.append(am_pix[i])

            am_img_list = []
            for i in range(textual_anomaly.shape[0]):
                am_img_list.append(textual_anomaly[i])

            return am_img_list, am_pix_list
        else:
            assert 'task error'
# 한국어 코드 안내: 이 파일은 이상 탐지 방법의 학습·예측 제어을 담당합니다.
# 원본 실행 로직은 변경하지 않고, 초보자용 설명 주석만 추가했습니다.
