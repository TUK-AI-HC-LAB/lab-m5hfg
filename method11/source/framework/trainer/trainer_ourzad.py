"""
QueCo 트레이너: 학습 없이 테스트셋만으로 이상을 탐지.

    Pass 1  테스트셋 전체를 인코딩해 메모리 뱅크 구축(정상+이상 섞인 bank이며 label부여 없이 구축).
    Pass 2  이미지마다 가까운 이웃 이미지를 골라 패치를 비교(시각 브랜치)하고,
            텍스트 브랜치와 융합해 이상 맵과 이미지 점수를 만든다.

세부 연산은 QueCo_lib 패키지에...
"""
import time

import tqdm
import torch
import torch.nn.functional as F
from loguru import logger

from trainer.trainer import Trainer
from .QueCo_lib import load
from .QueCo_lib.prompt_ensemble import PromptLearner
from .QueCo_lib.metrics import image_level_metrics, pixel_level_metrics
from .QueCo_lib.similarity import calc_similarity_logits, regrid_upsample

from .QueCo_lib.analysis import FusionDiagnostics, save_objectness_map, save_overlay_map
from .QueCo_lib.config import InferenceConfig, apply_default_args
from .QueCo_lib.fusion import (
    apply_objectness_gate, fuse_branches, fuse_layers, logits_to_anomaly_map,
    normalize_logits, topk_image_score,
)
from .QueCo_lib.pose_aux import PoseAux
from .QueCo_lib.utils import normalize_mask, parse_mvtec_cls_type, set_seed
from .QueCo_lib.visual_branch import (
    aggregate_over_neighbors, dino_agreement_map, dino_outlier_score, downsample_to_grid,
    grid_positions, mutual_maxsim, objectness_map, ot_normality_batch, patch_scores_to_logits,
)

CLIP_BACKBONE = "ViT-L/14@336px"
SELF_COR_ATTN_LAYERS = 20
TEXT_TEMPERATURE = 0.07

IMAGE_METRICS = ("image-auroc", "image-ap", "image-f1")
PIXEL_METRICS = ("pixel-auroc", "pixel-ap", "pixel-f1", "pixel-aupro")


class Trainer_OurZAD(Trainer):

    # ------------------------------------------------------------------
    # 모델 준비
    # ------------------------------------------------------------------
    def load(self,
             backbone,
             layers_to_extract_from,
             device,
             input_shape,
             pretrain_embed_dimension,
             target_embed_dimension,
             patchsize,
             meta_epochs,
             aed_meta_epochs,
             gan_epochs,
             noise_std,
             dsc_layers,
             dsc_hidden,
             dsc_margin,
             dsc_lr,
             auto_noise,
             train_backbone,
             cos_lr,
             pre_proj,
             proj_layer_type,
             mix_noise,
             onnx,
             args):

        self.args = args
        self.device = device
        self.backbone = backbone

        self.initialize_model()

    def initialize_model(self, **kwargs):
        apply_default_args(self.args)
        set_seed(42)

        design_details = {
            "Prompt_length": self.args.n_ctx,
            "learnabel_text_embedding_depth": self.args.depth,
            "learnabel_text_embedding_length": self.args.t_n_ctx,
            "others": self.args,
        }
        self.model, _ = load(CLIP_BACKBONE, device=self.device, design_details=design_details)
        self.model.eval()
        self.prompt_learner = PromptLearner(self.model.to("cpu"), design_details)
        self.model.use_DAttn(self.args.dino_model)
        self.model.visual.replace_with_EAttn(to_layer=SELF_COR_ATTN_LAYERS, type=self.args.attn_type)

        self.prompt_learner.to(self.device)
        self.model.to(self.device)

        # metal_nut / transistor 전용 보조 맵 (args.enable_pose_aux로 켜고 끔)
        self.pose_aux = PoseAux(self.model, self.prompt_learner, self.args, self.device)

    def set_ea_modules(self):
        pass

    def train(self, training_data, val_data, test_data, dataset_name):
        """학습 없이 추론하고, 7개 지표를 공통 getter용으로 기록합니다."""
        self.dataset_name = dataset_name

        # 이전 실행의 지표가 섞이지 않도록 비웁니다.
        self.evaluation_results = []

        logger.info("Skipping heavy training; calling predict(test_data).")

        # 현재 원본 predict()가 계산해서 반환하는 지표 7개입니다.
        raw_metrics = self.predict(test_data)

        # 반환 순서와 동일하게 이름을 붙입니다.
        metric_names = (
            "image_auroc",
            "image_ap",
            "image_f1",
            "full_pixel_auroc",
            "pixel_ap",
            "pixel_f1",
            "pixel_aupro",
        )
        record = dict(zip(metric_names, raw_metrics))

        # 기존 공통 평가 열과 연결합니다.
        record["auroc"] = record["image_auroc"]
        record["pro"] = record["pixel_aupro"]

        # QueCo Pixel F1과 별개인 지표이므로 미계산으로 표시합니다.
        record["saliency_cr_f1"] = -1

        # 공통 get_evaluation_metrics()가 읽는 목록에 저장합니다.
        self.evaluation_results.append(record)

        # 원본처럼 7개 지표 반환도 유지합니다.
        return raw_metrics

    # ------------------------------------------------------------------
    # 추론
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, datas):
        cfg = InferenceConfig.from_args(self.args)
        device = self.device

        # Pass 1: 메모리 뱅크
        bank = self._build_bank(datas, cfg)
        self.pose_aux.calibrate(bank)

        # 이미지 특징으로 이웃 이미지 선택 (자기 자신 제외)
        img_feats = torch.stack([b["img_feat"] for b in bank], dim=0).to(device).float()
        img_feats = F.normalize(img_feats, dim=-1)
        sims = img_feats @ img_feats.t()
        sims.fill_diagonal_(-1.0)
        num_refs = min(cfg.ref_topR, len(bank) - 1)

        results = {"gt_sp": [], "pr_sp": [], "imgs_masks": [], "anomaly_maps": []}
        diagnostics = FusionDiagnostics(
            self.args, self.dataset_name, cfg.fusion_mode, cfg.fixed_visual_weight
        )

        # Pass 2: 이미지별 추론
        start_time = time.time()
        for t_idx in tqdm.tqdm(range(len(bank)), desc="Inferring (test-only)", leave=False):
            item = bank[t_idx]
            img_path = item.get("img_path", "")
            cls, _ = parse_mvtec_cls_type(img_path)
            ref_ids = torch.topk(sims[t_idx], k=num_refs).indices.tolist()

            # (1) 시각 브랜치, (2) 텍스트 브랜치
            visual_logits = self._visual_branch(item, bank, ref_ids, cls, cfg)
            text_logits, text_feats = self._text_branch(item, cfg)

            # (3) 브랜치 융합
            if text_logits is not None:
                logits, stats, visual_norm, text_norm = fuse_branches(
                    visual_logits, text_logits,
                    mode=cfg.fusion_mode,
                    fixed_visual_weight=cfg.fixed_visual_weight,
                    T_branch=cfg.T_branch,
                )
                if cfg.fusion_diagnostics:
                    diagnostics.update(img_path, item["mask"], visual_norm, text_norm, logits, stats)
            else:
                logits = normalize_logits(visual_logits, T=cfg.T_branch)

            # (4) 배경 억제
            if cfg.use_objectness_gate:
                logits = apply_objectness_gate(
                    logits, item["obj_full"].to(device).unsqueeze(0),
                    gamma=cfg.gamma_gate, lam=cfg.lambda_gate,
                )

            # (5) 최종 이상 맵
            anomaly = logits_to_anomaly_map(logits)
            if self.pose_aux.enabled and cls in PoseAux.CLASSES:
                size = int(cfg.imagesize)
                aux_map = self.pose_aux.anomaly_map(item, cls, size, size)
                anomaly = torch.maximum(anomaly, aux_map.to(anomaly.device))
            anomaly = anomaly.detach().cpu()

            if cfg.save_vis:
                self._save_anomaly_map(anomaly, item["img_path"], cfg)

            # (6) 이미지 점수
            score = self._image_score(anomaly, item, text_feats, cfg)

            results["imgs_masks"].append(item["mask"])
            results["gt_sp"].append(item["label"])
            results["anomaly_maps"].append(anomaly.unsqueeze(0))
            results["pr_sp"].append(float(score))

        print(f"{len(bank) / max(time.time() - start_time, 1e-6):.3f} it/s")

        diagnostics.save()
        return self._compute_metrics(results)

    # ------------------------------------------------------------------
    # Pass 1: 인코딩 / 메모리 뱅크
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _encode_image(self, image):
        vfm = getattr(self.model, "vfm", None)
        if vfm is not None:
            vfm.to(device=image.device, dtype=image.dtype)

        return self.model.encode_image(
            image, self.args.features_list, self_cor_attn_layers=SELF_COR_ATTN_LAYERS
        )

    @torch.no_grad()
    def _build_bank(self, datas, cfg):
        """
        테스트 이미지마다 특징을 뽑아 CPU에 보관한다. (배치 크기 1 가정)

        항목: img_feat [D], patches (레이어별 [N, D]), grid, P [N, 2],
              obj_full [H, W], O_grid [N], dino_score [N], mask, label, img_path
        """
        device = self.device
        bank = []
        for data_item in datas:
            image = data_item["image"].to(device)
            img_path = data_item.get("image_path", None)

            img_feat, patch_feats, dino_feats, dino_feats2 = self._encode_image(image)
            img_feat = F.normalize(img_feat, dim=-1)[0].detach().cpu()

            pf_stack = torch.stack(patch_feats, dim=0)
            if pf_stack.dim() == 3:
                pf_stack = pf_stack.unsqueeze(1)
            patches = [
                F.normalize(pf_stack[l, 0], dim=-1).contiguous().cpu()
                for l in range(pf_stack.size(0))
            ]

            num_patches = patches[-1].size(0)
            grid = int(round(num_patches ** 0.5))
            if grid * grid != num_patches:
                raise ValueError(f"Patch count N={num_patches} is not square")

            obj_full = objectness_map(dino_feats2[0], cfg.imagesize)[0].detach().cpu()
            if cfg.save_obj_vis:
                obj_vis = dino_agreement_map(dino_feats, out_size=cfg.imagesize)[0]
                save_objectness_map(obj_vis, img_path, save_dir="./dino_feats_vis", tag="dino_agree")

            dino_score = dino_outlier_score(dino_feats, target_grid=grid, topk=32)[0].detach().cpu()

            bank.append(dict(
                img_feat=img_feat,
                patches=patches,
                grid=grid,
                P=grid_positions(grid, device).cpu(),
                obj_full=obj_full,
                O_grid=downsample_to_grid(obj_full, grid, device).cpu(),
                dino_score=dino_score,
                mask=normalize_mask(data_item["mask"]).cpu(),
                label=int(data_item["is_anomaly"]),
                img_path=img_path,
            ))
        return bank

    # ------------------------------------------------------------------
    # Pass 2: 브랜치별 계산
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _visual_branch(self, item, bank, ref_ids, cls, cfg):
        """
        이웃 이미지의 패치와 비교해 "비슷한 패치를 못 찾는 정도"를 이상 점수로 쓴다.
        레이어마다 OT 정상도와 mutual-kNN 정상도를 합치고, 레이어 결과를 평균한다.

        return: [1, H, W, 2]
        """
        device = self.device
        grid = item["grid"]
        Pi = item["P"].to(device)
        Oi = item["O_grid"].to(device)
        is_background = (Oi >= cfg.obj_tau).view(1, -1)

        use_ot = cfg.ablation_mode in ("all", "ot")
        use_mutual = cfg.ablation_mode in ("all", "mutual")

        # transistor는 위치가 어긋난 결함이 많아 mutual 매칭을 가까운 위치로 제한한다
        radius = cfg.transistor_mutual_radius if cls == "transistor" else cfg.mutual_local_radius

        layer_logits = []
        for l in range(len(item["patches"])):
            Zi = item["patches"][l].to(device)

            normality = []
            for s in range(0, len(ref_ids), cfg.neighbor_chunk):
                chunk_ids = ref_ids[s:s + cfg.neighbor_chunk]
                Zjs = torch.stack([bank[j]["patches"][l] for j in chunk_ids], dim=0).to(device)

                if use_ot:
                    Pj = bank[chunk_ids[0]]["P"].to(device)
                    Ojs = torch.stack([bank[j]["O_grid"] for j in chunk_ids], dim=0).to(device)
                    s_ot = ot_normality_batch(Zi, Zjs, Pi, Pj, Oi, Ojs, **cfg.ot)

                if use_mutual:
                    s_mut = torch.stack([
                        mutual_maxsim(
                            Zi, Zjs[b], topk=cfg.mutual_topk, tau=cfg.mutual_tau,
                            Pi=Pi, Pj=bank[j]["P"].to(device) if radius > 0 else None,
                            G=grid, radius=radius,
                        )
                        for b, j in enumerate(chunk_ids)
                    ], dim=0)

                if use_ot and use_mutual:
                    s_hyb = torch.sqrt(s_ot.clamp(min=1e-6) * s_mut.clamp(min=1e-6))
                    # 배경 패치는 OT 점수만 사용 (배경 잡음 억제)
                    s_hyb = torch.where(is_background, s_ot, s_hyb)
                else:
                    s_hyb = s_ot if use_ot else s_mut
                normality.append(s_hyb)

            sim_q = aggregate_over_neighbors(
                torch.cat(normality, dim=0), tau=cfg.mutual_tau, m=cfg.support_m, q=cfg.agg_q
            )
            patch_anomaly = (1.0 - sim_q).clamp(0, 1)
            layer_logits.append(
                patch_scores_to_logits(patch_anomaly, grid, cfg.imagesize, cfg.sigma, device)
            )

        obj_mask = item["obj_full"].unsqueeze(0).to(device)
        return fuse_layers(layer_logits, T_layer=cfg.T_layer, mask=obj_mask)

    @torch.no_grad()
    def _text_branch(self, item, cfg):
        """
        이미지에서 뽑은 프로토타입으로 정상/이상 프롬프트를 만들고 패치와의 유사도를 구한다.

        return: (logits [1, H, W, 2] 또는 None, 레이어별 텍스트 특징 리스트)
        """
        if self.prompt_learner is None or not cfg.use_text_branch:
            return None, []

        device = self.device
        img_feat = item["img_feat"].to(device).unsqueeze(0)
        dino_score = item["dino_score"].to(device).unsqueeze(0)

        layer_logits, text_feats = [], []
        for l in range(len(item["patches"])):
            patches = item["patches"][l].to(device).unsqueeze(0)

            prompts, tokenized_prompts, compound_prompts_text, _ = self.prompt_learner(
                F=patches, g=img_feat, dino_score=dino_score
            )
            t_normal = self.model.encode_text_learn(prompts[0], tokenized_prompts[0], compound_prompts_text)
            t_anomaly = self.model.encode_text_learn(prompts[1], tokenized_prompts[1], compound_prompts_text)

            t_feat = F.normalize(torch.stack([t_normal, t_anomaly], dim=1), dim=-1).float()
            text_feats.append(t_feat)

            patch_logits = calc_similarity_logits(patches, t_feat, temp=TEXT_TEMPERATURE)
            layer_logits.append(regrid_upsample(patch_logits, cfg.imagesize))

        return fuse_layers(layer_logits, T_layer=cfg.T_layer), text_feats

    @torch.no_grad()
    def _image_score(self, anomaly, item, text_feats, cfg):
        """이상 맵의 상위 픽셀 점수와, 이미지-텍스트 유사도 기반 점수를 섞는다."""
        score = topk_image_score(anomaly, cfg.topk_ratio)
        if self.prompt_learner is None or len(text_feats) == 0:
            return score

        t_feat = torch.stack(text_feats, dim=0).mean(dim=0)
        img_feat = item["img_feat"].to(self.device).unsqueeze(0)
        img_logits = calc_similarity_logits(img_feat, t_feat, temp=TEXT_TEMPERATURE)
        gap = (img_logits[:, 1] - img_logits[:, 0]).squeeze(0)

        a = cfg.alpha_imgscore
        return (1 - a) * score + a * torch.sigmoid(gap)

    # ------------------------------------------------------------------
    # 저장 / 평가
    # ------------------------------------------------------------------
    def _save_anomaly_map(self, anomaly, img_path, cfg):
        if isinstance(img_path, (list, tuple)) and len(img_path) > 0:
            img_path = img_path[0]
        if img_path is None:
            return
        save_overlay_map(
            anomaly.unsqueeze(0).numpy(), [img_path],
            save_dir=cfg.vis_dir, overlay=True, alpha=0.5, add_colorbar=False,
        )

    def _compute_metrics(self, results):
        """return: (image_auroc, image_ap, image_f1, pixel_auroc, pixel_ap, pixel_f1, pixel_aupro)"""
        results["imgs_masks"] = torch.cat(results["imgs_masks"]).cpu()
        results["anomaly_maps"] = torch.cat(results["anomaly_maps"]).cpu()
        results = {self.dataset_name: results}

        try:
            image_scores = [image_level_metrics(results, self.dataset_name, m) for m in IMAGE_METRICS]
            pixel_scores = [pixel_level_metrics(results, self.dataset_name, m) for m in PIXEL_METRICS]
        except Exception:
            logger.exception("Metric computation failed; returning -1 for all metrics.")
            return (-1,) * (len(IMAGE_METRICS) + len(PIXEL_METRICS))

        return tuple(image_scores + pixel_scores)