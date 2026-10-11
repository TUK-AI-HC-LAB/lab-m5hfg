"""기본 인자값과 추론 설정."""
from dataclasses import dataclass, field

# args에 없을 때만 채워 넣는 기본값
DEFAULT_ARGS = dict(
    sigma=4.0,                       # 패치 맵 가우시안 스무딩
    topk_ratio=0.02,                 # 이미지 점수에 쓰는 상위 픽셀 비율

    save_vis=True,                   # 오버레이/히트맵 png 저장
    vis_dir="./anomaly_map_results",
    use_text_branch=True,
    T_layer=1.0,
    T_branch=1.0,
    lambda_gate=0.5,
    gamma_gate=4.0,
    alpha_imgscore=0.4,              # 이미지 점수 혼합비 (텍스트 쪽 비중)

    # OT 매칭
    ot_eps=0.05,                     # 엔트로피 정규화
    ot_iters=30,
    ot_ws=1.0,                       # 내용(유사도) 가중
    ot_wp=0.05,                      # 위치 제약 가중
    ot_wo=0.05,                      # objectness 차이 가중
    ot_wb=0.10,                      # 전경/배경 불일치 페널티
)

ABLATION_MODES = ("all", "ot", "mutual")


def apply_default_args(args):
    """args에 없는 항목만 DEFAULT_ARGS 값으로 채운다."""
    for key, value in DEFAULT_ARGS.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


@dataclass(frozen=True)
class InferenceConfig:
    """predict()가 쓰는 설정을 args에서 한 번만 읽어 모아 둔 것."""

    # 입력 / 특징
    imagesize: int
    features_list: list

    # 시각 브랜치 (패치 매칭)
    ref_topR: int                    # 타깃당 참조 이웃 이미지 수
    neighbor_chunk: int              # GPU에 한 번에 올리는 이웃 수
    mutual_tau: float
    mutual_topk: int
    mutual_local_radius: int
    transistor_mutual_radius: int    # transistor 클래스 전용 탐색 반경
    support_m: int
    agg_q: float
    ablation_mode: str               # "all" | "ot" | "mutual"
    ot: dict = field(default_factory=dict)

    # 맵 후처리 / 융합
    sigma: float = 4.0
    T_layer: float = 1.0
    T_branch: float = 1.0
    use_text_branch: bool = True
    fusion_mode: str = "entropy"
    fixed_visual_weight: float = 0.5

    # objectness
    obj_tau: float = 0.5
    use_objectness_gate: bool = True
    gamma_gate: float = 4.0
    lambda_gate: float = 0.5

    # 이미지 점수
    topk_ratio: float = 0.02
    alpha_imgscore: float = 0.4

    # 저장 / 진단
    save_vis: bool = False
    vis_dir: str = "./anomaly_map_results"
    save_obj_vis: bool = False
    fusion_diagnostics: bool = True

    @classmethod
    def from_args(cls, args):
        ablation_mode = str(getattr(args, "ablation_mode", "all")).lower()
        if ablation_mode not in ABLATION_MODES:
            raise ValueError(f"Unknown ablation_mode: {ablation_mode}")

        obj_tau = float(getattr(args, "obj_tau", 0.5))
        ot = dict(
            ws=float(getattr(args, "ot_ws", 1.0)),
            wp=float(getattr(args, "ot_wp", 0.05)),
            wo=float(getattr(args, "ot_wo", 0.05)),
            wb=float(getattr(args, "ot_wb", 0.10)),
            obj_tau=obj_tau,
            eps=float(getattr(args, "ot_eps", 0.05)),
            iters=int(getattr(args, "ot_iters", 30)),
        )

        return cls(
            imagesize=args.imagesize,
            features_list=args.features_list,

            ref_topR=int(getattr(args, "ref_topR", 10)),
            neighbor_chunk=int(getattr(args, "neighbor_chunk", 8)),
            mutual_tau=float(getattr(args, "mutual_tau", 0.85)),
            mutual_topk=int(getattr(args, "mutual_topk", 5)),
            mutual_local_radius=int(getattr(args, "mutual_local_radius", 0)),
            transistor_mutual_radius=int(getattr(args, "transistor_mutual_radius", 1)),
            support_m=int(getattr(args, "support_m", 5)),
            agg_q=float(getattr(args, "agg_percentile_q", 0.85)),
            ablation_mode=ablation_mode,
            ot=ot,

            sigma=float(args.sigma),
            T_layer=args.T_layer,
            T_branch=args.T_branch,
            use_text_branch=bool(args.use_text_branch),
            fusion_mode=str(getattr(args, "fusion_mode", "entropy")).lower(),
            fixed_visual_weight=float(getattr(args, "fixed_visual_weight", 0.5)),

            obj_tau=obj_tau,
            use_objectness_gate=bool(getattr(args, "use_objectness_gate", True)),
            gamma_gate=float(args.gamma_gate),
            lambda_gate=float(args.lambda_gate),

            topk_ratio=float(args.topk_ratio),
            alpha_imgscore=float(args.alpha_imgscore),

            save_vis=bool(getattr(args, "save_vis", False)),
            vis_dir=getattr(args, "vis_dir", "./anomaly_map_results"),
            save_obj_vis=bool(getattr(args, "save_obj_vis", False)),
            fusion_diagnostics=bool(getattr(args, "fusion_diagnostics", True)),
        )
