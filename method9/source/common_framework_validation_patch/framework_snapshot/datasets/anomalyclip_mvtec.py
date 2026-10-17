"""AnomalyCLIP 공식 평가 전처리를 사용하는 MVTec Dataset variant입니다.

파일 탐색, sample dictionary, mask 처리 방식은 공통 MVTecDataset을 그대로
상속합니다. 차이는 image와 mask가 Resize(short edge) 뒤 CenterCrop을 거친다는
점입니다. 이 순서는 공식 AnomalyCLIP image_transform과 맞춥니다.
"""

from __future__ import annotations

from torchvision import transforms

from datasets.mvtec import DatasetSplit, MVTecDataset as BaseMVTecDataset
from datasets.base import IMAGENET_MEAN, IMAGENET_STD


class MVTecDataset(BaseMVTecDataset):
    """공식 AnomalyCLIP 공간 전처리를 공통 sample contract로 제공하는 Dataset입니다."""

    def __init__(
        self,
        source,
        classname,
        resize,
        imagesize,
        split,
        **kwargs,
    ):
        """공통 MVTecDataset을 초기화한 뒤 AnomalyCLIP transform으로 교체합니다.

        Parameters
        ----------
        source:
            MVTec root directory입니다.
        classname:
            현재 category 이름입니다.
        resize, imagesize:
            공통 constructor 호환 인자입니다. 공식 AnomalyCLIP은 imagesize를
            shortest-edge resize와 center crop의 target size로 사용합니다.
        split:
            train, validation, test split 정보입니다.
        kwargs:
            공통 args를 포함한 추가 설정입니다.
        """

        # 역할: 공통 file layout을 유지하면서 AnomalyCLIP용 transform만 설정합니다.
        # 매개변수: source, classname, resize, imagesize, split, **kwargs.
        # 반환값: 없음. 초기화된 MVTecDataset 객체가 생성됩니다.
        super().__init__(source, classname, resize, imagesize, split, **kwargs)
        args = kwargs["args"]
        # 공식 image_transform(is_train=False): Resize(short edge) → CenterCrop → RGB → tensor → normalize.
        # BaseDataset은 PIL RGB image를 이미 전달하므로 RGB 변환은 여기서 중복하지 않습니다.
        self.transform_img = transforms.Compose(
            [
                transforms.Resize(
                    args.imagesize,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(args.imagesize),
                transforms.ToTensor(),
                # Adapter가 이 ImageNet normalization을 CLIP normalization으로 변환합니다.
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        # image와 같은 공간 변환을 적용해야 pixel ground truth mask와 map이 정렬됩니다.
        self.transform_mask = transforms.Compose(
            [
                # Keep categorical ground-truth labels at the resize boundary.
                transforms.Resize(
                    args.masksize,
                    interpolation=transforms.InterpolationMode.NEAREST,
                ),
                transforms.CenterCrop(args.masksize),
                transforms.ToTensor(),
            ]
        )
