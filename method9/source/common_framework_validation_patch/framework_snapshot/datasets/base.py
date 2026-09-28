import random
import os
from PIL import Image
from loguru import logger
from enum import Enum
import PIL
import torch
import torch.nn.functional as F
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# /data1/limited/code/sam2_install/sam2/utils/transforms.py 도 동일한 값으로 normalize 함을 확인함

class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class RandomRotation90(torch.nn.Module):
    def __init__(self):
        # 역할: 객체를 사용할 수 있도록 필요한 속성과 구성요소를 초기화.
        # 매개변수: 없음.
        # 반환값: 없음. 초기화가 끝난 객체를 Python이 생성합니다..
        # 상세 흐름: 주요 호출은 `super.__init__`, `super`입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 0개입니다.
        super().__init__()

    def forward(self, img):
        # 역할: 입력을 신경망 계층에 통과시켜 출력 tensor를 계산.
        # 매개변수: img.
        # 반환값: 모델이 계산한 tensor 또는 모델 출력입니다..
        # 상세 흐름: 주요 호출은 `torch.randint.item`, `torch.rot90`, `torch.randint`입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        degree = torch.randint(0, 4, (1,)).item()
        ret = torch.rot90(img, degree, dims=[1, 2])
        return ret


class RandomCropMore(torch.nn.Module):
    def __init__(self, img_size, crop_prob=0.5, crop_range=(0.5, 1.0)):
        # 역할: 객체를 사용할 수 있도록 필요한 속성과 구성요소를 초기화.
        # 매개변수: img_size, crop_prob, crop_range.
        # 반환값: 없음. 초기화가 끝난 객체를 Python이 생성합니다..
        # 상세 흐름: 주요 호출은 `super.__init__`, `super`입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: `self.crop_prob`, `self.crop_range`, `self.img_size`; return 경로는 0개입니다.
        super().__init__()
        self.crop_prob = crop_prob
        self.crop_range = crop_range
        self.img_size = img_size

    def forward(self, img):
        # 역할: 입력을 신경망 계층에 통과시켜 출력 tensor를 계산.
        # 매개변수: img.
        # 반환값: 모델이 계산한 tensor 또는 모델 출력입니다..
        # 상세 흐름: 주요 호출은 `torch.rand.item`, `int`, `transforms.RandomCrop`, `torch.rand`입니다.
        # 제어 흐름: 조건 분기 1개입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 2개입니다.
        prob = torch.rand(1).item()
        if prob > self.crop_prob:
            return img
        crop_size_prob = torch.rand(1).item()
        crop_size = crop_size_prob * \
            (self.crop_range[1]-self.crop_range[0]) + self.crop_range[0]
        crop_size = int(crop_size * self.img_size)
        img = transforms.RandomCrop(crop_size)(img)
        return img


class Crop(torch.nn.Module):
    """Crop transform kept here because dataset cropping is a runtime option."""

    def __init__(self, top, left, height, width):
        # 역할: 객체를 사용할 수 있도록 필요한 속성과 구성요소를 초기화.
        # 매개변수: top, left, height, width.
        # 반환값: 없음. 초기화가 끝난 객체를 Python이 생성합니다..
        # 상세 흐름: 주요 호출은 `super.__init__`, `super`입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: `self.top`, `self.left`, `self.height`, `self.width`; return 경로는 0개입니다.
        super().__init__()
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def forward(self, image):
        # 역할: 입력을 신경망 계층에 통과시켜 출력 tensor를 계산.
        # 매개변수: image.
        # 반환값: 모델이 계산한 tensor 또는 모델 출력입니다..
        # 상세 흐름: 주요 호출은 `transforms.functional.crop`입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        return transforms.functional.crop(
            image, self.top, self.left, self.height, self.width
        )


class BaseDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MVTec.
    """

    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        rotate_degrees=0,
        translate=0,
        brightness_factor=0,
        contrast_factor=0,
        saturation_factor=0,
        gray_p=0,
        h_flip_p=0,
        v_flip_p=0,
        scale=0,
        subtest: bool = False,
        **kwargs,
    ):
        # 역할: 객체를 사용할 수 있도록 필요한 속성과 구성요소를 초기화.
        # 매개변수: source, classname, resize, imagesize, split, train_val_split, rotate_degrees, translate, brightness_factor, contrast_factor, saturation_factor, gray_p, h_flip_p, v_flip_p, scale, subtest, **kwargs.
        # 반환값: 없음. 초기화가 끝난 객체를 Python이 생성합니다..
        # 상세 흐름: 주요 호출은 `super.__init__`, `isinstance`, `self.get_image_data`, `transforms.Compose`, `transforms.Resize`입니다.
        # 제어 흐름: 조건 분기 1개입니다.
        # 상태 영향: `self.root`, `self.source`, `self.split`, `self.train_val_split`; return 경로는 0개입니다.
        """
        Args:
            source: [str]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.root = self.source = source
        self.split = split
        self.train_val_split = train_val_split
        # train/validation 분할이 매 실행마다 달라지지 않도록 전달받은 seed를 보관합니다.
        # 자식 Dataset은 이 값을 이용해 같은 이미지 목록을 재현 가능하게 나눕니다.
        self.seed = kwargs.get("seed", 0)
        self.transform_std = IMAGENET_STD
        self.transform_mean = IMAGENET_MEAN
        self.args = kwargs['args']
        if isinstance(classname, list):
            self.classnames_to_use = classname
        else:
            self.classnames_to_use = [classname]
        self.few_shot_mode = False
        # 자식 Dataset이 파일 구조를 탐색해 [class, anomaly, image_path, mask_path] 목록을 만듭니다.
        # 이후 __getitem__은 이 목록의 한 항목을 실제 tensor dict로 바꿉니다.
        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data(
            subtest)

        """
        brightness_factor 0.0
        contrast_factor 0.0
        saturation_factor 0.0
        h_flip_p 0.0
        v_flip_p 0.0
        gray_p 0.0
        rotate_degrees 0
        translate 0
        scale 0.0
        """
        # 주의: 아래 주석 처리된 augmentation은 현재 실행되지 않습니다.
        # 활성화된 경로는 crop(선택) → resize → tensor 변환 → ImageNet 정규화입니다.
        self.transform_img = [
            Crop(*self.args.crop) if self.args.crop is not None else lambda x: x,
            transforms.Resize((resize, resize)),
            # transforms.RandomRotation(rotate_degrees, transforms.InterpolationMode.BILINEAR),
            #transforms.ColorJitter(
            #    brightness_factor, contrast_factor, saturation_factor),
            #transforms.RandomHorizontalFlip(h_flip_p),
            #transforms.RandomVerticalFlip(v_flip_p),
            #transforms.RandomGrayscale(gray_p),
            #transforms.RandomAffine(rotate_degrees,
            #                        translate=(translate, translate),
            #                        scale=(1.0-scale, 1.0+scale),
            #                        interpolation=transforms.InterpolationMode.BILINEAR),
            #transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            # Ground-truth mask is categorical (normal=0, defect=1). NEAREST
            # preserves its boundary labels; bilinear would create fractional
            # values that __getitem__ subsequently turns into defect pixels.
            transforms.Resize(
                (self.args.masksize, self.args.masksize),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            #transforms.CenterCrop(self.args.masksize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.transform_fg = transforms.Compose(
            [
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            ]
        )

        self.imagesize = (3, imagesize, imagesize)

    def having_mask(self):
        #logger.info("마스크를 사용하지 않도록 되어 있습니다")
        # 역할: `having_mask`에 해당하는 작업을 수행.
        # 매개변수: 없음.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 상세 흐름: 주요 호출은 직접 호출하는 다른 함수 없음입니다.
        # 제어 흐름: 반복문 1개, 조건 분기 1개입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 2개입니다.
        for c, a, i, m in self.data_to_iterate:
            if m is not None:
                return True
        return False

    def _gettext(self, path):
        #text = '_'.join(path.split('/')[-4:]) # NOTE 이것이 더 말이 됨. like 'bottle_train_good_0001.png'
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: path.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 상세 흐름: 주요 호출은 `.split`, `path.split`입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        text = path.split("/")[-1].split('.')[0][-3:] # '0001'
        return text

    def __getitem__(self, idx):
        # 역할: 요청한 위치의 데이터 항목을 읽어 반환.
        # 매개변수: idx.
        # 반환값: 요청한 index의 데이터 또는 요소입니다..
        # 상세 흐름: 주요 호출은 `PIL.Image.open.convert`, `self.transform_img`, `self._gettext`, `os.path.join`, `int`입니다.
        # 제어 흐름: 조건 분기 4개입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        # 파일 목록의 메타데이터를 꺼내 실제 이미지와 정답 mask를 읽습니다.
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        text = self._gettext(image_path)

        # 모든 방법이 공통으로 사용할 수 있도록 sample을 dict 형식으로 통일합니다.
        # is_anomaly는 image-level 평가, mask는 pixel-level 평가에 쓰입니다.
        ret = {
                "image": image,
                "classname": classname,
                "anomaly": anomaly,
                "is_anomaly": int(anomaly != "good"),
                "image_name": "/".join(image_path.split("/")[-4:]),
                "image_path": image_path,
                "text": text,
            }

        # mask loading
        # 정상 이미지 또는 train split에는 정답 결함 위치가 없으므로 all-zero mask를 사용합니다.
        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            if len(mask.mode) > 1: # 만약 mask 가 1차원이 아니면, 1차원으로 변경
                mask = mask.convert("L")
            mask = self.transform_mask(mask)
            mask[mask != 0] = 1.0 # bilinear interpolation 으로 인해 0이 아닌 값이 나올 수 있음
        else:
            mask = torch.zeros(1, self.args.masksize, self.args.masksize)
        ret['mask'] = mask

        # fg loading
        fgmask_path = os.path.join(image_path.split(classname)[0], classname, 'fg_mask', os.path.split(image_path)[-1])
        if self.split == DatasetSplit.TRAIN and os.path.exists(fgmask_path):
            fg = PIL.Image.open(fgmask_path)
            if len(fg.mode) > 1: # 만약 fg 가 1차원이 아니면, 1차원으로 변경
                fg = fg.convert("L")
            fg = self.transform_fg(fg)
            fg[fg != 0] == 1.0 # bilinear interpolation 으로 인해 0이 아닌 값이 나올 수 있음
        else:
            fg = torch.ones(1, self.args.masksize, self.args.masksize) # fg 가 없으면 모두 1임

        ret['fg'] = fg

        return ret

    def __len__(self):
        # 역할: 데이터 또는 컨테이너에 들어 있는 항목 수를 계산.
        # 매개변수: 없음.
        # 반환값: 항목 수를 나타내는 정수입니다..
        # 상세 흐름: 주요 호출은 `len`입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        return len(self.data_to_iterate)

    def get_image_data(self, subtest: bool = False):
        # 역할: 요청한 내부 정보 또는 계산 결과를 가져옴.
        # 매개변수: subtest.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 상세 흐름: 주요 호출은 `self._get_image_data`, `random.shuffle`, `print`, `data_to_iterate2_wo_good.append`입니다.
        # 제어 흐름: 반복문 1개, 조건 분기 2개입니다.
        # 상태 영향: `self.split`; return 경로는 2개입니다.
        if True:
            _, data_to_iterate = self._get_image_data(subtest)
            return None, data_to_iterate
        else:
            _, data_to_iterate1 = self._get_image_data(subtest)
            self.split = DatasetSplit.TEST
            _, data_to_iterate2 = self._get_image_data(subtest)

            data_to_iterate2_wo_good = []
            for data in data_to_iterate2:
                if data[1] != "good":
                    data_to_iterate2_wo_good.append(data)

            # shuffle data_to_iterate2_wo_good
            random.shuffle(data_to_iterate2_wo_good)

            data_to_iterate = data_to_iterate1 + \
                data_to_iterate2_wo_good[:self.args.n_abnormal]
            print(data_to_iterate)
            return None, data_to_iterate

    @classmethod
    def get_classname(cls):
        # 역할: 요청한 내부 정보 또는 계산 결과를 가져옴.
        # 매개변수: 없음.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 상세 흐름: 주요 호출은 직접 호출하는 다른 함수 없음입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        return cls._CLASSNAMES
# 한국어 코드 안내: 이 파일은 이 모듈에 포함된 기능의 구현을 담당합니다.
# 원본 실행 로직은 변경하지 않고, 초보자용 설명 주석만 추가했습니다.


