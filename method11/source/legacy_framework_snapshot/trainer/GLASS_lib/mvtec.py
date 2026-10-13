from torchvision import transforms
from .perlin import perlin_mask
from enum import Enum
from datasets.mvtec import MVTecDataset as MVT
from datasets.base import DatasetSplit

import numpy as np
import pandas as pd

import PIL
import torch
import os
import glob

_CLASSNAMES = [
    "carpet",
    "grid",
    "leather",
    "tile",
    "wood",
    "bottle",
    "cable",
    "capsule",
    "hazelnut",
    "metal_nut",
    "pill",
    "screw",
    "toothbrush",
    "transistor",
    "zipper",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class MVTecDataset(MVT):
    """
    PyTorch Dataset for MVTec.
    """

    def __init__(
            self,
            source,
            anomaly_source_path='/data1/limited/datasets/eccv24glass/dtd/images',
            dataset_name='mvtec',
            classname='leather',
            resize=288,
            imagesize=288,
            split=DatasetSplit.TRAIN,
            rotate_degrees=0,
            translate=0,
            brightness_factor=0,
            contrast_factor=0,
            saturation_factor=0,
            gray_p=0,
            h_flip_p=0,
            v_flip_p=0,
            distribution=0,
            mean=0.5,
            std=0.1,
            fg=0,
            rand_aug=1,
            scale=0,
            batch_size=8,
            **kwargs,
    ):
        # 역할: 객체를 사용할 수 있도록 필요한 속성과 구성요소를 초기화.
        # 매개변수: source, anomaly_source_path, dataset_name, classname, resize, imagesize, split, rotate_degrees, translate, brightness_factor, contrast_factor, saturation_factor, gray_p, h_flip_p, v_flip_p, distribution, mean, std, fg, rand_aug, scale, batch_size, **kwargs.
        # 반환값: 없음. 초기화가 끝난 객체를 Python이 생성합니다..
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
        super().__init__(source, classname, resize, imagesize, split, **kwargs)
        self.source = source
        self.split = split
        self.batch_size = batch_size
        self.distribution = distribution
        self.mean = mean
        self.std = std
        self.fg = fg
        self.rand_aug = rand_aug
        self.resize = resize if self.distribution != 1 else [resize, resize]
        self.imgsize = imagesize
        self.imagesize = (3, self.imgsize, self.imgsize)
        self.classname = classname
        self.dataset_name = dataset_name

        if self.distribution != 1 and (self.classname == 'toothbrush' or self.classname == 'wood'):
            self.resize = round(self.imgsize * 329 / 288)

        xlsx_path = './datasets/excel/' + self.dataset_name + '_distribution.xlsx'
        if self.fg == 2:  # choose by file
            try:
                df = pd.read_excel(xlsx_path)
                self.class_fg = df.loc[df['Class'] == self.dataset_name + '_' + classname, 'Foreground'].values[0]
            except:
                self.class_fg = 1
        elif self.fg == 1:  # with foreground mask
            self.class_fg = 1
        else:  # without foreground mask
            self.class_fg = 0

        self.imgpaths_per_class, self.data_to_iterate = self._get_image_data()
        self.anomaly_source_paths = sorted(1 * glob.glob(anomaly_source_path + "/*/*.jpg") +
                                           0 * list(next(iter(self.imgpaths_per_class.values())).values())[0])

        self.transform_img = [
            transforms.Resize(self.resize),
            transforms.ColorJitter(brightness_factor, contrast_factor, saturation_factor),
            transforms.RandomHorizontalFlip(h_flip_p),
            transforms.RandomVerticalFlip(v_flip_p),
            transforms.RandomGrayscale(gray_p),
            transforms.RandomAffine(rotate_degrees,
                                    translate=(translate, translate),
                                    scale=(1.0 - scale, 1.0 + scale),
                                    interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.imgsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize(self.resize),
            transforms.CenterCrop(self.imgsize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

    def rand_augmenter(self):
        # 역할: `rand_augmenter`에 해당하는 작업을 수행.
        # 매개변수: 없음.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        list_aug = [
            transforms.ColorJitter(contrast=(0.8, 1.2)),
            transforms.ColorJitter(brightness=(0.8, 1.2)),
            transforms.ColorJitter(saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1),
            transforms.RandomGrayscale(p=1),
            transforms.RandomAutocontrast(p=1),
            transforms.RandomEqualize(p=1),
            transforms.RandomAffine(degrees=(-45, 45)),
        ]
        aug_idx = np.random.choice(np.arange(len(list_aug)), 3, replace=False)

        transform_aug = [
            transforms.Resize(self.resize),
            list_aug[aug_idx[0]],
            list_aug[aug_idx[1]],
            list_aug[aug_idx[2]],
            transforms.CenterCrop(self.imgsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]

        transform_aug = transforms.Compose(transform_aug)
        return transform_aug

    def __getitem__(self, idx):
        # 역할: 요청한 위치의 데이터 항목을 읽어 반환.
        # 매개변수: idx.
        # 반환값: 요청한 index의 데이터 또는 요소입니다..
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        mask_fg = mask_s = aug_image = torch.tensor([1])
        if self.split == DatasetSplit.TRAIN:
            aug = PIL.Image.open(np.random.choice(self.anomaly_source_paths)).convert("RGB")
            if self.rand_aug:
                transform_aug = self.rand_augmenter()
                aug = transform_aug(aug)
            else:
                aug = self.transform_img(aug)

            if self.class_fg:
                fgmask_path = image_path.split(classname)[0] + 'fg_mask/' + classname + '/' + os.path.split(image_path)[-1]
                mask_fg = PIL.Image.open(fgmask_path)
                mask_fg = torch.ceil(self.transform_mask(mask_fg)[0])

            mask_all = perlin_mask(image.shape, self.imgsize // 8, 0, 6, mask_fg, 1)
            mask_s = torch.from_numpy(mask_all[0])
            mask_l = torch.from_numpy(mask_all[1])

            beta = np.random.normal(loc=self.mean, scale=self.std)
            beta = np.clip(beta, .2, .8)
            aug_image = image * (1 - mask_l) + (1 - beta) * aug * mask_l + beta * image * mask_l

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask_gt = PIL.Image.open(mask_path).convert('L')
            mask_gt = self.transform_mask(mask_gt)
        else:
            mask_gt = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "aug": aug_image,
            "mask_s": mask_s,
            "mask_gt": mask_gt,
            "mask": mask_gt,
            "is_anomaly": int(anomaly != "good"),
            "classname": classname,
            "anomaly": anomaly,
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        # 역할: 데이터 또는 컨테이너에 들어 있는 항목 수를 계산.
        # 매개변수: 없음.
        # 반환값: 항목 수를 나타내는 정수입니다..
        return len(self.data_to_iterate)

    #def get_image_data(self):
    #    imgpaths_per_class = {}
    #    maskpaths_per_class = {}

    #    classpath = os.path.join(self.source, self.classname, self.split.value)
    #    maskpath = os.path.join(self.source, self.classname, "ground_truth")
    #    anomaly_types = os.listdir(classpath)

    #    imgpaths_per_class[self.classname] = {}
    #    maskpaths_per_class[self.classname] = {}

    #    for anomaly in anomaly_types:
    #        anomaly_path = os.path.join(classpath, anomaly)
    #        anomaly_files = sorted(os.listdir(anomaly_path))
    #        imgpaths_per_class[self.classname][anomaly] = [os.path.join(anomaly_path, x) for x in anomaly_files]

    #        if self.split == DatasetSplit.TEST and anomaly != "good":
    #            anomaly_mask_path = os.path.join(maskpath, anomaly)
    #            anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
    #            maskpaths_per_class[self.classname][anomaly] = [os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files]
    #        else:
    #            maskpaths_per_class[self.classname]["good"] = None

    #    data_to_iterate = []
    #    for classname in sorted(imgpaths_per_class.keys()):
    #        for anomaly in sorted(imgpaths_per_class[classname].keys()):
    #            for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
    #                data_tuple = [classname, anomaly, image_path]
    #                if self.split == DatasetSplit.TEST and anomaly != "good":
    #                    data_tuple.append(maskpaths_per_class[classname][anomaly][i])
    #                else:
    #                    data_tuple.append(None)
    #                data_to_iterate.append(data_tuple)

    #    return imgpaths_per_class, data_to_iterate
# 한국어 코드 안내: 이 파일은 이미지 데이터셋의 파일 탐색과 sample 생성을 담당합니다.
# 원본 실행 로직은 변경하지 않고, 초보자용 설명 주석만 추가했습니다.
