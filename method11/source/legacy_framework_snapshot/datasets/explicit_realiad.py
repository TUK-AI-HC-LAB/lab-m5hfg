import os
import os.path as osp
import json
import random
import torch
import PIL.Image
from PIL import Image
from torchvision import transforms

from .base import BaseDataset, DatasetSplit, IMAGENET_MEAN, IMAGENET_STD

class ExplicitDataset(BaseDataset):
    """
    PyTorch Dataset for Real-IAD style explicit dataset.
    meta_file: json or list of jsons (Real-IAD style)
    image_dir: prefix for image paths (optional, can be included in meta)
    split: DatasetSplit.TRAIN or TEST
    """

    def __init__(
        self,
        meta_file,
        image_dir=None,
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
        subtest=False,
        **kwargs,
    ):
        # Set attributes needed by get_image_data which will be called by the parent class's __init__
        # 역할: 객체를 사용할 수 있도록 필요한 속성과 구성요소를 초기화.
        # 매개변수: meta_file, image_dir, resize, imagesize, split, train_val_split, rotate_degrees, translate, brightness_factor, contrast_factor, saturation_factor, gray_p, h_flip_p, v_flip_p, scale, subtest, **kwargs.
        # 반환값: 없음. 초기화가 끝난 객체를 Python이 생성합니다..
        # 상세 흐름: 주요 호출은 `kwargs.get`, `super.__init__`, `isinstance`, `super`입니다.
        # 제어 흐름: 별도의 반복·조건 분기 없음입니다.
        # 상태 영향: `self.args`, `self.meta_file`, `self.image_dir`; return 경로는 0개입니다.
        self.args = kwargs.get('args', None)
        self.meta_file = meta_file if isinstance(meta_file, list) else [meta_file]
        self.image_dir = image_dir
        
        # Call parent's init with our customized source
        super().__init__(
            source=image_dir if image_dir is not None else "",
            classname=None,  # We'll handle classes through the meta files
            resize=resize,
            imagesize=imagesize,
            split=split,
            train_val_split=train_val_split,
            rotate_degrees=rotate_degrees,
            translate=translate,
            brightness_factor=brightness_factor,
            contrast_factor=contrast_factor,
            saturation_factor=saturation_factor,
            gray_p=gray_p,
            h_flip_p=h_flip_p,
            v_flip_p=v_flip_p,
            scale=scale,
            subtest=subtest,
            **kwargs,
        )

    def get_image_data(self, subtest: bool = False):
        # 역할: 요청한 내부 정보 또는 계산 결과를 가져옴.
        # 매개변수: subtest.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 상세 흐름: 주요 호출은 `hasattr`, `random.sample`, `open`, `json.load`, `range`입니다.
        # 제어 흐름: 반복문 3개, 조건 분기 9개, 자원/문맥 관리 1개입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        """
        Override BaseDataset's get_image_data method to load from our meta file format.
        Returns a tuple of (imgpaths_per_class, data_to_iterate)
        """
        data_to_iterate = []
        imgpaths_per_class = {}
        
        for path in self.meta_file:
            with open(path, 'r') as fp:
                info = json.load(fp)
                meta = info['meta']
                train = info['train']
                test = info['test']
                samples = train if self.split == DatasetSplit.TRAIN else test
                normal_class = meta['normal_class']
                
                # 기본 디렉토리 경로
                base_dir = self.image_dir
                
                if self.split == DatasetSplit.TRAIN:
                    for sample in samples:
                        category = sample['category']
                        
                        # JSON에 있는 상대 경로를 그대로 사용
                        rel_path = sample['image_path']
                        
                        # 특정 카테고리에 대해 _realiad 접미사 추가
                        folder_category = f"{category}_realiad" if category in ["zipper", "toothbrush"] else category
                        
                        # 최종 경로 생성: base_dir/folder_category/rel_path
                        image_path = os.path.join(base_dir, folder_category, rel_path)
                        
                        # 파일 존재 확인
                        if not os.path.exists(image_path):
                            print(f"Warning: File not found - {image_path}")
                            print(f"Original path in JSON: {rel_path}")
                            continue  # 파일이 없으면 건너뜀
                        
                        # Group images by class for the return value
                        if category not in imgpaths_per_class:
                            imgpaths_per_class[category] = []
                        imgpaths_per_class[category].append(image_path)
                        
                        data_tuple = [
                            category,                    # classname
                            "good",                      # anomaly (train is always "good")
                            image_path,                  # image_path
                            None                         # mask_path (None for train)
                        ]
                        data_to_iterate.append(data_tuple)
                else:
                    for sample in samples:
                        category = sample['category']
                        
                        # JSON에 있는 상대 경로를 그대로 사용
                        rel_path = sample['image_path']
                        
                        # 특정 카테고리에 대해 _realiad 접미사 추가
                        folder_category = f"{category}_realiad" if category in ["zipper", "toothbrush"] else category
                        
                        # 최종 경로 생성: base_dir/folder_category/rel_path
                        image_path = os.path.join(base_dir, folder_category, rel_path)
                        
                        # 정상/비정상 판단
                        is_normal = (sample['mask_path'] is None or sample['anomaly_class'] == normal_class)
                        anomaly_class = "good" if is_normal else sample['anomaly_class']
                        
                        # 마스크 경로 처리
                        mask_path = None
                        if not is_normal and sample['mask_path'] is not None:
                            mask_rel_path = sample['mask_path']
                            mask_path = os.path.join(base_dir, folder_category, mask_rel_path)
                            
                            # 마스크 파일 존재 확인
                            if not os.path.exists(mask_path):
                                print(f"Warning: Mask file not found - {mask_path}")
                                print(f"Original mask path in JSON: {mask_rel_path}")
                                mask_path = None
                        
                        # 파일 존재 확인
                        if not os.path.exists(image_path):
                            print(f"Warning: File not found - {image_path}")
                            print(f"Original path in JSON: {rel_path}")
                            continue  # 파일이 없으면 건너뜀
                        
                        # Group images by class for the return value
                        if category not in imgpaths_per_class:
                            imgpaths_per_class[category] = []
                        imgpaths_per_class[category].append(image_path)
                        
                        data_tuple = [
                            category,                    # classname
                            anomaly_class,               # anomaly
                            image_path,                  # image_path
                            mask_path                    # mask_path
                        ]
                        data_to_iterate.append(data_tuple)
        
        # Handle subtest option (only use 1% of the data)
        if subtest:
            data_to_iterate = data_to_iterate[:int(len(data_to_iterate) * 0.01)]
        
        # Handle subtrain option if available (only use 5% of the train data)
        if hasattr(self.args, 'subtrain') and self.args.subtrain and self.split == DatasetSplit.TRAIN:
            indices = random.sample(range(len(data_to_iterate)), int(len(data_to_iterate) * 0.05))
            data_to_iterate = [data_to_iterate[i] for i in indices]

        return imgpaths_per_class, data_to_iterate

    def __getitem__(self, idx):
        # 역할: 요청한 위치의 데이터 항목을 읽어 반환.
        # 매개변수: idx.
        # 반환값: 요청한 index의 데이터 또는 요소입니다..
        # 상세 흐름: 주요 호출은 `self._get_item_impl`, `print`, `self.__getitem__`, `len`입니다.
        # 제어 흐름: 조건 분기 1개입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 3개입니다.
        try:
            return self._get_item_impl(idx)
        except (OSError, PIL.UnidentifiedImageError) as e:
            print(f"Skipping corrupted image at index {idx}: {e}")
            # If we're at the last item, try the first one
            if idx >= len(self.data_to_iterate) - 1:
                return self._get_item_impl(0)
            # Otherwise try the next item
            return self.__getitem__(idx + 1)
        
    def _get_item_impl(self, idx):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: idx.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 상세 흐름: 주요 호출은 `torch.ones`, `PIL.Image.open`, `self.transform_img`, `int`, `.join`입니다.
        # 제어 흐름: 조건 분기 5개, 예외 발생 경로 1개입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        
        try:
            # Load and transform image
            image_obj = PIL.Image.open(image_path)
            try:
                image = image_obj.convert("RGB")
            except Exception as e:
                print(f"Error converting image {image_path}: {e}")
                # Try to use the image as is if conversion fails
                image = image_obj
            image = self.transform_img(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            raise  # Re-raise to be caught by __getitem__
        
        # Get text identifier
        text = image_path.split("/")[-1].split('.')[0][-3:]  # e.g., '001' from filename
        
        # Prepare base return dictionary
        ret = {
            "image": image,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
            "text": text,
        }
        
        # Handle mask loading for test images with anomalies
        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            if len(mask.mode) > 1:  # If mask is not single-channel, convert it
                mask = mask.convert("L")
            mask = self.transform_mask(mask)
            mask[mask != 0] = 1.0  # Binarize mask (anything non-zero becomes 1.0)
        else:
            mask_size = self.args.masksize if hasattr(self.args, "masksize") else 224
            mask = torch.zeros(1, mask_size, mask_size)
        
        ret['mask'] = mask
        
        mask_size = self.args.masksize if hasattr(self.args, "masksize") else 224
        fg = torch.ones(1, mask_size, mask_size)

        # Handle foreground mask loading if enabled
        if hasattr(self.args, "fg_mask") and self.args.fg_mask:
            fgmask_path = os.path.join(image_path.split(classname)[0], 
                                      f"{classname}_realiad" if classname in ["zipper", "toothbrush"] else classname, 
                                      'fg_mask', os.path.split(image_path)[-1])
            if self.split == DatasetSplit.TRAIN and os.path.exists(fgmask_path):
                fg = PIL.Image.open(fgmask_path)
                if len(fg.mode) > 1:  # If foreground mask is not single-channel, convert it
                    fg = fg.convert("L")
                fg = self.transform_fg(fg)
                fg[fg != 0] = 1.0  # Binarize foreground mask
        ret['fg'] = fg
        
        return ret
# 한국어 코드 안내: 이 파일은 이 모듈에 포함된 기능의 구현을 담당합니다.
# 원본 실행 로직은 변경하지 않고, 초보자용 설명 주석만 추가했습니다.
