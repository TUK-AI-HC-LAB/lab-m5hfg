import os
import random
from enum import Enum
from datasets.base import BaseDataset, DatasetSplit

_CLASSNAMES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


class MVTecDataset(BaseDataset):
    def _get_image_data(self, subtest: bool = False):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: subtest.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 상세 흐름: 주요 호출은 `sorted`, `os.path.join`, `os.listdir`, `imgpaths_per_class.keys`, `random.sample`입니다.
        # 제어 흐름: 반복문 5개, 조건 분기 3개입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            # MVTec의 train 폴더에는 정상(good) 이미지만 있습니다.
            # TRAIN과 VAL은 이 정상 폴더를 서로 겹치지 않게 나눠 쓰고,
            # 실제 결함이 포함된 test 폴더는 TEST일 때만 읽습니다.
            split_folder = "test" if self.split == DatasetSplit.TEST else "train"
            classpath = os.path.join(self.source, classname, split_folder)
            maskpath = os.path.join(self.source, classname, "ground_truth")
            anomaly_types = os.listdir(classpath)

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                # validation은 정상 이미지로만 구성합니다. 표준 MVTec train에는
                # good만 있지만, 사용자 데이터에 다른 폴더가 있어도 섞이지 않게 막습니다.
                if self.split in (DatasetSplit.TRAIN, DatasetSplit.VAL) and anomaly != "good":
                    continue
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = sorted(os.listdir(anomaly_path))

                # 정상 train 이미지를 고정 seed로 train/validation에 분리합니다.
                # train_val_split=0.9이면 90%는 학습, 나머지 10%는 validation입니다.
                # 1.0은 기존처럼 모든 정상 이미지를 학습에 쓰고 validation은 비웁니다.
                if self.split in (DatasetSplit.TRAIN, DatasetSplit.VAL):
                    if not 0 < self.train_val_split <= 1.0:
                        raise ValueError(
                            "train_val_split must be in (0, 1]. "
                            f"Received {self.train_val_split}."
                        )
                    if self.train_val_split < 1.0 and len(anomaly_files) > 1:
                        shuffled_files = list(anomaly_files)
                        split_rng = random.Random(
                            f"{self.seed}:{classname}:{anomaly}"
                        )
                        split_rng.shuffle(shuffled_files)
                        split_index = int(len(shuffled_files) * self.train_val_split)
                        # 아주 작은 클래스에서도 train과 validation이 모두 비지 않게 합니다.
                        split_index = max(1, min(len(shuffled_files) - 1, split_index))
                        if self.split == DatasetSplit.TRAIN:
                            anomaly_files = sorted(shuffled_files[:split_index])
                        else:
                            anomaly_files = sorted(shuffled_files[split_index:])
                    elif self.split == DatasetSplit.VAL:
                        anomaly_files = []
                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, x) for x in anomaly_files
                ]

                print(f"Class: {classname}, Anomaly: {anomaly}, # of images: {len(imgpaths_per_class[classname][anomaly])} from {len(anomaly_files)}")

                anomaly_mask_path = os.path.join(maskpath, anomaly)
                if self.split == DatasetSplit.TEST and anomaly != "good" and os.path.exists(anomaly_mask_path): # mask 없으면 건너뜀
                    anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                    maskpaths_per_class[classname][anomaly] = [
                        os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                    ]
                else:
                    maskpaths_per_class[classname]["good"] = None

        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                data_to_iterate_class = []
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    try: # mask 가 있을 때
                        data_tuple.append(
                            maskpaths_per_class[classname][anomaly][i])
                    except:
                        data_tuple.append(None)
                    data_to_iterate_class.append(data_tuple)

                if subtest:
                    data_to_iterate_class = data_to_iterate_class[:int(len(data_to_iterate_class) * 0.01)]
                data_to_iterate.extend(data_to_iterate_class)
        
        if self.args.subtrain and self.split == DatasetSplit.TRAIN: # training data 를 1/20 만 사용함
            indices = random.sample(range(len(data_to_iterate)), int(len(data_to_iterate) * 0.05))
            data_to_iterate = [data_to_iterate[i] for i in indices]

        return imgpaths_per_class, data_to_iterate
# 한국어 코드 안내: 이 파일은 이미지 데이터셋의 파일 탐색과 sample 생성을 담당합니다.
# 원본 실행 로직은 변경하지 않고, 초보자용 설명 주석만 추가했습니다.
