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
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, "train" if self.split ==
                                     DatasetSplit.TRAIN else "test")
            maskpath = os.path.join(self.source, classname, "ground_truth")
            anomaly_types = os.listdir(classpath)

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = sorted(os.listdir(anomaly_path))
                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, x) for x in anomaly_files
                ]

                print(f"Class: {classname}, Anomaly: {anomaly}, # of images: {len(imgpaths_per_class[classname][anomaly])} from {len(anomaly_files)}")

                #if self.train_val_split < 1.0:
                #    n_images = len(imgpaths_per_class[classname][anomaly])
                #    train_val_split_idx = int(n_images * self.train_val_split)
                #    if self.split == DatasetSplit.TRAIN:
                #        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                #            classname
                #        ][anomaly][:train_val_split_idx]
                #    elif self.split == DatasetSplit.VAL:
                #        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                #            classname
                #        ][anomaly][train_val_split_idx:]

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
