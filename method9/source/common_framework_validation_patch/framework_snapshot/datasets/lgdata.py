"""
Dataset folder structure:
./TargetTVTBalanced/
├── mean.npy
├── std.npy
├── test
│   ├── 0
│   ├── 1
│   ├── 2
│   └── 3
├── train
│   ├── 0
│   ├── 1
│   ├── 2
│   └── 3
└── val
    ├── 0
    ├── 1
    ├── 2
    └── 3
"""
import os
import json
# main.py 에서 사용됨
from datasets.base import BaseDataset, DatasetSplit


class LGDataset(BaseDataset):

    def _get_image_data(self, subtest: bool = False):
        # 역할: 공개 함수가 사용하는 내부 보조 작업을 수행.
        # 매개변수: subtest.
        # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
        # 상세 흐름: 주요 호출은 `os.path.join`, `open`, `json.load`, `sorted`, `str`입니다.
        # 제어 흐름: 반복문 5개, 조건 분기 3개, 자원/문맥 관리 1개, 예외 발생 경로 1개입니다.
        # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 2개입니다.
        data_to_iterate = []
        for dataset_name in self.classnames_to_use:
            """
            self.split == 'train' or 'test'
            """
            with open(self.root, 'r') as f:
                json_data = json.load(f)

            data_path = json_data['dataset_list'][dataset_name]
            class_categories = json_data['class_categories'][dataset_name]

            split_path = os.path.join(data_path, self.split.value)

            def make_data_to_iterate(split_path, class_, anomaly: bool):
                # 역할: `make_data_to_iterate`에 해당하는 작업을 수행.
                # 매개변수: split_path, class_, anomaly.
                # 반환값: 구현에서 계산한 결과이며, return 문이 없으면 None입니다..
                # 상세 흐름: 주요 호출은 `os.path.join`, `sorted`, `str`, `os.listdir`, `data_to_iterate.append`입니다.
                # 제어 흐름: 반복문 1개, 조건 분기 1개입니다.
                # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
                data_to_iterate = []
                class_path = os.path.join(split_path, str(class_))
                imgpaths = sorted(os.listdir(class_path))

                anomaly_text = "defect" if anomaly else "good"
                for imgpath in imgpaths:
                    data_to_iterate.append(
                        [dataset_name, anomaly_text, os.path.join(class_path, imgpath), None])

                if subtest:
                    data_to_iterate = data_to_iterate[:5]

                return data_to_iterate

            if self.split.value == "train":
                for c in class_categories['OK']:
                    data_to_iterate.extend(
                        make_data_to_iterate(split_path, c, False))
            elif self.split.value == "test" or self.split.value == "val":
                try:
                    for c in class_categories['OK']:
                        data_to_iterate.extend(
                            make_data_to_iterate(split_path, c, False))
                    for c in class_categories['NG']:
                        data_to_iterate.extend(
                            make_data_to_iterate(split_path, c, True))
                except:
                    pass
            else:
                raise NotImplementedError

        return None, data_to_iterate
# 한국어 코드 안내: 이 파일은 이 모듈에 포함된 기능의 구현을 담당합니다.
# 원본 실행 로직은 변경하지 않고, 초보자용 설명 주석만 추가했습니다.
