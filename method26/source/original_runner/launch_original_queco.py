"""제공된 QueCo 원본 run.py를 실행한다. 포팅한 QueCo Trainer는 사용하지 않는다.

원본 폴더에 없는 Dataset·기본 Trainer만 지원 경로에서 불러온다.
원본 runner가 빠뜨린 Dataset 옵션에는 기본값을 보충하며 모델 계산은 바꾸지 않는다.
환경 변수 QUECO_ORIGINAL_SOURCE와 QUECO_SUPPORT_SOURCE로 원본/지원 경로를 지정한다.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import runpy
import sys


def main():
    """원본 모듈을 명시적으로 선택하고 원본 CLI를 실행한다. 반환값은 없다."""
    original = Path(os.environ["QUECO_ORIGINAL_SOURCE"]).resolve()
    support = Path(os.environ["QUECO_SUPPORT_SOURCE"]).resolve()
    # 지원 파일보다 원본 QueCo_lib가 먼저 검색되도록 순서를 고정한다.
    sys.path[:0] = [str(original), str(support)]
    source = original / "trainer_ourzad.py"
    spec = importlib.util.spec_from_file_location("trainer.trainer_ourzad", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    import QueCo_lib
    import datasets.base
    print(f"ORIGINAL_RUNNER={original / 'run.py'}", flush=True)
    print(f"ORIGINAL_TRAINER={module.__file__}", flush=True)
    print(f"ORIGINAL_LIBRARY={QueCo_lib.__file__}", flush=True)
    print(f"SUPPORT_DATASET={datasets.base.__file__}", flush=True)
    # run.py의 CLI에 없는 옵션만 보충한다. 이미 있는 값은 덮어쓰지 않는다.
    parse_args = argparse.ArgumentParser.parse_args

    def parse_with_dataset_defaults(parser, *args, **kwargs):
        """원본 CLI 인자를 읽고 Dataset 생성에 필요한 누락 기본값을 보충한다."""
        result = parse_args(parser, *args, **kwargs)
        defaults = {"crop": None, "masksize": 518, "subtrain": False, "n_abnormal": 5}
        for key, value in defaults.items():
            if not hasattr(result, key):
                setattr(result, key, value)
        return result

    argparse.ArgumentParser.parse_args = parse_with_dataset_defaults
    try:
        sys.argv[0] = str(original / "run.py")
        runpy.run_path(sys.argv[0], run_name="__main__")
    finally:
        argparse.ArgumentParser.parse_args = parse_args


if __name__ == "__main__":
    main()
