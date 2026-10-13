# 한국어 코드 안내: 이 파일은 사전학습 backbone을 이름으로 생성하는 기능을 담당합니다.
# 원본 실행 로직은 변경하지 않고, 초보자용 설명 주석만 추가했습니다.

import torchvision.models as models  # noqa

_BACKBONES = {
    "wideresnet50": lambda: models.wide_resnet50_2(weights="DEFAULT"),
}


def load(name):
    # 역할: `load` 기능을 수행합니다.
    # 매개변수: name.
    # 반환값: 구현에서 계산한 결과 또는 None입니다.
    # 상세 흐름: 주요 호출은 `factory`, `.join`, `ValueError`, `sorted`입니다.
    # 제어 흐름: 예외 발생 경로 1개입니다.
    # 상태 영향: 명시적 self 속성 갱신 없음; return 경로는 1개입니다.
    """한국어 설명:
    역할: 이름에 맞는 사전학습 backbone 모델을 생성합니다.
    매개변수: `name`은 등록된 backbone 이름입니다.
    반환값: 생성되어 초기화된 PyTorch 모델 객체입니다.
    """
    try:
        factory = _BACKBONES[name]
    except KeyError as exc:
        available = ", ".join(sorted(_BACKBONES))
        raise ValueError(f"Unknown backbone '{name}'. Available: {available}") from exc
    return factory()
