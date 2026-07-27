"""PatchCore의 핵심 아이디어를 작은 숫자 예제로 보여 주는 학습용 코드.

실제 PatchCore처럼 CNN으로 이미지 특징을 뽑지는 않는다. 대신 이미 뽑혀 있다고
가정한 정상 patch 특징을 사용해, coreset 선택과 최근접 거리 이상 점수 계산을 직접 구현한다.
"""

from __future__ import annotations

import numpy as np


def farthest_first_coreset(features: np.ndarray, count: int) -> np.ndarray:
    """서로 멀리 떨어진 정상 특징을 차례로 골라 대표 normal memory를 만든다."""
    chosen = [0]
    nearest_distance = np.linalg.norm(features - features[0], axis=1)

    while len(chosen) < count:
        next_index = int(np.argmax(nearest_distance))
        chosen.append(next_index)
        distance_to_new = np.linalg.norm(features - features[next_index], axis=1)
        nearest_distance = np.minimum(nearest_distance, distance_to_new)
    return features[chosen]


def anomaly_scores(test_features: np.ndarray, normal_memory: np.ndarray) -> np.ndarray:
    """각 테스트 patch와 가장 가까운 정상 patch의 거리를 이상 점수로 반환한다."""
    distances = np.linalg.norm(
        test_features[:, None, :] - normal_memory[None, :, :], axis=2
    )
    return distances.min(axis=1)


def main() -> None:
    # 2차원 좌표는 실제 이미지 특징 벡터를 아주 단순하게 표현한 예시다.
    normal_features = np.array([
        [0.0, 0.0], [0.1, 0.2], [0.2, 0.1], [0.9, 0.8], [1.0, 1.0], [0.8, 1.1],
    ])
    test_features = np.array([
        [0.15, 0.10],  # 정상 무리와 가까운 조각
        [0.85, 0.95],  # 정상 무리와 가까운 조각
        [2.20, 2.00],  # 정상 무리와 먼 조각: 이상 후보
    ])

    normal_memory = farthest_first_coreset(normal_features, count=3)
    scores = anomaly_scores(test_features, normal_memory)

    print("대표 normal memory:")
    print(normal_memory)
    print("\n테스트 patch별 이상 점수:")
    for index, score in enumerate(scores):
        print(f"patch {index}: {score:.3f}")
    print(f"\n가장 이상한 patch: {int(np.argmax(scores))}")


if __name__ == "__main__":
    main()
