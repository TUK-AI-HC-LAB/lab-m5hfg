
from loguru import logger
from PIL import Image
from sklearn.manifold import TSNE
import pandas as pd
import csv
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import tqdm
import torchvision.transforms as transforms

LOGGER = logging.getLogger(__name__)

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import math

import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
import numpy as np
import torch
# import pydensecrf.densecrf as dcrf
# from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral

def get_masked_area(mask_tensor):
    """
    mask_tensor 에서 값이 1인 영역을 포함하는
    """

def cosine_similarity_matrix(A, B):
    """
    두 텐서 A와 B 간의 코사인 유사도 매트릭스를 계산합니다.
    
    Args:
        A (torch.Tensor): 크기 (N, D)의 텐서
        B (torch.Tensor): 크기 (M, D)의 텐서
    
    Returns:
        torch.Tensor: 크기 (N, M)의 코사인 유사도 매트릭스
    """
    # 벡터 정규화 (L2 노름)
    A_norm = A / A.norm(dim=1, keepdim=True)
    B_norm = B / B.norm(dim=1, keepdim=True)
    
    # 코사인 유사도 매트릭스 계산 (행렬 곱셈)
    cosine_sim = torch.mm(A_norm, B_norm.t())
    
    return cosine_sim

# def apply_crf(features: torch.Tensor, prob_maps: torch.Tensor,
#               sdims=(80, 80), schan=10, compat=10, iterations=5, img_ch=3) -> torch.Tensor:
#     """
#     CRF 후처리를 적용하는 함수

#     Parameters:
#     - features: 입력 특성 맵들 (B x C x H x W), torch Tensor
#     - prob_maps: 모델의 예측 확률 지도들 (B x 1 x H x W), torch Tensor
#     - sdims: 공간 차원의 표준편차 (tuple), 디폴트 (80, 80)
#     - schan: 채널 차원의 표준편차 (int 또는 list), 디폴트 10
#     - compat: 쌍별 포텐셜의 호환성 파라미터, 디폴트 10
#     - iterations: CRF 최적화 반복 횟수, 디폴트 5
#     - img_ch: 변경할 이미지 채널 수, 디폴트 3

#     Returns:
#     - CRF 후처리된 결과들 (B x 1 x H x W), torch Tensor
#     """
#     import pydensecrf.densecrf as dcrf
#     from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
#     import numpy as np
#     import torch

#     if features.shape[1] != img_ch:
#         features = torch.nn.functional.adaptive_avg_pool1d(features.permute(0, 2, 3, 1).reshape(-1, features.shape[1]), img_ch).reshape(features.shape[0], features.shape[2], features.shape[3], img_ch).permute(0, 3, 1, 2)

#     features = torch.nn.functional.interpolate(features, size=(prob_maps.shape[2], prob_maps.shape[3]), mode='bilinear', align_corners=False)

#     B, C, H, W = features.shape
#     results = []

#     for i in range(B):
#         feature = features[i]  # C x H x W
#         prob_map = prob_maps[i]  # 1 x H x W

#         # 이미지를 H x W x C 형태로 변환
#         feature_np = feature.detach().cpu().permute(1, 2, 0).numpy()  # H x W x C

#         # 클래스 수 설정 (정상=0, 비정상=1 두 클래스)
#         num_classes = 2

#         # CRF 설정
#         d = dcrf.DenseCRF2D(W, H, num_classes)

#         # 확률 지도 준비 (2 x H x W)
#         prob_map_np = prob_map.detach().cpu().numpy()[0]  # H x W
#         probs = np.stack([1 - prob_map_np, prob_map_np], axis=0)  # 2 x H x W

#         # Unary potential 설정
#         unary = unary_from_softmax(probs)
#         d.setUnaryEnergy(unary)

#         # schan이 int인 경우 채널 수에 맞게 리스트로 변환
#         if isinstance(schan, int):
#             schan_list = [schan] * C
#         else:
#             schan_list = schan

#         # Pairwise potential 설정
#         pairwise_bilateral = create_pairwise_bilateral(
#             sdims=sdims,
#             schan=schan_list,
#             img=feature_np,
#             chdim=2
#         )
#         d.addPairwiseEnergy(pairwise_bilateral, compat=compat)

#         # CRF 최적화 수행
#         Q = d.inference(iterations)
#         result = np.array(Q)[1, :].reshape(H, W) # [2, H*W] -> [H, W], argmax 했을 때, 1인 값에 가중치를 줬었으므로, index 1만 사용
#         #result = np.argmax(Q, axis=0).reshape(H, W)

#         # 결과를 Tensor로 변환하고 배치 차원 추가
#         result_tensor = torch.tensor(result).to(float).unsqueeze(0).unsqueeze(0)  # 1 x 1 x H x W
#         results.append(result_tensor)

#     # 배치 차원을 따라 결과들을 결합
#     results = torch.cat(results, dim=0)  # B x 1 x H x W

#     return results


class CustomDatasetProxy(Dataset):
    def __init__(self, *data):
        self.data1 = data[0]

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, idx):
        return self.data1[idx]


#class CustomDataset(Dataset):
#    def __init__(self, data1, data2, data3, data4, data5):
#        self.data1 = data1
#        self.data2 = data2
#        self.data3 = data3
#        self.data4 = data4
#        self.data5 = data5
#
#    def __len__(self):
#        return len(self.data1)
#
#    def __getitem__(self, idx):
#        return self.data1[idx], self.data2[idx], self.data3[idx], self.data4[idx], self.data5[idx]

class CustomDataset(Dataset):
    def __init__(self, *args):
        self.data = args

    def __len__(self):
        return len(self.data[0]) if self.data else 0

    def __getitem__(self, idx):
        return tuple(data[idx] for data in self.data)



def get_optimal_threshold(prediction, ground_truth):
    from sklearn.metrics import f1_score, precision_recall_curve
    # Flatten the tensors
    pred_flat = prediction.flatten()
    gt_flat = ground_truth.flatten()

    # Convert to numpy arrays for sklearn
    pred_flat_np = pred_flat.detach().cpu().numpy()
    gt_flat_np = gt_flat.detach().cpu().numpy()


    precision, recall, thresholds = precision_recall_curve(
       gt_flat_np, pred_flat_np
    )
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    return thresholds[np.argmax(F1_scores)]

def generate_confusion_matrix_image(prediction, ground_truth, save_path):
    """
    pixel-level 에서 best AUROC 에 대한 threshold 를 계산한 후, 이를 기반으로 confusion matrix 이미지를 생성합니다.
    색상의 의미는 다음과 같습니다:
        - TP (True Positive): 파랑
        - TN (True Negative): 녹색
        - FP (False Positive): 빨강
        - FN (False Negative): 노랑
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import f1_score, precision_recall_curve

    # Ensure prediction and ground_truth are torch tensors
    if not isinstance(prediction, torch.Tensor):
        prediction = torch.tensor(prediction)
    if not isinstance(ground_truth, torch.Tensor):
        ground_truth = torch.tensor(ground_truth)

    # Flatten the tensors
    pred_flat = prediction.flatten()
    gt_flat = ground_truth.flatten()

    # Convert to numpy arrays for sklearn
    pred_flat_np = pred_flat.detach().cpu().numpy()
    gt_flat_np = gt_flat.detach().cpu().numpy()


    precision, recall, thresholds = precision_recall_curve(
       gt_flat_np, pred_flat_np
    )
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    best_threshold = thresholds[np.argmax(F1_scores)]
    best_f1 = np.max(F1_scores)

    print(f"Best threshold: {best_threshold}, Best F1 score: {best_f1}")

    # Binarize the prediction using the best threshold
    pred_binary = (prediction >= best_threshold).int()

    # Compute confusion matrix per pixel
    TP = ((pred_binary == 1) & (ground_truth == 1))
    TN = ((pred_binary == 0) & (ground_truth == 0))
    FP = ((pred_binary == 1) & (ground_truth == 0))
    FN = ((pred_binary == 0) & (ground_truth == 1))

    # Convert to numpy arrays
    TP = TP.detach().cpu().numpy()
    TN = TN.detach().cpu().numpy()
    FP = FP.detach().cpu().numpy()
    FN = FN.detach().cpu().numpy()

    # Create confusion map
    confusion_map = np.zeros((prediction.shape[0], prediction.shape[1]), dtype=np.uint8)
    confusion_map[TP == 1] = 1
    confusion_map[TN == 1] = 2
    confusion_map[FP == 1] = 3
    confusion_map[FN == 1] = 4

    # Define colors for each class
    colors = {
        1: [0, 0, 255],    # TP: blue
        2: [0, 255, 0],    # TN: green
        3: [255, 0, 0],    # FP: red
        4: [255, 255, 0],  # FN: yellow
    }

    # Create the confusion matrix image
    confusion_image = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    for label, color in colors.items():
        confusion_image[confusion_map == label] = color

    # Plot the images
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(prediction.detach().cpu().numpy(), cmap='gray')
    axs[0].set_title('Mask Prediction')
    axs[0].axis('off')

    axs[1].imshow(ground_truth.detach().cpu().numpy(), cmap='gray')
    axs[1].set_title('Ground Truth Label')
    axs[1].axis('off')

    axs[2].imshow(confusion_image)
    axs[2].set_title('Confusion Matrix Image')
    axs[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def cut_horizontal_image_into_squares(image_path):
    """
    좌우로 긴 이미지를 높이와 동일한 크기의 정사각형으로 잘라서 저장하는 함수.
    잘린 이미지는 원본 이미지의 표현을 변경하지 않고 그대로 저장되며,
    파일 이름 뒤에 '_cut{i}'를 붙여서 저장합니다.

    Args:
        image_path (str): 원본 이미지의 경로.

    저장된 이미지들은 원본 이미지와 동일한 형식으로 저장됩니다.
    """
    import os
    from PIL import Image

    # 이미지 열기
    image = Image.open(image_path)
    width, height = image.size

    # 정사각형의 크기 (높이와 동일)
    square_size = height

    # 자를 수 있는 정사각형의 개수 계산
    num_squares = width // square_size

    # 이미지의 파일 이름과 확장자 분리
    base_name, ext = os.path.splitext(image_path)

    # 각 정사각형 이미지 자르고 저장
    for i in range(num_squares):
        left = i * square_size
        upper = 0
        right = left + square_size
        lower = height

        # 이미지 크롭
        cropped_image = image.crop((left, upper, right, lower))

        # 새로운 파일 이름 생성
        new_image_path = f"{base_name}_cut{i}{ext}"

        # 이미지 저장
        cropped_image.save(new_image_path)

    # 남은 부분이 있으면 마지막으로 저장
    if width % square_size != 0:
        left = num_squares * square_size
        upper = 0
        right = width
        lower = height

        cropped_image = image.crop((left, upper, right, lower))
        new_image_path = f"{base_name}_cut{num_squares}{ext}"
        cropped_image.save(new_image_path)


def save_image_with_edges(concatenated_image, file_path, value_range):
    """
    Given a concatenated image, detect edges where pixel values fall outside value_range,
    overlay edges on the image, and save the image to file_path.

    Args:
        concatenated_image (numpy.ndarray): The image to process.
        file_path (str): Path to save the resulting image.
        value_range (tuple of float): (min_value, max_value). Pixels outside this range are considered for edge detection.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage import measure

    min_value, max_value = value_range

    # If the image is color, convert it to grayscale for processing
    if concatenated_image.ndim == 3 and concatenated_image.shape[2] == 3:
        gray_image = np.mean(concatenated_image, axis=2)
    else:
        gray_image = concatenated_image

    # Create a mask of pixels outside the specified value range
    mask = (gray_image <= min_value) | (gray_image >= max_value)

    # Find contours in the mask
    contours = measure.find_contours(mask.astype(np.uint8), level=0.5)

    # Set up the figure size according to the image size
    dpi = 100  # Adjust as needed
    height, width = concatenated_image.shape[0], concatenated_image.shape[1]
    figsize = (width / dpi, height / dpi)

    # Plot and save the image
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if concatenated_image.ndim == 2:
        # Grayscale image
        ax.imshow(concatenated_image, cmap='gray', vmin=0, vmax=1)
    elif concatenated_image.ndim == 3 and concatenated_image.shape[2] == 3:
        # Color image
        ax.imshow(concatenated_image, vmin=0, vmax=1)
    else:
        raise ValueError(f"Unexpected image shape: {concatenated_image.shape}")

    # Draw contours on the image
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='red')

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(file_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()


def visualize_tensor_list_as_a_tensor(tensor_list, file_path, norm=True):
    """
    텐서 리스트를 받아 텐서를 오른쪽으로 이어 붙이고 저장하는 함수.
    각 텐서는 [B, C, H, W] 형태이며, Min-Max 정규화를 적용합니다.
    모든 이미지는 256x256으로 리사이즈됩니다.
    이미지가 좌우로 길어져도 해상도가 충분하도록 plt.savefig를 사용하여 저장합니다.

    Args:
        tensor_list (list of torch.Tensor): 시각화할 텐서의 리스트. 각 텐서는 [B, C, H, W] 형태.
        file_path (str): 결과 이미지를 저장할 경로.
        norm (bool): True일 경우 Min-Max 정규화를 적용합니다.
    """
    import torch
    import torch.nn.functional as F
    import numpy as np
    import matplotlib.pyplot as plt

    # 모든 텐서를 배치 차원을 펼쳐 하나의 리스트로 수집
    all_tensors = []
    for tensor in tensor_list:
        B, C, H, W = tensor.shape
        tensor = tensor.cpu()
        # 배치 차원을 펼침
        for b in range(B):
            all_tensors.append(tensor[b])  # [C, H, W]

    # 모든 이미지를 256x256으로 리사이즈
    resized_tensors = []
    for tensor in all_tensors:
        tensor = F.interpolate(tensor.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False)[0]  # [C, 256, 256]
        resized_tensors.append(tensor)

    # 모든 텐서를 스택하여 전역 최소/최대값 계산
    tensors_stack = torch.stack(resized_tensors)  # [N, C, H, W]

    # 텐서를 numpy 이미지로 변환
    numpy_images = []
    for tensor in tensors_stack:
        C, H, W = tensor.shape
        if C == 1:
            image = tensor.squeeze(0).numpy()  # [H, W]
        elif C == 3:
            image = tensor.permute(1, 2, 0).numpy()  # [H, W, C]
        else:
            # 다른 채널 수의 경우 채널 평균
            image = tensor.mean(0).numpy()  # [H, W]
        numpy_images.append(image)

    # 모든 이미지를 폭 방향으로 연결
    concatenated_image = np.concatenate(numpy_images, axis=1)  # 폭 방향으로 연결

    # Min-Max 정규화
    if norm:
        global_min = concatenated_image.min()
        global_max = concatenated_image.max()
        concatenated_image = (concatenated_image - global_min) / (global_max - global_min + 1e-20)
    else:
        concatenated_image = concatenated_image


    # 이미지 크기에 따른 figsize 계산
    dpi = 100  # 해상도를 조절하고 싶다면 이 값을 변경하세요
    height, width = concatenated_image.shape[0], concatenated_image.shape[1]
    figsize = (width / dpi, height / dpi)  # 인치 단위로 변환

    backgroundv = concatenated_image[0][0]
    save_image_with_edges(concatenated_image, file_path + '_edges.png', (backgroundv-1e-5, backgroundv+1e-5))
    cut_horizontal_image_into_squares(file_path + '_edges.png')





def visualize_tensor_list(names_list, tensor_list, file_path, norm=True, res_ratio=1.0, method='bilinear'):
    """
    이름 리스트와 텐서 리스트를 받아 텐서를 시각화하고 저장하는 함수.
    각 텐서는 [B, C, H, W] 형태이며, Min-Max 정규화를 적용합니다.
    가능한 한 가로와 세로의 플롯 수가 비슷하도록 그리드를 구성합니다.
    모든 이미지는 256x256으로 리사이즈됩니다.
    만약 채널 수가 3이라면 컬러 이미지로 저장합니다.
    
    Args:
        names_list (list of str): 각 텐서에 대한 이름의 리스트.
        tensor_list (list of torch.Tensor): 시각화할 텐서의 리스트. 각 텐서는 [B, C, H, W] 형태.
        file_path (str): 결과 이미지를 저장할 경로.
    """
    # 이름 리스트와 텐서 리스트의 길이가 동일한지 확인
    assert len(names_list) == len(tensor_list), "names_list와 tensor_list의 길이는 같아야 합니다."

    # Normalize 함수 정의
    def min_max_normalize(tensor):
        if not norm:
            return tensor
        tensor = tensor - tensor.min()
        tensor = tensor / (tensor.max() - tensor.min() + 1e-6)  # 0으로 나누는 것을 방지하기 위해 작은 값을 더함
        return tensor

    # 총 텐서 수
    num_tensors = len(tensor_list)

    # 그리드 크기 계산 (가로와 세로의 개수)
    cols = int(math.ceil(math.sqrt(num_tensors)))
    rows = int(math.ceil(num_tensors / cols))

    # 그림 크기 설정 (각 서브플롯을 고정된 크기로 설정)
    aa = 2.56
    bb = int(100*res_ratio)
    plt.figure(figsize=(cols * aa, rows * aa), dpi=bb)

    # 각 텐서를 시각화
    for idx, (name, tensor) in enumerate(zip(names_list, tensor_list)):
        # 텐서: [B, C, H, W]
        # 배치 크기는 1이라고 가정
        tensor = tensor[0]  # [C, H, W]

        # 텐서를 256x256으로 리사이즈
        dd = int(256*res_ratio)
        if tensor.shape[-1] > dd:
            tensor = F.interpolate(tensor.unsqueeze(0), size=(dd, dd), mode=method, align_corners=False)[0]  # [C, 256, 256]

        # Min-Max 정규화
        tensor = min_max_normalize(tensor)

        # 텐서의 채널 수 확인
        C = tensor.shape[0]

        # CPU로 이동 및 numpy 변환
        tensor = tensor.cpu().numpy()

        plt.subplot(rows, cols, idx + 1)

        if C == 1:
            # 채널이 1개인 경우 (흑백 이미지)
            tensor = tensor.squeeze(0)  # [256, 256]
            plt.imshow(tensor, cmap='gray')
        elif C == 3:
            # 채널이 3개인 경우 (컬러 이미지)
            tensor = tensor.transpose(1, 2, 0)  # [256, 256, 3]
            plt.imshow(tensor)
        else:
            # 다른 채널 수의 경우는 평균하여 흑백 이미지로 변환
            tensor = tensor.mean(axis=0)  # [256, 256]
            plt.imshow(tensor, cmap='gray')

        plt.title(name)
        plt.axis('off')

    plt.tight_layout()
    cc = int(300*res_ratio)
    plt.savefig(file_path, dpi=cc, bbox_inches='tight', pad_inches=0)
    plt.cla()
    plt.clf()
    plt.close()  # 플롯을 닫아 메모리 사용을 줄임

def find_mismatched_layers(modelA, modelB, input_data):
    """
    모델A와 모델B의 모든 레이어에서 동일한 입력에 대한 출력을 비교하여
    다른 값이 나오는 레이어 이름을 반환합니다.
    
    Args:
        modelA (torch.nn.Module): 첫 번째 모델
        modelB (torch.nn.Module): 두 번째 모델
        input_data (torch.Tensor): 입력 데이터
        
    Returns:
        List[str]: 값이 다른 레이어 이름 리스트
    """
    import torch

    # Ensure both models are in evaluation mode
    modelA.eval()
    modelB.eval()
    
    mismatched_layers = []
    
    outputsA = {}
    outputsB = {}
    
    # Define hook function
    def get_activation(name, outputs):
        def hook(model, input, output):
            outputs[name] = output
        return hook

    # Register hooks for all layers in both models
    hooksA = []
    hooksB = []

    for name, layer in modelA.named_modules():
        try:
            hooksA.append(layer.register_forward_hook(get_activation(name, outputsA)))
        except Exception as e:
            print(f"Error occurred for modelA layer: {name}, {e}")
    
    for name, layer in modelB.named_modules():
        try:
            hooksB.append(layer.register_forward_hook(get_activation(name, outputsB)))
        except Exception as e:
            print(f"Error occurred for modelB layer: {name}, {e}")

    # Forward pass
    with torch.no_grad():
        modelA(input_data)
        modelB(input_data)
    
    # Compare outputs
    for name in outputsA:
        if name in outputsB:
            outputA = outputsA[name]
            outputB = outputsB[name]
            if isinstance(outputA, torch.Tensor) and isinstance(outputB, torch.Tensor):
                if not torch.equal(outputA, outputB):
                    mismatched_layers.append(name)
                else:
                    print(f"Output for layer {name} is identical.")
            else:
                # Handle other cases if needed
                #mismatched_layers.append(name)
                print(f"Output for layer {name} is not a tensor.")
    
    # Remove hooks
    for hook in hooksA + hooksB:
        hook.remove()
    
    return mismatched_layers



def visualize_all_features(image, mask, features_list, file_path, feature_name='layer'):
    # image: [B, C, H, W]
    # mask: [B, 1, H, W] (binary mask) or None
    # features_list: 각 레이어의 피처 맵 리스트, 각 텐서는 [B, C, H, W]
    def normalize(tensor):
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())  # Normalize to [0, 1]
        tensor = tensor * 255.0  # Scale to [0, 255]
        tensor = tensor.byte()  # Convert to 8-bit integer
        return tensor

    # 필요한 변환 정의
    to_pil = transforms.ToPILImage()
    resize = transforms.Resize((256, 256))

    # 배치 크기는 1이라고 가정
    image = image[0]  # [C, H, W]

    # 원본 이미지가 RGB일 경우 (채널 수가 3)
    if image.shape[0] == 3:
        image = image.permute(1, 2, 0).cpu().numpy()  # [C, H, W] -> [H, W, C]
        image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0, 1]
    else:
        raise ValueError("The original image should have 3 channels (RGB).")

    # 서브플롯 개수 설정 (원본 이미지 + 마스크 (있을 경우) + 레이어 수)
    num_layers = len(features_list)
    total_plots = num_layers + 1 if mask is None else num_layers + 2

    # 그리드 크기 계산 (가로와 세로의 개수)
    cols = int(math.ceil(math.sqrt(total_plots)))
    rows = int(math.ceil(total_plots / cols))

    # 그림 크기 조정 (각 서브플롯을 256x256 픽셀로 고정)
    plt.figure(figsize=(cols * 2.56, rows * 2.56), dpi=300)  # 각 서브플롯당 256x256 해상도 (2.56 인치 * 100 DPI)

    # 첫 번째 서브플롯에 원본 이미지 표시
    plt.subplot(rows, cols, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    plot_idx = 2  # 두 번째 서브플롯부터 시작

    # 마스크가 있을 경우 마스크 출력
    if mask is not None:
        mask = mask[0].squeeze().cpu().numpy()  # [H, W]
        mask = normalize(torch.from_numpy(mask)).cpu().numpy()
        plt.subplot(rows, cols, plot_idx)
        plt.imshow(mask, cmap='gray')
        plt.title('Mask')
        plt.axis('off')
        plot_idx += 1

    for idx, feature in enumerate(features_list):
        # feature: [B, C, H, W]
        feature = feature[0]  # [C, H, W]
        # 채널 축소 (예: 평균)
        feature_map = torch.mean(feature, dim=0)  # [H, W]
        # Normalize and resize
        feature_map = normalize(feature_map).cpu().numpy()  # [H, W]
        feature_map = resize(to_pil(torch.from_numpy(feature_map).unsqueeze(0)))

        plt.subplot(rows, cols, plot_idx)
        plt.imshow(feature_map, cmap='gray')
        plt.title(f'{feature_name}_{idx+1}')
        plt.axis('off')
        plot_idx += 1

    plt.tight_layout()
    plt.savefig(file_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.cla()
    plt.clf()
    plt.close()  # 플롯을 닫아 메모리 사용을 줄임


def save_tensor_as_image(tensor, filename):
    # 텐서가 3채널인지 1채널인지 확인
    if tensor.ndimension() == 3 and tensor.size(0) == 3:  # 3채널 (RGB)
        # Normalize 후 0~255로 스케일링
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())  # Normalize to [0,1]
        tensor = tensor * 255.0  # Scale to [0, 255]
        tensor = tensor.byte()  # Convert to 8-bit integer

        # 텐서를 PIL 이미지로 변환 (CHW -> HWC)
        transform = transforms.ToPILImage(mode='RGB')
        img = transform(tensor)

    elif tensor.ndimension() == 3 and tensor.size(0) == 1:  # 1채널 (그레이스케일)
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())  # Normalize to [0,1]
        tensor = tensor * 255.0  # Scale to [0, 255]
        tensor = tensor.byte()  # Convert to 8-bit integer

        # 텐서를 PIL 이미지로 변환 (1채널은 GRAY로)
        transform = transforms.ToPILImage(mode='L')  # 'L' for grayscale
        img = transform(tensor.squeeze(0))  # Remove the first dimension (channel)

    else:
        raise ValueError("Input tensor must have 1 or 3 channels")

    # 파일로 저장
    img.save(filename)


#def plot_segmentation_images(
#    savefolder,
#    image_paths,
#    segmentations,
#    anomaly_scores=None,
#    mask_paths=None,
#    image_labels=None,
#    image_transform=lambda x: x,
#    mask_transform=lambda x: x,
#    save_depth=4,
#):
#    """Generate anomaly segmentation images.
#
#    Args:
#        image_paths: List[str] List of paths to images.
#        segmentations: [List[np.ndarray]] Generated anomaly segmentations.
#        anomaly_scores: [List[float]] Anomaly scores for each image.
#        mask_paths: [List[str]] List of paths to ground truth masks.
#        image_transform: [function or lambda] Optional transformation of images.
#        mask_transform: [function or lambda] Optional transformation of masks.
#        save_depth: [int] Number of path-strings to use for image savenames.
#    """
#    if mask_paths is None:
#        mask_paths = ["-1" for _ in range(len(image_paths))]
#    masks_provided = mask_paths[0] != "-1"
#    if anomaly_scores is None:
#        anomaly_scores = ["-1" for _ in range(len(image_paths))]
#
#    os.makedirs(savefolder, exist_ok=True)
#
#    mapped_image_score_df = pd.DataFrame(
#        columns=["Image_path", "Anomaly_score", "Label"])
#    for image_path, mask_path, anomaly_score, segmentation, image_label in tqdm.tqdm(
#        zip(image_paths, mask_paths, anomaly_scores, segmentations, image_labels),
#        total=len(image_paths),
#        desc="Generating Segmentation Images...",
#        leave=False,
#    ):
#        image = PIL.Image.open(image_path).convert("RGB")
#        image = image_transform(image)
#        if segmentation is None:
#            segmentation = np.zeros_like(mask)
#        if not isinstance(image, np.ndarray):
#            image = image.numpy()
#
#        if masks_provided:
#            if mask_path is not None:
#                mask = PIL.Image.open(mask_path).convert("RGB")
#                mask = mask_transform(mask)
#                if not isinstance(mask, np.ndarray):
#                    mask = mask.numpy()
#            else:
#                mask = np.zeros_like(image)
#
#        savename = image_path.split("/")
#        savename = "_".join(savename[-save_depth:])
#        savename = os.path.join(savefolder, savename)
#
#        # add row to dataframe
#        mapped_image_score_df.loc[len(mapped_image_score_df)] = [
#            savename.split("/")[-1], anomaly_score, image_label]
#
#        f, axes = plt.subplots(1, 2 + int(masks_provided))
#        axes[0].imshow(image.transpose(1, 2, 0))
#        axes[1].imshow(mask.transpose(1, 2, 0))
#        axes[2].imshow(segmentation)
#
#        # add title to axes
#        axes[0].set_title(
#            f"Label {image_label} image score {anomaly_score:.3}")
#        axes[1].set_title("Mask label")
#        axes[2].set_title("Anomaly map")
#
#        f.set_size_inches(3 * (2 + int(masks_provided)), 3)
#        f.tight_layout()
#        f.savefig(savename)
#        plt.close()
#        mask = mask.mean(0).astype(bool)
#        if mask.sum() > 0: # good 이면 아래 함수 불가능
#            try:
#                generate_confusion_matrix_image(segmentation, mask, savename + "_confusion.png")
#            except:
#                pass
#
#
#    # save dict to json file
#    mapped_image_score_df.to_csv(
#        os.path.join(savefolder, "image_score.csv"), index=False)

import os
import numpy as np
import pandas as pd
import PIL.Image
import matplotlib.pyplot as plt
import tqdm
from sklearn import metrics

def plot_segmentation_images(
    savefolder,
    image_paths,
    segmentations,
    anomaly_scores=None,
    mask_paths=None,
    image_labels=None,
    image_transform=lambda x: x,
    mask_transform=lambda x: x,
    save_depth=4,
):
    """Generate anomaly segmentation images and save them into TP, FP, TN, FN folders with CAM-like visualization.

    Args:
        savefolder: Path to save the images and results.
        image_paths: List[str] List of paths to images.
        segmentations: List[np.ndarray] Generated anomaly segmentations.
        anomaly_scores: List[float] Anomaly scores for each image.
        mask_paths: List[str] List of paths to ground truth masks.
        image_labels: List[int] Ground truth labels for each image (0 or 1).
        image_transform: function or lambda Optional transformation of images.
        mask_transform: function or lambda Optional transformation of masks.
        save_depth: int Number of path-strings to use for image savenames.
    """
    if mask_paths is None:
        mask_paths = [None for _ in range(len(image_paths))]
    masks_provided = any(mask_path is not None for mask_path in mask_paths)
    if anomaly_scores is None:
        anomaly_scores = [0.0 for _ in range(len(image_paths))]

    os.makedirs(savefolder, exist_ok=True)

    # Calculate optimal threshold
    if anomaly_scores and image_labels:
        fpr, tpr, thresholds = metrics.roc_curve(image_labels, anomaly_scores)
        j_scores = tpr - fpr
        j_ordered = np.argmax(j_scores)
        optimal_threshold = thresholds[j_ordered]
    else:
        optimal_threshold = 0.5

    # Create TP, FP, TN, FN folders
    tp_folder = os.path.join(savefolder, "TP")
    fp_folder = os.path.join(savefolder, "FP")
    tn_folder = os.path.join(savefolder, "TN")
    fn_folder = os.path.join(savefolder, "FN")

    os.makedirs(tp_folder, exist_ok=True)
    os.makedirs(fp_folder, exist_ok=True)
    os.makedirs(tn_folder, exist_ok=True)
    os.makedirs(fn_folder, exist_ok=True)

    mapped_image_score_df = pd.DataFrame(
        columns=["Image_path", "Anomaly_score", "Label", "Prediction", "Category"]
    )

    for image_path, mask_path, anomaly_score, segmentation, image_label in tqdm.tqdm(
        zip(image_paths, mask_paths, anomaly_scores, segmentations, image_labels),
        total=len(image_paths),
        desc="Generating Segmentation Images...",
        leave=False,
    ):
        image = PIL.Image.open(image_path).convert("RGB")
        image = image_transform(image)
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        if masks_provided and mask_path is not None:
            mask = PIL.Image.open(mask_path).convert("RGB")
            mask = mask_transform(mask)
            if not isinstance(mask, np.ndarray):
                mask = np.array(mask)
        else:
            mask = np.zeros_like(image)

        savename_parts = image_path.split("/")
        savename = "_".join(savename_parts[-save_depth:])

        # Determine prediction
        prediction = 1 if anomaly_score >= optimal_threshold else 0

        # Determine category
        if prediction == 1 and image_label == 1:
            category = "TP"
            category_folder = tp_folder
        elif prediction == 1 and image_label == 0:
            category = "FP"
            category_folder = fp_folder
        elif prediction == 0 and image_label == 0:
            category = "TN"
            category_folder = tn_folder
        elif prediction == 0 and image_label == 1:
            category = "FN"
            category_folder = fn_folder

        # Save path
        savename_full = os.path.join(category_folder, savename)

        # Update dataframe
        mapped_image_score_df.loc[len(mapped_image_score_df)] = [
            savename, anomaly_score, image_label, prediction, category
        ]

        # Visualization
        f, axes = plt.subplots(1, 3 + int(masks_provided))

        # Original Image
        axes[0].imshow(image.transpose(1, 2, 0))
        axes[0].set_title(f"Label {image_label} | Score {anomaly_score:.3f}")

        # CAM-like Visualization
        axes[1].imshow(image.transpose(1, 2, 0), alpha=0.6)
        axes[1].imshow(segmentation, cmap="jet", alpha=0.4)
        axes[1].set_title("Image + Segmentation")

        axes[2].imshow(segmentation)
        axes[2].set_title("Anomaly Map")

        # Mask (if available)
        if masks_provided and mask_path is not None:
            axes[3].imshow(mask.transpose(1, 2, 0))
            axes[3].set_title("Mask")

        f.set_size_inches(3 * (3 + int(masks_provided)), 3)
        f.tight_layout()
        f.savefig(savename_full)
        plt.close()

        # Optional: Save confusion matrix visualization
        if masks_provided and mask_path is not None:
            mask_binary = mask.mean(0).astype(bool)
            segmentation_binary = segmentation.mean(0).astype(bool)

            if mask_binary.sum() > 0:
                try:
                    confusion_savename = os.path.join(
                        category_folder, savename + "_confusion.png"
                    )
                    generate_confusion_matrix_image(
                        segmentation_binary, mask_binary, confusion_savename
                    )
                except Exception as e:
                    print(f"Error generating confusion matrix for {savename}: {e}")

    # Save results
    mapped_image_score_df.to_csv(
        os.path.join(savefolder, "image_score.csv"), index=False
    )




def create_storage_folder(
    main_folder_path, project_folder, group_folder, run_name, mode="iterate"
):
    os.makedirs(main_folder_path, exist_ok=True)
    project_path = os.path.join(main_folder_path, project_folder)
    os.makedirs(project_path, exist_ok=True)
    save_path = os.path.join(project_path, group_folder, run_name)
    if mode == "iterate":
        counter = 0
        while os.path.exists(save_path):
            save_path = os.path.join(
                project_path, group_folder + "_" + str(counter))
            counter += 1
        os.makedirs(save_path)
    elif mode == "overwrite":
        os.makedirs(save_path, exist_ok=True)

    return save_path


def set_torch_device(gpu_ids):
    """Returns correct torch.device.

    Args:
        gpu_ids: [list] list of gpu ids. If empty, cpu is used.
    """
    if len(gpu_ids):
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        return torch.device("cuda:{}".format(gpu_ids[0]))
    return torch.device("cpu")


def fix_seeds(seed, with_torch=True, with_cuda=True):
    """Fixed available seeds for reproducibility.

    Args:
        seed: [int] Seed value.
        with_torch: Flag. If true, torch-related seeds are fixed.
        with_cuda: Flag. If true, torch+cuda-related seeds are fixed
    """
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)
    if with_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_and_store_final_results(
    results_path,
    results,
    row_names=None,
    column_names=[
        "Instance AUROC",
        "Full Pixel AUROC",
        "Full PRO",
        "Anomaly Pixel AUROC",
        "Anomaly PRO",
    ],
    onnx="no",
):
    """Store computed results as CSV file.

    Args:
        results_path: [str] Where to store result csv.
        results: [List[List]] List of lists containing results per dataset,
                 with results[i][0] == 'dataset_name' and results[i][1:6] =
                 [instance_auroc, full_pixelwisew_auroc, full_pro,
                 anomaly-only_pw_auroc, anomaly-only_pro]
    """
    if row_names is not None:
        assert len(row_names) == len(results), "#Rownames != #Result-rows."

    mean_metrics = {}
    for i, result_key in enumerate(column_names):
        mean_metrics[result_key] = np.mean([x[i] for x in results])
        LOGGER.info("{0}: {1:3.3f}".format(
            result_key, mean_metrics[result_key]))

    savename = os.path.join(results_path, f"results_ONNX{onnx}.csv")
    with open(savename, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        header = column_names
        if row_names is not None:
            header = ["Row Names"] + header

        csv_writer.writerow(header)
        for i, result_list in enumerate(results):
            csv_row = result_list
            if row_names is not None:
                csv_row = [row_names[i]] + result_list
            csv_writer.writerow(csv_row)
        mean_scores = list(mean_metrics.values())
        if row_names is not None:
            mean_scores = ["Mean"] + mean_scores
        csv_writer.writerow(mean_scores)

    mean_metrics = {"mean_{0}".format(
        key): item for key, item in mean_metrics.items()}
    return mean_metrics

def hdbscan_clustering(features, min_cluster_size=5, min_samples=None, cluster_selection_epsilon=0.0, metric='euclidean', cluster_selection_method='eom'):
    """
    Parameters
    ----------
    min_cluster_size: 클러스터의 최소 크기를 지정합니다. 값을 작게 설정하면 더 많은 클러스터가 형성될 수 있습니다.
    min_samples: 한 포인트가 코어 포인트로 간주되기 위한 최소 샘플 수입니다. 기본값은 min_cluster_size와 동일합니다.
    cluster_selection_epsilon: 클러스터 선택 시 사용되는 거리 임계값입니다. 이 값을 증가시키면 더 많은 클러스터가 형성될 수 있습니다.
    """
    import hdbscan
    # HDBSCAN 클러스터링 수행
    if metric == "euclidean":
        mat = torch.cdist(features, features)
    elif metric == "cosine":
        mat = cosine_similarity_matrix(features, features)
    else:
        raise NotImplementedError(f"Metric {metric} is not implemented.")

    clusterer = hdbscan.HDBSCAN( min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_epsilon=cluster_selection_epsilon, cluster_selection_method=cluster_selection_method, metric="precomputed",)
    clusterer.fit(mat.cpu().to(float).numpy())
    # 클러스터 레이블 가져오기
    targets = clusterer.labels_
    # 이상치 점수 가져오기 (가능한 경우)
    dists = clusterer.outlier_scores_
    return dists, targets


def kmeans_clustering(features, n_clusters):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++')
    kmeans.fit(features)
    targets = kmeans.labels_
    # 각 샘플에서 가장 가까운 센트로이드까지의 거리 계산
    dists = kmeans.transform(features).min(axis=1)
    #return torch.from_numpy(dists), torch.from_numpy(targets)
    return dists, targets

def optics_clustering(features, min_samples=5, max_eps=np.inf, cluster_method='xi'):
    from sklearn.cluster import OPTICS

    # OPTICS 클러스터링 수행
    clusterer = OPTICS(min_samples=min_samples, max_eps=max_eps, cluster_method=cluster_method)
    clusterer.fit(features)

    # 클러스터 레이블 및 노이즈 처리
    targets = clusterer.labels_
    
    # Reachability distances: 각 포인트의 밀도 기반 거리
    dists = clusterer.reachability_

    return dists, targets

def mask_to_patch(mask, patch_shape):
    """
    예., 224x224 크기의 mask 를 28x28 의 patch 로 나누고, patch 마다 mask 의 max 값을 patch 값으로 사용. 즉, 28x28 크기의 patch mask 를 반환.

    Parameters
    ----------
    mask: np.ndarray
        mask.shape == (224, 224)
    patch_shape: tuple
        patch_shape == (28, 28)
    """
    # mask.shape == (batch, 1(ch), height, width)
    kernel_size = (mask.shape[2] // patch_shape[0], mask.shape[3] // patch_shape[1])
    
    # stride는 kernel_size와 동일하게 설정
    stride = kernel_size
    
    # max_pooling2d을 사용하여 패치 크기만큼 최대값 추출 (stride 적용)
    patch = torch.nn.functional.max_pool2d(mask, kernel_size=kernel_size, stride=stride)
    #patch[patch < 1.0] = 0.0
    
    return patch

def save_mask_as_png(mask, filename):
    """
    Example
    -------
    mask = np.random.rand(224, 224)  # (224, 224) 크기의 랜덤 마스크
    save_mask_as_png(mask, 'mask_output.png')
    """
    # 마스크의 최소, 최대값 계산
    mask_min = mask.min()
    mask_max = mask.max()
    
    # 0~1 사이로 정규화
    mask = (mask - mask_min) / (mask_max - mask_min)# 0~1 값을 0~255로 변환 (그레이스케일)

    mask_255 = (mask * 255).astype(np.uint8)
    
    # 마스크를 PIL 이미지로 변환
    img = Image.fromarray(mask_255)
    
    # PNG 파일로 저장
    img.save(filename)

def save_image_and_mask_as_png(image_path, mask, filename):
    image = Image.open(image_path)
    # 마스크의 최소, 최대값 계산
    mask_min = mask.min()
    mask_max = mask.max()

    # 0~1 사이로 정규화
    mask = (mask - mask_min) / (mask_max - mask_min)

    # 0~1 값을 0~255로 변환 (그레이스케일)
    mask_255 = (mask * 255).astype(np.uint8)

    # 마스크를 PIL 이미지로 변환
    mask_img = Image.fromarray(mask_255)

    # 이미지와 마스크의 크기를 동일하게 맞춰줌
    if image.size != mask_img.size:
        mask_img = mask_img.resize(image.size)

    # 이미지를 나란히 이어붙임 (왼쪽에 이미지, 오른쪽에 마스크)
    combined_width = image.width + mask_img.width
    combined_image = Image.new('RGB', (combined_width, image.height))

    # 이미지를 왼쪽에, 마스크를 오른쪽에 배치
    combined_image.paste(image, (0, 0))
    combined_image.paste(mask_img.convert('RGB'), (image.width, 0))

    # PNG 파일로 저장
    combined_image.save(filename)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D

def plot_tsne(vector, path, color_indices=[], shape_indices=[], sizes_indices={}, resolution=300, font_size=12, legend_labels={}, method='euclidean'):
    """
    Parameters
    ----------
    vector : numpy.ndarray
        시각화할 데이터 벡터, 형태는 (n_samples, n_features).
    path : str
        결과 이미지를 저장할 경로.
    color_indices : list of lists
        각 서브리스트는 동일한 색상 인덱스를 할당할 데이터 포인트의 인덱스를 포함합니다.
        색상 인덱스가 -1인 경우 기본 색상 'black'이 할당됩니다.
    shape_indices : list of lists
        각 서브리스트는 동일한 모양 인덱스를 할당할 데이터 포인트의 인덱스를 포함합니다.
        모양 인덱스가 -1인 경우 기본 모양 'o'가 할당됩니다.
    sizes_indices : dict
        키는 크기, 값은 해당 크기를 할당할 데이터 포인트의 인덱스 리스트입니다.
        기본 크기는 0.1입니다.
    resolution : int, optional
        저장되는 이미지의 DPI 해상도. 기본값은 300입니다.
    font_size : int, optional
        축 레이블의 폰트 크기. 기본값은 12입니다.
    legend_labels : dict
        범례 라벨을 위한 딕셔너리입니다. 키는 다음과 같습니다:
            - 모양 인덱스(int): 모양(회색)만 사용하여 범례를 생성합니다.
            - (색상 인덱스, 모양 인덱스): 색상과 모양을 모두 사용하여 범례를 생성합니다.
    method: str, optional
        t-SNE에 사용할 거리 측정 방법. 'euclidean' 또는 'cosine' 중 하나를 선택할 수 있습니다.

    Examples
    --------
    vectors = np.random.rand(100, 50)

    color_indices = [
        [i for i in range(50)],       # 색상 인덱스 0
        [i for i in range(50, 100)],  # 색상 인덱스 1
        [i for i in range(100, 150)], # 색상 인덱스 -1 (기본 'black')
    ]

    shape_indices = [
        [i for i in range(0, 30)],    # 모양 인덱스 0
        [i for i in range(30, 60)],   # 모양 인덱스 1
        [i for i in range(60, 90)],   # 모양 인덱스 -1 (기본 'o')
    ]

    sizes_indices = {
        0.5: [i for i in range(150)]   # 모든 포인트 크기 0.5
    }

    legend_labels = {
        0: 'Group A',                 # 모양 인덱스 0 (모양만 사용하여 범례 생성)
        (1, 1): 'Group B',            # 색상 인덱스 1, 모양 인덱스 1 (색상과 모양 모두 사용)
        (-1, -1): 'Default Group',    # 기본 색상과 기본 모양 사용
    }

    plot_tsne(vectors, 'tsne.png', color_indices, shape_indices, sizes_indices, legend_labels=legend_labels)
    """
    num_points = vector.shape[0]
    default_color = 'black'
    default_shape = 'o'
    default_size = 0.1

    colors = [default_color] * num_points  # 기본 색상으로 초기화
    shapes = [default_shape] * num_points  # 기본 모양으로 초기화
    sizes = [default_size] * num_points    # 기본 크기로 초기화

    # 색상 할당
    num_color_groups = max(0, max(idx for idx, _ in enumerate(color_indices) if idx != -1) + 1)
    cmap = plt.get_cmap('viridis')
    assigned_colors = cmap(np.linspace(0, 1, num_color_groups))

    for idx, indices in enumerate(color_indices):
        # 인덱스가 -1인 경우 기본 색상 할당
        if idx == -1:
            color = default_color
        else:
            color = assigned_colors[idx]
        indices = indices.tolist() if isinstance(indices, (np.ndarray, np.generic)) else indices
        for i in indices:
            colors[i] = color

    # 모양 할당
    available_shapes = ['^', 's', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']
    assigned_shapes = available_shapes[:len(shape_indices)]

    for idx, indices in enumerate(shape_indices):
        # 인덱스가 -1인 경우 기본 모양 할당
        if idx == -1:
            shape = default_shape
        else:
            shape = assigned_shapes[idx]
        indices = indices.tolist() if isinstance(indices, (np.ndarray, np.generic)) else indices
        for i in indices:
            shapes[i] = shape

    # 크기 할당
    for size, indices in sizes_indices.items():
        indices = indices.tolist() if isinstance(indices, (np.ndarray, np.generic)) else indices
        for i in indices:
            sizes[i] = size

    # t-SNE 계산
    perplexity = max(5, min(50, len(vector) // 5))
    if method == 'cosine':
        vector = F.normalize(torch.from_numpy(vector).to('cuda'), p=2, dim=1)
        cos_sim = F.linear(vector, vector)
        cos_sim = (cos_sim + 1)/2 # 모두 양수로 0~1 범위로
        distance = torch.clamp(1 - cos_sim, min=0) # 역으로 거리로
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0, metric='precomputed', init='random') # init=random 이 필요한 이유: t-SNE에서 metric="precomputed"를 사용할 때, init="pca"는 호환되지 않습니다. 이유는 precomputed를 사용하는 경우, 거리 행렬이 이미 계산되어 있고, 이 행렬에는 PCA 초기화에 필요한 고차원 공간의 정보가 포함되어 있지 않기 때문입니다.
        vector_2d = tsne.fit_transform(distance.cpu().numpy())
    else:
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0)
        vector_2d = tsne.fit_transform(vector)

    fig, ax = plt.subplots()

    logger.info(f"unique colors: {assigned_colors}, total length {len(assigned_colors)+1} including '{default_color}' (-1)")
    logger.info(f"unique shapes: {assigned_shapes}, total length {len(assigned_shapes)+1} including '{default_shape}' (-1)")

    # 데이터 포인트 플롯
    for x, y, color, shape, size in zip(vector_2d[:, 0], vector_2d[:, 1], colors, shapes, sizes):
        ax.scatter(x, y, c=[color], marker=shape, s=size)

    # 범례 생성
    legend_elements = []
    for key, label in legend_labels.items():
        if isinstance(key, int):
            # 모양 인덱스만 사용하는 경우 (회색으로 표시)
            shape_idx = key
            shape = default_shape if shape_idx == -1 else assigned_shapes[shape_idx]
            legend_elements.append(Line2D([0], [0], marker=shape, color='w', label=label,
                                          markerfacecolor='gray', markersize=10))
        elif isinstance(key, tuple) and len(key) == 2:
            color_idx, shape_idx = key
            color = default_color if color_idx == -1 else assigned_colors[color_idx]
            shape = default_shape if shape_idx == -1 else assigned_shapes[shape_idx]
            legend_elements.append(Line2D([0], [0], marker=shape, color='w', label=label,
                                          markerfacecolor=color, markersize=10))
        else:
            raise ValueError("legend_labels 키는 int 또는 (int, int) 여야 합니다.")

    # 축 레이블 및 범례 설정
    ax.set_xlabel('X', fontsize=font_size)
    ax.set_ylabel('Y', fontsize=font_size)
    if legend_elements:
        ax.legend(handles=legend_elements)

    # 플롯 저장
    plt.savefig(path, dpi=resolution, bbox_inches='tight')
    plt.close()


def plot_sampled_proxy_tsne(feats, labels, proxies, filename, n_samples=1000, method='euclidean'):
    """
    TSNE 시각화를 위해 샘플링된 데이터의 t-SNE를 생성하고 저장합니다.
    1. labels 중에 labels.unique() 별로 n_samples 만큼 indices 를 추출. 즉 len(labels.unique()) * n_samples 만큼 indices 를 추출.
    2. feats 와 labels 에서 추출한 indices 를 이용해서 추출한 subset 들을 다시 sub_feats, sub_labels 변수에 할당

    Args:
        feats (numpy.ndarray): Feature 행렬 (shape: [N, D])
        labels (numpy.ndarray): 레이블 벡터 (shape: [N])
        proxies (torch.Tensor): Proxy 텐서 (shape: [M, D])
        filename (str): 저장할 파일 이름
        n_samples (int): 각 클래스당 샘플 수
    """
    import numpy as np
    # 고유한 레이블 추출
    unique_labels = np.unique(labels)

    # 각 클래스별로 n_samples 만큼의 인덱스 추출
    sampled_indices = []
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        sampled_indices.extend(np.random.choice(label_indices, size=min(n_samples, len(label_indices)), replace=False))

    # 서브셋 생성
    sub_feats = feats[sampled_indices]
    sub_labels = labels[sampled_indices]

    indices_0 = np.where(sub_labels == 0)[0].tolist()
    indices_1 = np.where(sub_labels == 1)[0].tolist()

    color_indices = [indices_0, indices_1]

    # add proxies
    sub_feats = np.concatenate([sub_feats, proxies.proxies.cpu().detach().numpy()], axis=0)
    sub_labels = np.concatenate([sub_labels, np.ones(len(proxies.proxies)) + 1], axis=0) # 2를 할당
    color_indices.append(list(range(len(sub_feats) - len(proxies.proxies), len(sub_feats))))

    # shape_indices
    shape_indices_01 = np.concatenate([np.where(sub_labels == 0)[0], np.where(sub_labels == 1)[0]]).tolist()
    shape_indices_2 = np.where(sub_labels == 2)[0].tolist()
    shape_indices = [shape_indices_01, shape_indices_2]

    plot_tsne(sub_feats, filename, color_indices=color_indices, shape_indices=shape_indices, sizes_indices={2.0: list(range(len(sub_feats)))}, legend_labels={(0, 0): 'Normal', (1, 0): 'Anomaly', (2,1):'Proxy'}, method=method)

from loguru import logger
import torch
import numpy as np

# Helper function to extract shape
def get_shape(obj):
    if isinstance(obj, torch.Tensor):
        return tuple(obj.shape)
    elif isinstance(obj, np.ndarray):
        return obj.shape
    elif isinstance(obj, (list, tuple)):
        return [get_shape(o) for o in obj]
    else:
        return "N/A"

# Enhanced decorator with loguru
def log_shapes(func):
    def wrapper(*args, **kwargs):
        logger.debug(f"\n[{func.__name__}] Input Details:")

        # Execute the function
        result = func(*args, **kwargs)

        # Log inputs
        for i, arg in enumerate(args):
            try:
                shape = get_shape(arg)
                logger.debug(f"[{func.__name__}] Input Arg {i}: Name=arg{i}, Type={type(arg).__name__}, Shape={shape}")
            except Exception as e:
                logger.debug(f"[{func.__name__}] Input Arg {i}: Name=arg{i}, Type={type(arg).__name__}, Shape=Error({e})")

        for key, value in kwargs.items():
            try:
                shape = get_shape(value)
                logger.debug(f"[{func.__name__}] Input Kwarg {key}: Type={type(value).__name__}, Shape={shape}")
            except Exception as e:
                logger.debug(f"[{func.__name__}] Input Kwarg {key}: Type={type(value).__name__}, Shape=Error({e})")

        # Log outputs
        logger.debug(f"[{func.__name__}] Output Details:")
        if isinstance(result, (tuple, list)):
            for i, res in enumerate(result):
                try:
                    shape = get_shape(res)
                    logger.debug(f"[{func.__name__}] Output Arg {i}: Type={type(res).__name__}, Shape={shape}")
                except Exception as e:
                    logger.debug(f"[{func.__name__}] Output Arg {i}: Type={type(res).__name__}, Shape=Error({e})")
        else:
            try:
                shape = get_shape(result)
                logger.debug(f"[{func.__name__}] Output Arg: Type={type(result).__name__}, Shape={shape}")
            except Exception as e:
                logger.debug(f"[{func.__name__}] Output Arg: Type={type(result).__name__}, Shape=Error({e})")

        return result

    return wrapper

# Example function to test
@log_shapes
def example_function(tensor, array, nested_list, scalar=42):
    output_tensor = tensor.mean(dim=1)
    output_array = np.sum(array, axis=0)
    return output_tensor, output_array, nested_list, scalar

# Test the decorator
if __name__ == "__main__":
    # Configure logger level
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="DEBUG")  # Set level to DEBUG for full logs

    tensor = torch.randn(16, 3, 256, 256)
    array = np.random.rand(16, 256, 256)
    nested_list = [[torch.randn(1, 3), np.random.rand(2, 2)], [torch.randn(2, 2), [3, 4]]]
    
    # Run the example function with logging
    result = example_function(tensor, array, nested_list, scalar="test")

    # Change log level to INFO (suppress DEBUG logs)
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="INFO")

    # Run the example function again to demonstrate log level control
    result = example_function(tensor, array, nested_list, scalar="test")


    """
    # Example function to test
    @log_shapes
    def example_function(tensor, array, nested_list, scalar=42):
        output_tensor = tensor.mean(dim=1)
        output_array = np.sum(array, axis=0)
        return output_tensor, output_array, nested_list, scalar

    # Test the decorator
    if __name__ == "__main__":
        # Configure logger level
        logger.remove()
        logger.add(lambda msg: print(msg, end=""), level="DEBUG")  # Set level to DEBUG for full logs

        tensor = torch.randn(16, 3, 256, 256)
        array = np.random.rand(16, 256, 256)
        nested_list = [[torch.randn(1, 3), np.random.rand(2, 2)], [torch.randn(2, 2), [3, 4]]]
        
        # Run the example function with logging
        result = example_function(tensor, array, nested_list, scalar="test")

        # Change log level to INFO (suppress DEBUG logs)
        logger.remove()
        logger.add(lambda msg: print(msg, end=""), level="INFO")

        # Run the example function again to demonstrate log level control
        result = example_function(tensor, array, nested_list, scalar="test")
    """