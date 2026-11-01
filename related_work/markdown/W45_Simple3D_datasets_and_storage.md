# Simple3D 평가 데이터셋과 저장용량

## Paper Metadata

| Item | Content |
|---|---|
| Title | Towards High-Resolution 3D Anomaly Detection: A Scalable Dataset and Real-Time Framework for Subtle Industrial Defects |
| Authors | Yuqi Cheng, Yihan Sun, Hui Zhang, Weiming Shen, Yunkang Cao |
| Conference / Journal | AAAI, 40(5), 3327–3334 |
| Year | 2026 |
| Paper link | [출판 논문](https://ojs.aaai.org/index.php/AAAI/article/view/37328), [부록 포함 arXiv](https://arxiv.org/html/2507.07435v1) |
| GitHub / Official code | [MiniShift-Simple3D](https://github.com/hustCYQ/MiniShift-Simple3D) |
| Reason for investigation | 논문에서 평가한 데이터셋의 구성과 공식 배포 파일·카테고리별 용량 확인 |

Simple3D의 평가 데이터셋은 **MiniShift, Real3D-AD, Anomaly-ShapeNet, MulSen-AD** 총 4개다.

## 1. 데이터셋 정보

아래 샘플 수는 **학습과 테스트를 합친 전체 규모**다. Point Cloud 1개는 물체 형상을 나타내는 샘플 1개이며, Point 수는 그 샘플 안의 좌표 개수다.

| 데이터셋 | 평가 범주 수 | 전체 샘플 수: 논문 표기 | 샘플당 Point 수: Simple3D 부록 Table 6 | 원본 데이터 형태 | 평가 초점 |
|---|---:|---:|---:|---|---|
| MiniShift | 12 | 2,577 | 500,000 | MulSen-AD 기반 형상에 결함을 합성한 Point Cloud | 미세결함, Easy / Medium / Hard |
| Real3D-AD | 12 | 1,254 | 평균 169,022 | 실제 스캔 Point Cloud | 실제 물체의 형상 결함 |
| Anomaly-ShapeNet | 40 | 1,600 | 평균 58,604 | 합성 Point Cloud와 Mesh | 다양한 물체 형상의 합성 결함 |
| MulSen-AD | 15 | 2,035 | 평균 29,818 | RGB + 열화상 + 3D 형상 | 외관·형상·내부 결함의 센서별 관측 |

범주·샘플·해상도 수치의 기준은 [Simple3D 부록 Table 6](https://arxiv.org/html/2507.07435v1#A1)이다. 평균 Point 수는 모든 샘플의 점 개수가 같다는 뜻이 아니다. MiniShift는 샘플마다 500,000개로 고정되어 있다. 원본 형태는 각 [Real3D-AD](https://github.com/M-3LAB/Real3D-AD), [Anomaly-ShapeNet](https://github.com/Chopper-233/Anomaly-ShapeNet), [MulSen-AD](https://github.com/MapleBoat/MulSen-AD) 공식 설명을 함께 확인했다.

**Simple3D는 4개 모두에서 3D 형상을 입력으로 사용한다.** MulSen-AD에 이미지가 포함되어 있어도 Simple3D가 RGB·열화상 특징까지 융합하는 것은 아니다.

## 2. 공식 배포 용량

단위는 **GB = 10⁹ bytes**다. 이 문서의 용량은 로컬에 풀어 놓은 폴더를 측정한 값이 아니다.

- **다운로드 용량:** 공식 배포 서버에 기록된 ZIP·RAR 파일 크기.
- **압축 내부 원본 용량:** 공식 압축 파일의 목록에 기록된 개별 파일의 원래 크기를 합산한 값. 실제로 압축을 풀거나 다운샘플링하지 않고 확인했다.
- 모든 용량은 **학습 + 테스트 + 정답 등 해당 배포본에 포함된 파일** 기준이다. 모델·실행 환경·실험 결과는 포함하지 않는다.
- 공식 페이지가 카테고리별 용량을 별도로 발표한 것은 아니다. 아래 세부 표는 **공식 배포 파일의 내부 메타데이터에서 계산한 값**이다.


| 데이터셋 | 공식 배포 파일 | 다운로드 용량 (GB) | 압축 내부 원본 용량 (GB) | 포함 범위 |
|---|---|---|---|---|
| MiniShift | 카테고리별 ZIP 12개 | 38.878 | 99.796 | 12범주 Point Cloud·정답 |
| Real3D-AD | Real3D-AD-PCD.zip | 3.459 | 7.579 | 12범주 PCD·정답; 별도 PLY 배포본 제외 |
| Anomaly-ShapeNet | Anomaly-ShapeNet-v2.rar | 3.816 | 12.381 | PCD·Mesh·정답 및 추가 데이터 전체 |
| MulSen-AD | MulSen_AD.zip | 8.974 | 19.294 | 15범주 RGB·열화상·3D·정답 전체 |

공식 배포 링크: [MiniShift](https://huggingface.co/datasets/ChengYuQi99/MiniShift/tree/main), [Real3D-AD](https://github.com/M-3LAB/Real3D-AD#download), [Anomaly-ShapeNet](https://huggingface.co/datasets/Chopper233/Anomaly-ShapeNet/tree/main), [MulSen-AD](https://huggingface.co/datasets/orgjy314159/MulSen_AD/tree/main).

MulSen-AD는 `MulSen_AD.rar` **7.536 GB**, `MulSen_AD.zip` **8.974 GB**, `MulSen_AD_new.zip` **8.974 GB**를 제공한다. 이 문서는 내부 목록을 직접 확인한 **`MulSen_AD.zip`**을 세부 용량 기준으로 사용한다. 서로 다른 압축 파일의 용량을 합산하지 않았으며, RAR·ZIP·new ZIP의 내용이 완전히 동일하다고 가정하지 않는다.

## 3. MiniShift: 12개 카테고리

카테고리마다 별도의 ZIP을 제공하므로 필요한 카테고리만 내려받을 수 있다. 원본 용량은 Point Cloud와 GT를 포함한다. [공식 카테고리별 배포](https://huggingface.co/datasets/ChengYuQi99/MiniShift/tree/main)


| 카테고리             | 개별 ZIP 다운로드 (GB) | 압축 내부 원본 (GB) |
| ---------------- | ---------------- | ------------- |
| button_cell      | 3.120            | 8.218         |
| capsule          | 2.768            | 7.146         |
| cube             | 3.541            | 8.888         |
| flat_pad         | 3.122            | 8.126         |
| light            | 3.529            | 8.888         |
| nut              | 3.533            | 9.196         |
| piggy            | 3.488            | 8.880         |
| plastic_cylinder | 3.181            | 8.135         |
| screen           | 2.873            | 7.326         |
| screw            | 3.148            | 8.128         |
| spring_pad       | 3.092            | 7.978         |
| toothbrush       | 3.483            | 8.888         |
| **합계**           | 38.878           | 99.796        |

## 4. Real3D-AD: 12개 카테고리

공식 ZIP 하나에 카테고리들이 함께 들어 있다. 아래 값은 각 카테고리의 **PCD 학습·테스트 파일 및 GT 원본 크기**이며, 카테고리별 다운로드 ZIP 크기가 아니다. [공식 배포 안내](https://github.com/M-3LAB/Real3D-AD#download)


| 카테고리     | 학습 PCD (GB) | 테스트 PCD (GB) | GT (GB) | 합계 (GB) |
| -------- | ----------- | ------------ | ------- | ------- |
| airplane | 0.019       | 0.422        | 1.131   | 1.572   |
| candybar | 0.019       | 0.107        | 0.213   | 0.338   |
| car      | 0.053       | 0.157        | 0.318   | 0.528   |
| chicken  | 0.056       | 0.439        | 1.058   | 1.553   |
| diamond  | 0.095       | 0.091        | 0.176   | 0.362   |
| duck     | 0.084       | 0.259        | 0.481   | 0.824   |
| fish     | 0.012       | 0.133        | 0.260   | 0.404   |
| gemstone | 0.040       | 0.126        | 0.165   | 0.331   |
| seahorse | 0.009       | 0.100        | 0.195   | 0.305   |
| shell    | 0.014       | 0.151        | 0.279   | 0.443   |
| starfish | 0.010       | 0.106        | 0.214   | 0.330   |
| toffees  | 0.027       | 0.189        | 0.373   | 0.589   |
| **합계**   | 0.436       | 2.280        | 4.864   | 7.579   |

## 5. Anomaly-ShapeNet: 기존 평가 40개 카테고리

공식 RAR는 PCD뿐 아니라 Mesh(OBJ)와 추가 데이터도 포함한다. 아래는 **Simple3D가 평가한 기존 40개 카테고리**다. PCD 열에는 해당 카테고리의 학습·테스트·GT가 포함된다. Mesh를 쓰지 않는 실행이라면 PCD 열을 보면 된다. [공식 배포본](https://huggingface.co/datasets/Chopper233/Anomaly-ShapeNet/tree/main)


| 카테고리 | PCD·GT 원본 (GB) | Mesh 원본 (GB) | 합계 (GB) |
|---|---|---|---|
| ashtray0 | 0.054 | 0.112 | 0.166 |
| bag0 | 0.025 | 0.053 | 0.078 |
| bottle0 | 0.053 | 0.117 | 0.170 |
| bottle1 | 0.096 | 0.185 | 0.281 |
| bottle3 | 0.055 | 0.109 | 0.163 |
| bowl0 | 0.054 | 0.109 | 0.163 |
| bowl1 | 0.060 | 0.121 | 0.181 |
| bowl2 | 0.077 | 0.156 | 0.233 |
| bowl3 | 0.060 | 0.121 | 0.181 |
| bowl4 | 0.044 | 0.090 | 0.135 |
| bowl5 | 0.048 | 0.094 | 0.141 |
| bucket0 | 0.068 | 0.127 | 0.195 |
| bucket1 | 0.063 | 0.141 | 0.203 |
| cap0 | 0.036 | 0.080 | 0.116 |
| cap3 | 0.133 | 0.289 | 0.422 |
| cap4 | 0.166 | 0.354 | 0.520 |
| cap5 | 0.166 | 0.326 | 0.492 |
| cup0 | 0.048 | 0.106 | 0.154 |
| cup1 | 0.051 | 0.112 | 0.163 |
| eraser0 | 0.025 | 0.053 | 0.077 |
| headset0 | 0.055 | 0.119 | 0.174 |
| headset1 | 0.024 | 0.051 | 0.075 |
| helmet0 | 0.125 | 0.277 | 0.402 |
| helmet1 | 0.065 | 0.144 | 0.208 |
| helmet2 | 0.071 | 0.155 | 0.226 |
| helmet3 | 0.068 | 0.129 | 0.198 |
| jar0 | 0.034 | 0.073 | 0.107 |
| microphone0 | 0.050 | 0.109 | 0.159 |
| shelf0 | 0.051 | 0.116 | 0.168 |
| tap0 | 0.135 | 0.260 | 0.395 |
| tap1 | 0.072 | 0.144 | 0.216 |
| vase0 | 0.141 | 0.306 | 0.448 |
| vase1 | 0.050 | 0.109 | 0.159 |
| vase2 | 0.034 | 0.073 | 0.107 |
| vase3 | 0.058 | 0.132 | 0.190 |
| vase4 | 0.171 | 0.329 | 0.501 |
| vase5 | 0.054 | 0.118 | 0.172 |
| vase7 | 0.049 | 0.107 | 0.156 |
| vase8 | 0.071 | 0.160 | 0.232 |
| vase9 | 0.185 | 0.424 | 0.608 |
| **기존 40범주 합계** | 2.945 | 6.189 | 9.134 |

### 함께 배포된 추가 데이터: Simple3D의 40범주 평가 대상 밖

공식 README는 50범주 확장을 안내한다. 확인한 RAR의 `new_pcd`·`new_obj`에는 아래 **12개 폴더명**이 존재한다. 이는 압축 파일 구성에 대한 관찰이며, 논문의 평가 범주 수를 52개로 바꾸는 근거로 사용하지 않는다.


| 추가 폴더명 | PCD·GT 원본 (GB) | Mesh 원본 (GB) | 합계 (GB) |
|---|---|---|---|
| cabinet0 | 0.070 | 0.180 | 0.250 |
| cap1 | 0.092 | 0.188 | 0.280 |
| cap2 | 0.165 | 0.331 | 0.495 |
| chair0 | 0.028 | 0.078 | 0.105 |
| cup2 | 0.043 | 0.095 | 0.137 |
| desk0 | 0.094 | 0.245 | 0.339 |
| knife0 | 0.061 | 0.174 | 0.235 |
| knife1 | 0.094 | 0.254 | 0.348 |
| microphone1 | 0.040 | 0.134 | 0.175 |
| screen0 | 0.092 | 0.249 | 0.342 |
| vase10 | 0.107 | 0.214 | 0.321 |
| vase6 | 0.071 | 0.148 | 0.219 |

기존 40범주 **9.134 GB** + 추가 데이터 **3.248 GB** = 공식 RAR 내부 전체 **12.381 GB**. RAR 다운로드 용량 **3.816 GB**와는 다른 값이다.

## 6. MulSen-AD: 15개 카테고리

아래 값은 공식 `MulSen_AD.zip` 내부 원본 크기다. 각 센서 열은 해당 센서의 학습·테스트·정답 파일을 포함한다. **3D 열은 Pointcloud 폴더(STL 형상·정답 등)** 기준이다. Simple3D는 형상을 입력으로 사용하지만, 공식 로더가 물체 정답을 읽을 때 `RGB/GT/.../data.csv`도 참조한다. [공식 데이터 구조](https://github.com/MapleBoat/MulSen-AD#24-data-directory)


| 카테고리 | 3D 원본 (GB) | RGB 원본 (GB) | 열화상 원본 (GB) | 전체 (GB) |
|---|---|---|---|---|
| button_cell | 1.122 | 0.007 | 0.161 | 1.291 |
| capsule | 0.264 | 0.450 | 0.142 | 0.856 |
| cotton | 1.376 | 0.472 | 0.157 | 2.004 |
| cube | 1.884 | 0.594 | 0.198 | 2.676 |
| flat_pad | 0.333 | 0.011 | 0.027 | 0.371 |
| light | 1.331 | 0.022 | 0.192 | 1.545 |
| nut | 0.404 | 0.011 | 0.018 | 0.433 |
| piggy | 1.296 | 0.055 | 0.028 | 1.378 |
| plastic_cylinder | 1.148 | 0.021 | 0.157 | 1.326 |
| screen | 1.139 | 0.057 | 0.020 | 1.216 |
| screw | 0.983 | 0.009 | 0.020 | 1.013 |
| solar_panel | 1.125 | 0.057 | 0.171 | 1.354 |
| spring_pad | 1.036 | 0.004 | 0.015 | 1.055 |
| toothbrush | 1.262 | 0.014 | 0.178 | 1.454 |
| zipper | 1.070 | 0.098 | 0.155 | 1.322 |
| **합계** | 15.774 | 1.883 | 1.638 | 19.294 |

## 7. 해석과 근거

- 카테고리별 **압축 내부 원본 용량**은 개별 파일 크기의 합이다. 압축 형식·반올림·압축 헤더 때문에 다운로드 용량과 다르며, 실제 디스크 할당량도 아니다.
- MiniShift의 범주별 샘플 수를 합치면 정상 학습 1,017개, 정상 테스트 120개, 이상 테스트 1,440개로 총 2,577개다.
- Anomaly-ShapeNet의 샘플 수 1,600은 논문 표기다. 배포 압축 파일에는 추가 형상도 들어 있으므로 전체 압축 용량을 기존 40범주의 평가 입력 용량으로 해석하면 안 된다.

### 재확인 방법

- 공식 파일 목록·크기·해시: 로컬 `method12/source/results/official_archive_inventory_20261101/summary.json`.
- 원본 ZIP·RAR 내부 파일 목록: 위 로컬 근거 폴더의 `*_zip_entries.json`, `shapenet_rar_entries.json`.
- 조회·집계 코드: 로컬 `method12/source/inventory_official_archives.py`.
- 문서 표 생성: 로컬 `method12/source/render_official_dataset_note.py`.
