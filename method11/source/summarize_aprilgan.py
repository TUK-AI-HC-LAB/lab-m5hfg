"""Validate all APRIL-GAN categories and build local-run comparisons/report."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parents[1]
OUT = ROOT / 'results/mvtec_zero_shot_20261006'
def read(path):
    with path.open() as file:
        return list(csv.DictReader(file))
def write(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

env = json.loads((OUT / 'environment.json').read_text())
assert env['status'] == 'completed'
april = read(OUT / 'category_metrics.csv')
assert len(april) == 15 and sum(int(row['n_images']) for row in april) == 1725
musc = {row['category']: row for row in read(LAB / 'method10/source/results/mvtec_all_paper_20261002/category_metrics.csv')}
common = {row['category']: row for row in read(LAB / 'method10/source/results/w41_official_vs_common_all_20261006/category_comparison.csv')}
compared = []
table = ['| Category | Test 수 | APRIL Image AUROC (%) | MuSc 공식 실행 Image (%) | APRIL−MuSc (%p) | APRIL Pixel AUROC (%) | MuSc 공식 실행 Pixel (%) | APRIL−MuSc (%p) |',
         '|---|---:|---:|---:|---:|---:|---:|---:|']
for row in april:
    category = row['category']
    old = musc[category]
    assert int(old['n_images']) == int(row['n_images'])
    # Verify exact files/labels through official MuSc image SHA-256 evidence.
    before = {r['image_path']: (int(r['label']), r['sha256']) for r in
              read(LAB / f'method10/source/results/mvtec_all_paper_20261002/{category}/image_scores.csv')}
    now = {r['image_path']: (int(r['label']), r['sha256']) for r in read(OUT / category / 'image_scores.csv')}
    assert before == now, category
    item = dict(category=category, n_images=int(row['n_images']),
                april_image_auroc=float(row['image_auroc']), musc_official_image_auroc=float(old['image_auroc']),
                image_delta_pp=100*(float(row['image_auroc'])-float(old['image_auroc'])),
                april_pixel_auroc=float(row['pixel_auroc']), musc_official_pixel_auroc=float(old['pixel_auroc']),
                pixel_delta_pp=100*(float(row['pixel_auroc'])-float(old['pixel_auroc'])),
                musc_common_image_auroc=float(common[category]['common_image_auroc']),
                musc_common_pixel_auroc=float(common[category]['common_pixel_auroc']))
    compared.append(item)
    table.append('| {} | {} | {:.4f} | {:.4f} | {:+.4f} | {:.4f} | {:.4f} | {:+.4f} |'.format(
        category, row['n_images'], item['april_image_auroc']*100, item['musc_official_image_auroc']*100,
        item['image_delta_pp'], item['april_pixel_auroc']*100, item['musc_official_pixel_auroc']*100, item['pixel_delta_pp']))
write(OUT / 'april_vs_local_musc.csv', compared)
means = read(OUT / 'mean_metrics.csv')[0]
mf = dict(category='macro_mean', n_images=1725)
for key in compared[0]:
    if key not in ('category', 'n_images'):
        mf[key] = sum(row[key] for row in compared) / 15
write(OUT / 'april_vs_local_musc_macro.csv', [mf])
table.append('| macro_mean | 1725 | {:.4f} | {:.4f} | {:+.4f} | {:.4f} | {:.4f} | {:+.4f} |'.format(
    mf['april_image_auroc']*100, mf['musc_official_image_auroc']*100, mf['image_delta_pp'],
    mf['april_pixel_auroc']*100, mf['musc_official_pixel_auroc']*100, mf['pixel_delta_pp']))

report = '''# APRIL-GAN 공식 구현체 MVTec AD 실행 결과

작성일: 2026-10-06. 공개 checkpoint로 **0-shot·15개 category·총 1,725장**을 평가했다. target 데이터로 추가 학습하지 않았다. 공통 framework 포팅은 이번 작업의 범위가 아니며 공식 구현체를 실행했다. 모든 아래 수치는 내 PC의 측정값이다.

## 0. MuSc 논문 전체 재현에서의 위치

목표는 MuSc 논문의 비교표와 부록 실험을 자체 실행 결과로 다시 작성하는 것이다. 비교군은 해당 논문의 0-shot·4-shot·full-shot 등 정보 조건을 유지한다. 모든 방법의 resize·backbone·학습 조건을 MuSc와 동일하게 강제로 바꾸지 않는다.

MuSc 원문 7쪽 Table 1에서 APRIL-GAN은 MVTec AD 및 VisA에 각각 0-shot과 4-shot으로 등장한다. 원문은 APRIL-GAN이 별도의 labeled auxiliary dataset으로 학습한다고 명시한다. 따라서 공개 checkpoint 평가와 학습부터 재현한 실험을 구분해서 기록한다. 이번 실행은 **MVTec AD 0-shot 행의 자체 측정**이며 APRIL-GAN 관련 실험 전체 완료를 뜻하지 않는다.

- 완료: MVTec AD 0-shot, 15개 category, 7개 metric, 이미지별 score와 raw prediction 저장.
- 남은 주 비교표: MVTec AD 4-shot, VisA 0-shot·4-shot. 4-shot의 reference 추출 및 반복 조건을 원문·공식 코드에서 확인한 뒤 적용한다.
- 남은 부록: Table 11의 APRIL-GAN에 RsCIN 적용 전후, Table 13의 속도·GPU 메모리 조건, Table 14의 BTAD 0-shot·4-shot. 표별 backbone 및 평가 조건을 따로 대조한다.
- checkpoint의 보조 데이터 학습 조건이 MuSc 비교 조건을 충족하는지 확인하고, 학습부터 재현할 필요가 있으면 그 단계를 별도로 실행한다. 현재 공개 checkpoint를 사용한 결과를 자체 학습 결과로 쓰지 않는다.

자체 Image AUROC 86.1250%, Pixel AUROC 87.6288%, AUPRO 44.0371%는 MuSc Table 1의 해당 지표와 소수 첫째 자리에서 일치한다. 이 확인은 전체 metric의 수치 동등성이나 학습 과정 재현을 증명하지 않는다. 원문 값은 여기서 조건 검증에만 쓰며 결과 CSV는 자체 측정값을 유지한다.

근거: [MuSc 원문](../../method10/paper/ICLR24_MuSc_Zero_Shot_Industrial_Anomaly_Classification_and_Segmentation_with_Mutual_Scoring_of_the_Unlabeled_Images.pdf), 본 문서의 config·환경·결과 CSV.

## Paper Metadata

| Item | Content |
|---|---|
| Title | APRIL-GAN: A Zero-/Few-Shot Anomaly Classification and Segmentation Method for CVPR 2023 VAND Workshop Challenge Tracks 1&2: 1st Place on Zero-shot AD and 4th Place on Few-shot AD |
| Authors | Xuhai Chen, Yue Han, Jiangning Zhang |
| Conference / Journal | CVPR 2023 VAND Workshop Challenge의 technical report, arXiv:2305.17382 |
| Year | 2023 |
| Paper link | https://arxiv.org/abs/2305.17382 |
| Official code | https://github.com/ByChelsea/VAND-APRIL-GAN |
| Reason for investigation | MuSc 논문의 비교 기법 중 자체 측정 결과가 없던 APRIL-GAN을 실행해 category별 결과를 확보 |

## 1. 실행 조건과 scope

- 공식 commit: `{commit}`. 코드 clone은 `/home/test/VAND-APRIL-GAN`에 있고 scoring·전처리·metric source는 변경하지 않았다.
- 공식 `test_zero_shot.sh`의 MVTec 조건을 따른다. **VisA로 학습된 `visa_pretrained.pth`를 MVTec AD에 평가**한다. MVTec checkpoint를 MVTec에 평가한 결과가 아니다.
- ViT-L-14-336 / OpenAI CLIP, 입력 518×518, feature layers 6·12·18·24, batch 1, mode zero_shot, seed 42.
- RTX 5080, Python·Torch·CUDA와 전체 패키지 버전은 [environment.json](../source/results/mvtec_zero_shot_20261006/environment.json)에 기록했다. 기존 MuSc 실행과 비교를 위해 seed 42와 TF32 비활성화를 명시했다. 공식 기본 seed는 10이다.
- zero-shot은 target 정상 reference를 쓰지 않는다는 뜻이다. CLIP 사전학습 및 공개 projection checkpoint의 VisA 학습까지 없다는 뜻은 아니다.
- 현행 환경에서 실행했고 공식 requirements의 오래된 버전 조합을 그대로 설치하지 않았다. 버전 차이가 결과에 영향을 줄 가능성은 남는다.

## 2. 실행과 evidence

- script: [run_aprilgan_mvtec_zero_shot.sh](../source/run_aprilgan_mvtec_zero_shot.sh), [capture wrapper](../source/run_aprilgan_official.py).
- config / checkpoint checksum: [config.json](../source/results/mvtec_zero_shot_20261006/config.json), [environment.json](../source/results/mvtec_zero_shot_20261006/environment.json).
- category result: [category_metrics.csv](../source/results/mvtec_zero_shot_20261006/category_metrics.csv).
- macro mean: [mean_metrics.csv](../source/results/mvtec_zero_shot_20261006/mean_metrics.csv).
- dataset audit: [dataset_audit.csv](../source/results/mvtec_zero_shot_20261006/dataset_audit.csv). 元画像 파일·label의 SHA-256을 이전 MuSc 공식 실행과 대조해 1,725장 모두 일치함을 확인했다.
- raw prediction: `/home/test/aprilgan_results/mvtec_zero_shot_20261006/<category>/raw_predictions.npz`; image score 목록은 연구 원본 results의 category별 `image_scores.csv`에 있다. 대용량 raw array와 가중치는 연구 repo 및 Obsidian 색인에 복사하지 않았다.
- visualization: `/home/test/aprilgan_results/mvtec_zero_shot_20261006/official/imgs/`.
- original log: `/home/test/aprilgan_results/mvtec_zero_shot_20261006.log`, 공식 표 로그는 output의 `official/log.txt`.
- elapsed: 모델 로딩·전체 inference·공식 metric·시각화·raw evidence 저장까지 **{seconds:.1f}초 ({minutes:.1f}분)**. wrapper의 추가 저장 비용이 포함되어 공식 속도 benchmark가 아니다.

wrapper는 공식 `test()`를 호출하고 공식 `cal_pro_score()` 실행 시 실제 배열과 기존 unrounded metric을 수집했다. 공식 metric을 대체하거나 score를 다시 계산하지 않았다. AUPRO는 공식 함수의 threshold sweep과 FPR 범위를 유지했다.

## 3. 자체 측정 결과

아래는 15개 category의 비가중 단순 평균이다. %는 0–1 metric을 100배 한 값이다. 단일 seed이며 반복 평균·신뢰구간·통계적 유의성을 뜻하지 않는다.

| Metric | APRIL-GAN 내 PC (%) |
|---|---:|
| Image AUROC | {ia:.4f} |
| Image F1-max | {if1:.4f} |
| Image AP | {iap:.4f} |
| Pixel AUROC | {pa:.4f} |
| Pixel F1-max | {pf1:.4f} |
| Pixel AP | {pap:.4f} |
| AUPRO | {pro:.4f} |

## 4. 기존 MuSc 공식 실행과 category별 비교

Δ는 APRIL-GAN−MuSc이며 단위는 %p다. 각 row는 해당 category의 단일 실행이고 마지막 row는 15개 category의 비가중 평균이다. [비교 CSV](../source/results/mvtec_zero_shot_20261006/april_vs_local_musc.csv), [평균 비교 CSV](../source/results/mvtec_zero_shot_20261006/april_vs_local_musc_macro.csv), [집계 코드](../source/summarize_aprilgan.py)가 근거다.

{table}

## 5. 해석 범위

APRIL-GAN은 text prompt와 학습된 projection을 사용하는 반면 MuSc는 unlabeled test pool의 상호 비교를 사용한다. 정보 조건이 같다는 뜻은 아니다. 이번은 각 공식 구현체의 결과를 확보한 단계다.

특히 mask 처리 조건이 다르다. APRIL-GAN은 원본 mask를 이진화한 뒤 bilinear resize하고 0.5 threshold로 이진화한다. 이전 MuSc 공식 실행은 bilinear mask tensor를 최종 int32로 바꾼다. Pixel metric 차이를 순수 알고리즘의 기여라고 단정할 수 없다. 같은 raw mask·평가 처리로 다시 대조하는 것은 별도 후속 실험이다.

공통 framework의 MuSc 수치도 비교 CSV에 함께 남겼지만, 이번 APRIL-GAN을 공통 입력·metric으로 포팅한 결과라고 해석하지 않는다. 추가 학습, 4-shot 평가, VisA·BTAD 실행, 다중 seed 반복은 하지 않았다.
'''.format(commit=env['commit'], seconds=env['wall_seconds'], minutes=env['wall_seconds']/60,
           ia=100*float(means['image_auroc']), if1=100*float(means['image_f1_max']), iap=100*float(means['image_ap']),
           pa=100*float(means['pixel_auroc']), pf1=100*float(means['pixel_f1_max']), pap=100*float(means['pixel_ap']),
           pro=100*float(means['aupro']), table='\n'.join(table))
report = report.replace('元画像', '원본 이미지')
folder = ROOT.parent / 'markdown'
folder.mkdir(exist_ok=True)
path = folder / 'APRIL_GAN_mvtec_official_execution.md'
path.write_text(report)
print(path)
print(json.dumps(means, indent=2))
