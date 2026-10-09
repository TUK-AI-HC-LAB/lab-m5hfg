"""Refresh the original W41 brief after all common MuSc categories complete."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/w41_official_vs_common_all_20261006'
BRIEF = ROOT.parents[1] / 'meetings/2026-W41_brief.md'
summary = json.loads((OUT / 'summary.json').read_text())
assert summary['n_categories'] == 15 and summary['n_images'] == 1725
text = BRIEF.read_text()
if '### 실험 2: 15개 category 전체 비교' in text:
    raise SystemExit('All-category section already exists; review before updating again.')

def replace(old, new):
    global text
    assert old in text, 'Expected brief text missing: ' + old[:80]
    text = text.replace(old, new)

replace('공통 프레임워크에 MuSc를 세 파일로 등록했고, MVTec AD bottle의 테스트 83장에서 평가와 CSV 저장까지 완료했다. 공식 실행 대비 Image AUROC는 **−0.2381%p**, Pixel AUROC는 **+0.3522%p**다. 두 실행은 입력·정답 마스크·평가 경로가 달라서, 이 차이를 알고리즘 개선 또는 구현 오류의 증거로 단정할 수 없다. 공식 구현체는 15개 category를 완료했지만 공통 프레임워크는 bottle만 완료했다.',
        '공통 프레임워크에 MuSc를 세 파일로 등록했고, **MVTec AD 15개 category, 총 1,725장**의 평가와 CSV 저장까지 완료했다. bottle은 기존 2026-10-06 실행을 유지하고 나머지 14개를 같은 설정으로 실행했다. 공식 구현체 대비 15-category macro mean은 Image AUROC **{:+.4f}%p**, Pixel AUROC **{:+.4f}%p**다. 입력·정답 마스크·평가 경로가 달라 이 차이를 알고리즘 개선 또는 구현 오류로 단정할 수 없다.'.format(summary['image_delta_pp'], summary['pixel_delta_pp']))
replace('이번 bottle 실행에서 오류 없이 CSV를 생성했으며, 14개 나머지 category의 공통 실행은 아직 하지 않았다.',
        'bottle을 포함한 15개 category 모두 오류 없이 CSV를 생성했다. 이미지 1,725장의 해시·라벨·파일 순서가 공식 실행 당시와 같음을 별도 대조했다.')
replace('다음 단계의 evidence는 동일 조건에서의 score/map 대조와 나머지 category 실행 결과다.',
        '전체 category 실행 evidence를 확보했으며, 다음 단계는 동일 입력·정답·metric에서의 score/map 대조다.')
replace('| 전체 평균 비교는 아직 불가능함 | [category_coverage.csv](../method10/source/results/w41_official_vs_common_20261006/category_coverage.csv) | [기존 전체 CSV](../method10/source/results/mvtec_all_paper_20261002/category_metrics.csv) | 비교 범위를 bottle로 제한 |',
        '| 15-category 비교를 완료함 | [전체 실행 status](../method10/source/results/w41_official_vs_common_all_20261006/status.csv) | [category comparison](../method10/source/results/w41_official_vs_common_all_20261006/category_comparison.csv), [macro comparison](../method10/source/results/w41_official_vs_common_all_20261006/macro_comparison.csv) | 전체 관찰 차이와 category별 변동 확인 |\n| 동일 데이터 파일·라벨·순서임 | [검증 코드](../method10/source/audit_musc_common_data_w41.py) | [dataset identity](../method10/source/results/w41_official_vs_common_all_20261006/dataset_identity.csv) | 데이터 교체가 차이의 원인인지 점검 |')
replace('세 파일로 MuSc를 등록하고 bottle 전체 pool 실행을 완료했으며,', '세 파일로 MuSc를 등록하고 15개 category의 전체 pool 실행을 완료했으며,')
replace('문제 1. 동일 MVTec AD bottle 테스트 pool과 사전학습 CLIP이 주어졌을 때,',
        '문제 1. 동일 MVTec AD의 15개 category별 테스트 pool과 사전학습 CLIP이 주어졌을 때,')
replace('직접 대조 범위는 bottle 83장, seed 42의 각 1회 실행이다.',
        '직접 대조 범위는 15개 category 총 1,725장, seed 42의 category별 각 1회 실행이다. 아래 실험 1은 최초 bottle 통합 검증, 실험 2는 이후 15-category 확장 결과다.')
replace('동일 파일 집합에 대한 공통 실행의 이미지별 hash manifest와 raw prediction은 수집하지 않았다.',
        '후속 데이터 검증에서 공통 loader의 1,725장 모두 공식 실행의 이미지별 SHA-256·라벨·파일 순서와 일치했다. 공통 raw prediction은 수집하지 않았다.')
replace('공식 15-category macro mean은 Image AUROC 97.7694%, Pixel AUROC 97.1149%다([원본 mean CSV](../method10/source/results/mvtec_all_paper_20261002/mean_metrics.csv)). 공통의 나머지 14개 category는 미실행이므로 공통 macro mean이나 전체 개선율은 산출하지 않았다. 완료 범위는 [coverage CSV](../method10/source/results/w41_official_vs_common_20261006/category_coverage.csv)에서 확인할 수 있다.',
        '공식 15-category macro mean은 Image AUROC 97.7694%, Pixel AUROC 97.1149%다([원본 mean CSV](../method10/source/results/mvtec_all_paper_20261002/mean_metrics.csv)). 전체 공통 결과와의 대조는 실험 2에 정리했다. 기존 bottle-only [coverage CSV](../method10/source/results/w41_official_vs_common_20261006/category_coverage.csv)는 최초 작성 당시 기록으로 유지하며, 현재 완료 상태는 [전체 status](../method10/source/results/w41_official_vs_common_all_20261006/status.csv)가 기준이다.')
replace('현재 주장할 수 있는 것은 **공통 framework의 기존 입력·평가 경로로 MuSc bottle 전체 pool 실행을 완료했다**는 것이다.',
        '현재 주장할 수 있는 것은 **공통 framework의 기존 입력·평가 경로로 MuSc 15개 category 전체 pool 실행과 공식 실행 대비 category별 비교를 완료했다**는 것이다.')
replace('그 뒤 공통 조건으로 나머지 14개 category를 실행해 전체 coverage와 macro mean을 완성한다. 이러한 후속 실험은 이번 문서에서 계획한 사항이며 실행 완료로 기록하지 않는다.',
        '전체 category 실행과 macro mean 집계는 이번 후속 실행에서 완료했다. 동일 조건의 score/map 대조와 원인 분리 실험은 아직 계획 단계다.')
replace('**공통 전처리를 유지한 전체 category 확장보다, 동일 tensor·mask·metric에서 공식/포팅의 score-map 동등성을 먼저 확인하는 순서가 적절한가?**',
        '**전체 category 비교에서 관찰된 차이의 원인 분석을 위해, 동일 tensor·mask·metric의 score-map 대조를 다음 실험으로 우선하는 것이 적절한가?**')
replace('현재 판단은 동등성 대조를 먼저 하는 것이다. bottle의 두 AUROC가 반대 방향으로 변했고, mask와 입력 조건이 함께 달라 원인을 분리할 수 없기 때문이다. 다만 통합 coverage 확보가 더 시급한 연구 목표라면 전체 category 실행을 먼저 할 근거가 있다. 검토 결과에 따라 다음 실험을 원인 분리 또는 coverage 확장 중 하나로 좁힌다.',
        '현재 판단은 동일 조건 대조를 우선하는 것이다. 15개 category의 coverage는 확보했지만 mask와 입력 조건이 함께 달라 차이를 원인별로 분리할 수 없기 때문이다. 이 판단이 틀렸을 수 있는 이유는 현재 category별 차이가 입력보다 pool 구성·수치 계산에 더 크게 좌우될 가능성이다. 검토 결과에 따라 입력·mask 대조와 feature/scoring 대조 중 다음 실험의 우선순위를 좁힌다.')

highest_i = summary['image_higher_count']; lowest_i = summary['image_lower_count']; same_i = summary['image_same_count']
highest_p = summary['pixel_higher_count']; lowest_p = summary['pixel_lower_count']; same_p = summary['pixel_same_count']
section = '''### 실험 2: 15개 category 전체 비교

#### 질문·가설·기대

bottle에서 동작한 MuSc가 나머지 category에서도 동일 공통 실행 경로로 완료되는가? 전체 pool을 category별로 유지하면 15개 결과를 집계할 수 있다고 예상했다. 성능 차이의 방향은 미리 가정하지 않았으며, protocol 차이의 원인 검증은 이 실험의 범위가 아니다.

#### 설정과 근거

실험 1과 같은 method 설정, seed 42, batch size 1, 해상도 518, CLIP·stage·r·MSM·RsCIN을 유지했다. bottle CSV를 재사용하고 나머지 14개는 category별 별도 CLI process로 실행했다. 총 test 1,725장이다. framework source는 실행 전·후 SHA-256이 동일했다. 데이터 audit는 공통 loader와 공식 실행 당시의 이미지 파일 해시·라벨·순서를 대조했고 15개 모두 통과했다. 이는 raw mask나 변환 후 tensor의 동등성을 뜻하지 않는다.

- commit: 공통 codebase는 Git checkout이 아님; [실행 전·후 source hash와 명령](../method10/source/results/w41_official_vs_common_all_20261006/execution_manifest.json).
- script: [전체 실행 runner](../method10/source/run_musc_common_all_w41.py), [비교 생성 코드](../method10/source/compare_musc_all_w41.py), [데이터 audit](../method10/source/audit_musc_common_data_w41.py).
- result: [공통 category metrics](../method10/source/results/w41_official_vs_common_all_20261006/common_category_metrics.csv), [공식 category metrics](../method10/source/results/mvtec_all_paper_20261002/category_metrics.csv), [category comparison](../method10/source/results/w41_official_vs_common_all_20261006/category_comparison.csv), [macro comparison](../method10/source/results/w41_official_vs_common_all_20261006/macro_comparison.csv).
- status/log: [status.csv](../method10/source/results/w41_official_vs_common_all_20261006/status.csv). 원 로그는 `/home/test/musc_results/common_all_w41_20261006/<category>.log`; bottle은 기존 log를 재사용한다. 각 category의 원 CSV와 실행 시간도 status에 있다.
- dataset identity: [검증 CSV](../method10/source/results/w41_official_vs_common_all_20261006/dataset_identity.csv).

#### 실제 결과

각 행은 category 하나의 단일 실행이다. AUROC는 %, Δ는 공통−공식의 %p다. 마지막 행은 15개 category의 **비가중 단순 평균**이며 Test 수 1,725는 전체 이미지 수다. 통합 1,725장의 pooled AUROC를 뜻하지 않는다. 값의 원본은 위 category/macro CSV다.

''' + (OUT / 'category_table.md').read_text() + '''
Image AUROC는 {}개 상승, {}개 하락, {}개 동일이고 Pixel AUROC는 {}개 상승, {}개 하락, {}개 동일이다. 동일은 절댓값 Δ≤0.0001%p로 분류해 부동소수점 오차 수준의 차이를 제외했다. 표에서 동일 범위의 Δ는 0으로 표시하고 CSV에는 원래 차이를 보존했다. 이 개수는 유의성 판정이 아니다. F1-max·AP·AUPRO와 saliency CR F1은 [별도 CSV](../method10/source/results/w41_official_vs_common_all_20261006/nonmatching_metrics.csv)에 보존했고, 다른 정의의 F1끼리 차이를 계산하지 않았다.

#### 해석

15개 category 모두 실행·평가·CSV 저장을 완료했다. 전체 평균과 category별 차이는 관찰 evidence이며 알고리즘의 개선/열화 원인을 확정하지 않는다. 동일 데이터 확인으로 이미지 교체 가능성은 줄였지만 CLIP 입력 변환, mask 경계, implementation과 precision 차이는 여전히 함께 존재한다. 다음 검증은 차이가 큰 category와 bottle에서 동일 tensor·mask·metric으로 score/map을 대조하는 것이다.

'''.format(highest_i, lowest_i, same_i, highest_p, lowest_p, same_p)
largest_i = summary['image_largest_absolute_change']
largest_p = summary['pixel_largest_absolute_change']
section += '절댓값 기준 가장 큰 Image AUROC 변화는 **{} ({:+.4f}%p)**, Pixel AUROC 변화는 **{} ({:+.4f}%p)**다. 해당 category는 다음 원인 분리 대조의 우선 후보이며, 큰 변화 자체가 구현 오류나 개선을 증명하지는 않는다.\n\n'.format(largest_i['category'], largest_i['delta_pp'], largest_p['category'], largest_p['delta_pp'])
replace('## 6. 논의', section + '## 6. 논의')
BRIEF.write_text(text)
print(BRIEF)
