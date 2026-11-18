"""Reproducible macro stage ranking after the completed 52-category Top-3 run."""
import collections
import csv
import statistics
from pathlib import Path
import run_simple3d_capsule_flashfps as bench

ROOT=bench.HERE/'results/post_top3_bottleneck_20261118'
SOURCE=bench.HERE/'results/all_selective_top3_20261118'
REPORT=bench.HERE.parent/'markdown/W48_Simple3D_post_Top3_bottleneck_and_pilot.md'


def analyze():
    ROOT.mkdir(parents=True,exist_ok=True)
    assert bench.load(SOURCE/'progress.json')['status']=='complete_verified'
    with (SOURCE/'category_comparison.csv').open() as f: categories=list(csv.DictReader(f))
    count={(r['dataset'],r['category']):int(r['top3_test_count']) for r in categories}
    overall={d:statistics.mean(float(r['top3_inference_mean_ms']) for r in categories if r['dataset']==d)
             for d in ('minishift','shapenet')}
    groups=collections.defaultdict(lambda:collections.defaultdict(float))
    with (SOURCE/'stage_comparison.csv').open() as f:
        for r in csv.DictReader(f):
            if r['variant']!='top3' or r['phase']!='test': continue
            detail=''
            if r['stage']=='fps': detail='output_count='+r['detail'].split('output_count=')[-1]
            if r['stage']=='coordinate_knn_gpu_kdtree': detail=r['detail'].split(';')[0]
            if r['stage']=='point_score_interpolation':
                detail='4096 centers' if '[1, 3, 4096]' in r['detail'] else '1024 centers'
            groups[(r['dataset'],r['stage'],detail)][r['category']]+=1000*float(r['total_s'])/count[r['dataset'],r['category']]
    rows=[]
    for (d,s,detail),v in groups.items():
        assert len(v)==(12 if d=='minishift' else 40)
        ms=statistics.mean(v.values())
        rows.append(dict(dataset=d,stage=s,detail=detail,categories=len(v),macro_ms=ms,
                         ratio_to_overall_percent=100*ms/overall[d]))
    rows.sort(key=lambda r:(r['dataset'],-r['macro_ms']))
    bench.writecsv(ROOT/'stage_ranking.csv',rows)
    lines=['# W48 Simple3D — Top-3 이후 병목과 후속 실험','',
        '## 확인한 문제','',
        '52개 카테고리 비교 완료 결과의 선택 Top-3판만 집계함. 각 단계의 카테고리별 테스트 총 시간을 테스트 수로 나눈 뒤 카테고리 동일 가중 평균을 계산함.',
        'FPFH용 120-NN이 두 데이터셋 모두 가장 큰 계측 구간임. 구조 생성과 조회를 분리해서 개선 후보를 검토함.',
        '비중은 단계 평균/전체 추론 평균의 참고값임. 하위 구간이 상위 구간과 중첩되며 입력 읽기처럼 전체 추론 밖의 단계도 있으므로 모든 행을 합산하면 안 됨.','',
        '| 데이터셋 | 단계 | 구분 | 평균 (ms/샘플) | 전체 추론 대비 참고 비중 (%) |',
        '|---|---|---|---:|---:|']
    for r in rows:
        if r['macro_ms']>=4:
            lines.append(f"| {r['dataset']} | {r['stage']} | {r['detail']} | {r['macro_ms']:.3f} | {r['ratio_to_overall_percent']:.2f} |")
    lines += ['', '## 원인 가설과 선택한 시험','',
        '- H1: STEP 계열 독립 구현에서 쿼리마다 512개 군집 하한을 완전 정렬하는 비용 중 일부가 불필요함.',
        '- H2: 완전 정렬을 없애면 더 먼 군집을 일찍 방문해 후보 거리 계산이 늘어날 수 있어 오히려 느려질 수 있음.',
        '- 선택: 가장 작은 하한의 군집을 먼저 방문한 뒤 나머지를 군집 번호순으로 검사하는 시험. 현재 120번째 거리보다 군집 하한이 큰 경우에만 건너뜀.',
        '- 유지: STEP 512·Lloyd 50회·동일 인덱스·Float64 거리/하한·거리/원본 인덱스 동점 순서·FPFH 40/80/120·FlashFPS·집계/평활화 설정. 공식 STEP 코드 자체를 재현한 구현은 아님.',
        '- 기대: 이웃 인덱스와 FPFH가 완전히 같고 실제 조회 시간이 감소하면 전체 파이프라인 시험으로 진행함. 조회가 느려지면 이 후보는 제외하고 이유를 남김.',
        '- 독립 연산 시험은 전체 추론·AUROC/AP 결과가 아니며 속도 개선도 인덱스 생성 시간을 포함해 해석함.',
        '', '## 재현 근거','',
        '- 기준 commit: `66956cf11839902944dffc5082210cb859f9b5f8` (새 파일은 아직 미커밋, 실행 소스 해시를 별도 기록).',
        '- [분석 코드](../source/analyze_simple3d_post_top3.py) · [실행 스크립트](../source/run_simple3d_post_top3_pilot.sh)',
        '- [전체 단계 집계 CSV](../source/results/post_top3_bottleneck_20261118/stage_ranking.csv)',
        '- [52개 원본 결과](../source/results/all_selective_top3_20261118/stage_comparison.csv)',
        '- [후속 연산 시험](../source/results/post_top3_bottleneck_20261118/pilot/)',
        '- 새로움은 미확인. 실험 결과만으로 논문 기여라고 주장하지 않음.']
    tmp=REPORT.with_suffix('.md.tmp');tmp.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    bench.replace_with_retry(tmp,REPORT)
    return rows


if __name__=='__main__': analyze()
