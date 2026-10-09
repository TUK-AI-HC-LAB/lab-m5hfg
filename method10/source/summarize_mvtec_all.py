"""Build paper comparison exclusively from verified category CSVs."""
import csv,json,math
from pathlib import Path
BASE=Path(__file__).resolve().parent
RESULT=BASE/'results/mvtec_all_paper_20261002'
KEYS=['image_auroc','image_f1_max','image_ap','pixel_auroc','pixel_f1_max','pixel_ap','aupro']
NAMES=['Image AUROC','Image F1-max','Image AP','Pixel AUROC','Pixel F1-max','Pixel AP','AUPRO']
rows=list(csv.DictReader((RESULT/'category_metrics.csv').open()))
assert len(rows)==15 and len({r['category'] for r in rows})==15
paper={r['category']:r for r in csv.DictReader((RESULT/'paper_table17.csv').open())}
mean=next(csv.DictReader((RESULT/'mean_metrics.csv').open()))
audit={r['category']:r for r in csv.DictReader((RESULT/'dataset_audit.csv').open())}
statuses=list(csv.DictReader((RESULT/'status.csv').open()))
assert all(s['status']=='completed' for s in statuses) and len(statuses)==15
comparison=[]
envs={}
for r in rows:
    cat=r['category']
    assert int(r['n_images'])==int(audit[cat]['test_images'])
    scores=list(csv.DictReader((RESULT/cat/'image_scores.csv').open()))
    assert len(scores)==int(r['n_images'])
    assert sum(int(s['label']) for s in scores)==int(audit[cat]['abnormal'])
    env=json.loads((RESULT/cat/'environment.json').read_text())
    assert env['status']=='completed'
    envs[cat]=env
    for k in KEYS:
        actual=float(r[k])*100
        # sklearn AP can return 1 + machine epsilon; preserve the raw value.
        assert math.isfinite(actual) and -1e-10<=actual<=100+1e-10
        reference=float(paper[cat][k])
        comparison.append(dict(category=cat,metric=k,paper_percent=reference,local_percent=actual,delta_percentage_points=actual-reference))
for k in KEYS:
    assert abs(float(mean[k])-sum(float(r[k]) for r in rows)/15)<1e-12
    comparison.append(dict(category='Mean',metric=k,paper_percent=float(paper['Mean'][k]),local_percent=float(mean[k])*100,delta_percentage_points=float(mean[k])*100-float(paper['Mean'][k])))
with (RESULT/'paper_vs_local_all.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(comparison[0]));w.writeheader();w.writerows(comparison)
with (RESULT/'execution_costs.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['category','n_images','wall_seconds_including_load_save_metrics','peak_allocated_bytes','peak_reserved_bytes'])
    for r in rows:
        e=envs[r['category']]
        w.writerow([r['category'],r['n_images'],e['wall_seconds_including_model_loading_and_metrics'],e['peak_allocated_bytes'],e['peak_reserved_bytes']])
batch=json.loads((RESULT/'batch_summary.json').read_text())
lines=['# MuSc MVTec AD 전체 category 실행 및 논문 비교',
       '', '## 실제 결과와 판단', '',
       f'MVTec AD 15개 category, test 이미지 {sum(int(r["n_images"]) for r in rows):,}장 평가를 완료했다. 정상 train은 사용하지 않고 category별 전체 unlabeled test pool로 scoring했다. 완료된 bottle은 기존 실행을 재사용했고 나머지 14개는 같은 설정으로 새로 실행했다. VisA와 ablation은 이번 결과에 포함하지 않는다.',
       '', '## 전체 평균: 논문 Table 17과 비교', '',
       '모든 값은 %. Δ = 내 실행 − 논문 공개값, 단위 %p. 각 category 지표의 동일 가중 평균(macro mean)이며 모든 픽셀을 한꺼번에 합친 micro 지표가 아니다. 논문은 소수점 한 자리만 공개하므로 Δ에는 논문 반올림 오차가 포함된다.',
       '', '| 지표 | 논문 (%) | 내 실행 (%) | Δ (%p) |', '|---|---:|---:|---:|']
for k,n in zip(KEYS,NAMES):
    actual=float(mean[k])*100; ref=float(paper['Mean'][k])
    lines.append(f'| {n} | {ref:.1f} | {actual:.4f} | {actual-ref:+.4f} |')
lines += ['', '## Category별 실제 결과', '',
          '지표 순서는 Image AUROC, Image F1-max, Image AP, Pixel AUROC, Pixel F1-max, Pixel AP, AUPRO. 모든 값은 %다.', '',
          '| Category | 이미지 수 | I-AUROC | I-F1 | I-AP | P-AUROC | P-F1 | P-AP | AUPRO |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append('| '+r['category']+' | '+r['n_images']+' | '+' | '.join(f'{float(r[k])*100:.4f}' for k in KEYS)+' |')
lines += ['', '## Category별 논문 대비 차이', '',
          '다음 표의 모든 값은 Δ(%p)다. 음수는 내 실행 값이 낮다는 의미다. 논문 Table 17의 같은 category 행과 비교했다.', '',
          '| Category | I-AUROC | I-F1 | I-AP | P-AUROC | P-F1 | P-AP | AUPRO |',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append('| '+r['category']+' | '+' | '.join(f'{float(r[k])*100-float(paper[r["category"]][k]):+.4f}' for k in KEYS)+' |')
worst=max((c for c in comparison if c['category']!='Mean'),key=lambda c:abs(c['delta_percentage_points']))
lines += ['',f'공개 반올림 값 기준 가장 큰 절대 차이는 {worst["category"]}의 {NAMES[KEYS.index(worst["metric"])]}: {worst["delta_percentage_points"]:+.4f}%p다. 차이의 원인은 코드 버전·환경·전처리·metric을 하나씩 고정해 추가 진단해야 하며 이 실행만으로 확정하지 않는다.',
          '', '## 설정과 환경', '',
          '- 공식 commit: `b76b93da8bd3096a99964a96ae29d46f197a0651`.',
          '- RTX 5080 / WSL2 / Python 3.12.13 / PyTorch 2.11.0+cu128 / CUDA 12.8.',
          '- OpenAI CLIP ViT-L/14-336, 입력 518×518, feature layers `[5,11,17,23]`, aggregation `[1,3,5]`.',
          '- IA 낮은 30%, RsCIN 공식 MVTec `[1,2,3]`, batch 4, seed 42, TF32 off, 공식 autocast 유지.',
          '- 모든 category에서 divide_num=1. 해상도 축소·pool 분할·scoring 알고리즘 변경 없음.',
          '- 논문 당시 환경과 dependency 버전 및 출판 결과 생성 commit은 동일성 미확인. 공식 scoring/loader/metric 코드는 수정하지 않았다.',
          '- AUPRO는 공식 200-threshold 및 FPR<0.3 선택 구간 min–max 정규화 구현을 사용한다. mask 변환 역시 공식 loader를 유지한다.',
          '', '## 시간과 GPU 메모리', '',
          f'재개한 마지막 batch wall time: {batch["wall_seconds_this_batch"]:.2f}초 ({batch["wall_seconds_this_batch"]/60:.2f}분). 이미 완료된 10개 category는 재사용하고 남은 5개를 실행한 시간이다. 중단 이전 대기 시간까지 포함한 최초 시작부터의 경과 시간은 아니다.',
          f'최종 채택된 15개 category의 성공한 실행 wall time 합계: {sum(e["wall_seconds_including_model_loading_and_metrics"] for e in envs.values()):.2f}초 ({sum(e["wall_seconds_including_model_loading_and_metrics"] for e in envs.values())/60:.2f}분). bottle을 포함하고 중단된 tile 시도는 제외한다. category별 시간에는 모델 로딩·raw 저장·metric·시각화가 포함되며, 각 실행 wrapper 바깥의 interpreter 시작 비용은 제외된다.',
          '최초 batch는 tile의 지표 계산 도중 종료되어 2026-10-03 재개했다. 중단된 출력은 `/home/test/musc_results/mvtec_all_paper_20261002/tile_interrupted_20261003/`에 보존하고 tile부터 새로 실행했다. 최초 완료 상태는 `status_before_resume.csv`에 남겼다.',
          '', '| Category | 전체 실행 (초) | Peak allocated (GiB) | Peak reserved (GiB) |', '|---|---:|---:|---:|']
for r in rows:
    e=envs[r['category']]
    lines.append(f'| {r["category"]} | {e["wall_seconds_including_model_loading_and_metrics"]:.2f} | {e["peak_allocated_bytes"]/2**30:.3f} | {e["peak_reserved_bytes"]/2**30:.3f} |')
lines += ['', 'Peak는 model load 후 reset한 PyTorch allocator 값이며 GPU 전체 사용량이 아니다. warm-up·반복·동일 측정 범위를 갖추지 않았으므로 논문 RTX 3090 대비 speedup을 주장하지 않는다.',
          '', '## 검증 및 근거 경로', '',
          '- 15개 category 완료 상태, 이미지별 score 행 수, 정상/이상 count, 지표 범위와 macro 평균을 자동 확인했다.',
          '- commit: `b76b93da8bd3096a99964a96ae29d46f197a0651`',
          '- sh: [run_mvtec_all.sh](../source/run_mvtec_all.sh)',
          '- result: [category_metrics.csv](../source/results/mvtec_all_paper_20261002/category_metrics.csv), [mean_metrics.csv](../source/results/mvtec_all_paper_20261002/mean_metrics.csv)',
          '- 논문 raw table: [paper_table17.csv](../source/results/mvtec_all_paper_20261002/paper_table17.csv)',
          '- 모든 지표 비교: [paper_vs_local_all.csv](../source/results/mvtec_all_paper_20261002/paper_vs_local_all.csv)',
          '- 데이터 점검: [dataset_audit.csv](../source/results/mvtec_all_paper_20261002/dataset_audit.csv)',
          '- 실행 상태: [status.csv](../source/results/mvtec_all_paper_20261002/status.csv)',
          '- 시간/메모리 raw: [execution_costs.csv](../source/results/mvtec_all_paper_20261002/execution_costs.csv)',
          '- category별 CSV·image scores·config·environment: `../source/results/mvtec_all_paper_20261002/<category>/`.',
          '- 대용량 NPZ·로그·heatmap 원본: `/home/test/musc_results/mvtec_all_paper_20261002/<category>/`. bottle만 `/home/test/musc_results/bottle_paper_20261002/`.',
          '- 총괄 로그: `/home/test/musc_results/mvtec_all_paper_20261002/batch.log`.',
          '- 논문 근거: [공식 PDF](../paper/ICLR24_MuSc_Zero_Shot_Industrial_Anomaly_Classification_and_Segmentation_with_Mutual_Scoring_of_the_Unlabeled_Images.pdf), p.20 Table 17 [1].',
          '', '## 참고문헌', '', '[1] Li, Xurui, et al. "MuSc: Zero-Shot Industrial Anomaly Classification and Segmentation with Mutual Scoring of the Unlabeled Images." International Conference on Learning Representations, 2024.', '']
out=BASE.parent/'markdown/MuSc_paper_vs_local_mvtec_all.md'
out.write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(dict(report=str(out),mean_percent={k:float(mean[k])*100 for k in KEYS},batch_wall_seconds=batch['wall_seconds_this_batch']),indent=2))
