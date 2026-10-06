"""Run each missing MuSc category through the unchanged common CLI; resume safely."""
import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--framework', type=Path, required=True, help='Common-framework checkout containing main.py.')
parser.add_argument('--python', dest='python_executable', required=True, help='Python executable for the common framework.')
parser.add_argument('--data', type=Path, required=True, help='MVTec AD dataset root.')
parser.add_argument('--run-root', type=Path, required=True, help='Fresh external directory for runtime outputs and logs.')
parser.add_argument('--reuse-bottle-root', type=Path, required=True, help='Existing bottle output directory to reuse.')
args = parser.parse_args()
FRAMEWORK = args.framework
PYTHON = args.python_executable
RUN = args.run_root
EVIDENCE = ROOT / 'results/w41_official_vs_common_all_20261006'
RUN.mkdir(parents=True, exist_ok=True)
EVIDENCE.mkdir(parents=True, exist_ok=True)
with (ROOT / 'results/mvtec_all_paper_20261002/category_metrics.csv').open() as file:
    categories = list(csv.DictReader(file))
assert len(categories) == 15 and sum(int(r['n_images']) for r in categories) == 1725
sources = [FRAMEWORK / p for p in ('trainer/trainer_musc.py', 'configs/musc.yaml', 'component_registry.py',
                                   'datasets/base.py', 'trainer/trainer.py', 'metrics_gpu.py')]
hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
manifest = {'started': '2026-10-06', 'framework': str(FRAMEWORK), 'python': PYTHON,
            'seed': 42, 'data': str(args.data), 'sha256_before': hashes,
            'bottle_reused_from': str(args.reuse_bottle_root),
            'runs': []}
status = []

def save():
    (EVIDENCE / 'execution_manifest.json').write_text(json.dumps(manifest, indent=2))
    with (EVIDENCE / 'status.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['category', 'n_images', 'status', 'seconds', 'returncode', 'csv', 'log'])
        writer.writeheader(); writer.writerows(status)

for category in categories:
    name = category['category']
    output = args.reuse_bottle_root if name == 'bottle' else RUN / name
    result = output / 'musc__layer2_layer3_/results_musc.csv'
    log = args.reuse_bottle_root.with_suffix('.log') if name == 'bottle' else RUN / (name + '.log')
    command = [PYTHON, 'main.py', '--method', 'musc', '--dataset', 'mvtec', '--category', name,
               '--data-path', str(args.data), '--results-path', str(output), '--seed', '42', '--num-workers', '1']
    if result.exists():
        code, seconds, state = 0, None, 'reused'
    else:
        print('START', name, category['n_images'], flush=True)
        started = time.monotonic()
        with log.open('w') as file:
            file.write('COMMAND: ' + ' '.join(command) + '\n'); file.flush()
            completed = subprocess.run(command, cwd=FRAMEWORK, stdout=file, stderr=subprocess.STDOUT)
        seconds, code = time.monotonic() - started, completed.returncode
        state = 'completed' if code == 0 and result.exists() else 'failed'
    if state != 'failed':
        with result.open() as file:
            rows = list(csv.DictReader(file))
        assert len(rows) == 1 and rows[0]['dataset_name'] == 'mvtec_' + name
        assert int(rows[0]['seed']) == 42
        for key in ('auroc_mean', 'pixel_auroc_mean'):
            assert 0 <= float(rows[0][key]) <= 1
        # Independently recorded common loader size must match official dataset count.
        assert 'test=' + category['n_images'] in log.read_text()
        target = EVIDENCE / name
        target.mkdir(exist_ok=True)
        (target / 'results_musc.csv').write_bytes(result.read_bytes())
    status.append(dict(category=name, n_images=category['n_images'], status=state,
                       seconds=seconds, returncode=code, csv=str(result), log=str(log)))
    manifest['runs'].append(dict(category=name, command=command, status=state, seconds=seconds,
                                 result=str(result), log=str(log)))
    save()
    print(state.upper(), name, seconds, flush=True)

manifest['sha256_after'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
assert manifest['sha256_after'] == hashes, 'Framework source changed during execution'
manifest['complete'] = all(r['status'] != 'failed' for r in status)
save()
if not manifest['complete']:
    raise SystemExit('One or more categories failed; inspect status/logs and resume.')
print('ALL 15 COMPLETE', flush=True)
