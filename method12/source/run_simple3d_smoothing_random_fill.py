"""One-category controlled smoothing-fill ablation; reuse the ordered-fill bank."""
import hashlib
import sys
import traceback
from pathlib import Path

import run_simple3d_capsule_flashfps as bench

ROOT = bench.HERE/'results/capsule_smoothing_random_fill_20261126'
ASSETS = Path('/home/test/simple3d-capsule-smoothing-random-fill-20261126')
BASE = bench.HERE/'results/capsule_gpu_kdtree_flashfps_random25_feature50_20261111/gpu_kdtree/minishift/capsule'


def main():
    if '--mode' in sys.argv:
        import measure_simple3d_capsule_flashfps_smooth_original as worker
        from simple3d_flashfps_smoothing_random_fill import install

        worker.ROOT, worker.ASSETS = ROOT, ASSETS
        worker.main(fps_install=install)
        return

    if (ROOT/'progress.json').exists():
        raise RuntimeError('Existing run preserved')
    bench.ROOT = ROOT
    worker = Path(__file__).resolve()
    sources = [worker, bench.HERE/'simple3d_flashfps_smoothing_random_fill.py',
               bench.HERE/'measure_simple3d_capsule_flashfps_smooth_original.py',
               bench.HERE/'simple3d_flashfps_random25_feature50.py',
               bench.HERE/'simple3d_flashfps_random25.py', bench.HERE/'simple3d_flashfps.py']
    hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    try:
        bench.state(status='preflight', completed_jobs=0, target_jobs=1)
        bench.run(bench.ITEM, smoke=True, worker=worker)
        result = bench.run(bench.ITEM, worker=worker)
        leaf = ROOT/'gpu_kdtree/minishift/capsule'
        assert bench.load(leaf/'manifest.json') == bench.load(BASE/'manifest.json')
        env = bench.load(leaf/'environment.json')
        assert env['fps']['smoothing_fill_count'] == 768
        assert env['memory_bank']['sha256'] == hashlib.sha256(
            Path(env['memory_bank']['path']).read_bytes()).hexdigest()
        assert (result['train_count'], result['test_count']) == (54, 130)
        assert hashes == {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        bench.atomic(ROOT/'verification.json', dict(status='verified', source_sha256=hashes,
                     memory_bank=env['memory_bank'], fps=env['fps']))
        bench.state(status='complete_verified', completed_jobs=1, phase='complete')
    except BaseException:
        bench.state(status='failed', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
