"""End-to-end MiniShift capsule A/B; swap only the FPFH all-point 120-NN."""
import hashlib
import os
import sys
import traceback
from pathlib import Path

import run_simple3d_capsule_flashfps as bench

ROOT = bench.HERE / 'results/capsule_step_flashfps_e2e_20261114'
ASSETS = Path('/home/test/simple3d-step-flashfps-e2e-20261114')
VARIANTS = ('kdtree', 'step')
STATE = {}


def worker_main(variant):
    import measure_simple3d_capsule_flashfps as worker
    import simple3d_gpu_kdtree as backend
    from simple3d_shapenet_hybrid_fps import install

    worker.ROOT = ROOT / variant
    worker.ASSETS = ASSETS / variant
    if variant == 'step':
        from measure_simple3d_capsule_step import indexing, step_search
        native_tree = backend.KDTree

        class FpfhOnlyTree:
            def __init__(self, xyz):
                self.xyz = xyz
                self.native = native_tree(xyz)
                self.replaced = 0

            def query(self, q, *, k, eps, p):
                if k != 120:
                    assert k in (128, 12)
                    return self.native.query(q, k=k, eps=eps, p=p)
                assert q is self.xyz and eps == 0 and p == 2 and self.replaced == 0
                self.replaced += 1
                def timed(fn):
                    backend.sync()
                    value = fn()
                    backend.sync()
                    return value
                idx = worker.observed('step_cluster_index_build',
                    lambda: timed(lambda: indexing(q,512,50)), 'gpu_wall', '512 clusters; 50 Lloyd updates')
                ids = worker.observed('step_query_k120',
                    lambda: timed(lambda: step_search(q,idx,512)), 'gpu_wall', 'exact 3D L2; distance/index order')
                # The stock FPFH path consumes only indices. No distances from
                # this call may be used (unlike the unchanged 128/12 paths).
                return None, ids.astype(ids.dtype,copy=False)

        backend.KDTree = FpfhOnlyTree
    worker.main(lambda native: install(native, 'hybrid'))


def main():
    if '--mode' in sys.argv:
        worker_main(os.environ['SIMPLE3D_STEP_VARIANT'])
        return
    if (ROOT/'progress.json').exists():
        raise RuntimeError('Existing run preserved')
    original_state = bench.state
    def state(**kwargs):
        STATE.update(kwargs)
        current = bench.ROOT
        bench.ROOT = ROOT
        try:
            original_state(**STATE)
        finally:
            bench.ROOT = current
    bench.state = state
    sources = ('simple3d_gpu_kdtree.py','simple3d_gpu_kdtree.cu','simple3d_step_index.cu',
               'measure_simple3d_capsule_step.py','simple3d_shapenet_hybrid_fps.py',
               'measure_simple3d_capsule_flashfps.py','run_simple3d_capsule_flashfps.py',Path(__file__).name)
    def hashes():
        return {n:hashlib.sha256((bench.HERE/n).read_bytes()).hexdigest() for n in sources}
    initial = hashes()
    outputs = {}
    state(status='preflight',completed_jobs=0,total_jobs=2)
    try:
        for smoke in (True,False):
            for variant in VARIANTS:
                os.environ['SIMPLE3D_STEP_VARIANT'] = variant
                bench.ROOT = ROOT / variant
                state(variant=variant)
                result = bench.run(bench.ITEM,smoke=smoke,worker=Path(__file__).resolve())
                if not smoke:
                    assert (result['train_count'],result['test_count']) == (54,130)
                    assert len(list(Path(result['raw_assets']).glob('*.npz'))) == 130
                    outputs[variant] = result
                    state(completed_jobs=len(outputs))
        a,b = [ROOT/v/'gpu_kdtree/minishift/capsule' for v in VARIANTS]
        assert bench.load(a/'manifest.json') == bench.load(b/'manifest.json') == bench.load(bench.BASE/'manifest.json')
        assert bench.load(a/'environment.json') == bench.load(b/'environment.json') and initial == hashes()
        bench.atomic(ROOT/'verification.json',dict(summaries=outputs,source_sha256=initial))
        state(status='complete_verified',phase='complete')
    except BaseException:
        state(status='failed',error=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
