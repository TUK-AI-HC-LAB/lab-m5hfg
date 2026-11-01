"""Open3D tensor CUDA descriptor path, not the upstream legacy CPU API.

The huge upstream radius bounds contain this complete cloud. Exact max_nn-only
queries therefore express the same unrestricted nearest-neighbor objective.
Tensor vs legacy eigensolver/FPFH numerical differences are NOT bitwise equality.
MSND duplicate channels remain explicit: this changes devices, not concatenation.
"""
import numpy as np
import os
import open3d as o3d
import torch


def install(run):
    module = run.fpfh_module
    device = o3d.core.Device('CUDA:0')
    if not o3d.core.cuda.is_available():
        raise RuntimeError('Open3D CUDA unavailable; refusing CPU fallback')

    def get_fpfh_features(self, unorganized_pc, voxel_size=.1):
        xyz_cpu = unorganized_pc.squeeze(0).numpy()
        assert xyz_cpu.shape[1] == 3
        extent = float(np.linalg.norm(xyz_cpu.max(0)-xyz_cpu.min(0)))
        if extent >= 10000:
            raise RuntimeError('Cannot omit upstream radius constraint: cloud extent too large')
        # Official loader input is float32; legacy CPU internally promotes to double.
        # This explicit option keeps a slow Float64 trial separate from Float32.
        use_double = os.environ.get('SIMPLE3D_O3D_DTYPE', 'float32') == 'float64'
        cloud = o3d.t.geometry.PointCloud(o3d.core.Tensor(
            xyz_cpu.astype(np.float64 if use_double else np.float32),
            dtype=o3d.core.Dtype.Float64 if use_double else o3d.core.Dtype.Float32, device=device))

        def normals():
            cloud.estimate_normals(max_nn=10)
            o3d.core.cuda.synchronize()
            assert str(cloud.point.normals.device) == 'CUDA:0'

        run.observed('normal_estimation_including_neighbor_search', normals, 'cuda:0',
                     'Open3D tensor CUDA; exact max_nn=10; omitted nonbinding radius=10000')
        scales = []
        for k in (self.args.max_nn, 2*self.args.max_nn, 3*self.args.max_nn):
            def compute(k=k):
                value = o3d.t.pipelines.registration.compute_fpfh_feature(cloud, max_nn=k)
                o3d.core.cuda.synchronize()
                assert str(value.device) == 'CUDA:0' and list(value.shape) == [len(xyz_cpu), 33]
                return value
            value = run.observed(f'fpfh_scale_{k}', compute, 'cuda:0',
                                 f'Open3D tensor CUDA; exact max_nn={k}; omitted nonbinding radius=1000000')
            tensor = torch.utils.dlpack.from_dlpack(value.to_dlpack()).to(torch.float32)
            assert tensor.is_cuda
            scales.append(tensor)
        # Preserve upstream official code's repeated 80-neighbor descriptor.
        features = torch.cat([scales[0], scales[1], scales[1], scales[2]], dim=-1)
        xyz = torch.from_numpy(xyz_cpu).to(self.args.device).unsqueeze(0).contiguous()
        centers, _ = module.fps(xyz, self.args.num_group)
        knn = module.KNN(k=self.args.group_size, transpose_mode=True)
        indices = module.batched_knn(knn, xyz, centers)
        neighborhoods = features.reshape(-1, 132)[indices.reshape(-1), :]
        neighborhoods = neighborhoods.reshape(1, self.args.num_group, self.args.group_size, 132).contiguous()
        aggregate = torch.mean(neighborhoods, -2).squeeze()
        return aggregate, torch.tensor(xyz_cpu), xyz, centers

    run.base.FPFHFeatures.get_fpfh_features = get_fpfh_features
