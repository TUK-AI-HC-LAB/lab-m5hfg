"""Isolated G variant: only FPFH neighborhood scales are 20/40/60."""
import cupy as cp
from cupyx.scipy.spatial import KDTree
import numpy as np
import open3d as o3d
import torch
from simple3d_gpu_kdtree import sync, torch_view, descriptors

CACHE={}

def install(run, detail_tracing=False):
    module=run.fpfh_module
    def observed(name,fn,detail):
        # CuPy can use another CUDA stream; synchronized wall timing avoids
        # falsely attributing work to a PyTorch event on a different stream.
        def wrapped():
            sync();result=fn();sync();return result
        return run.observed(name,wrapped,'gpu_wall',detail)

    class TreeKNN:
        def __init__(self,k,transpose_mode=True):
            if not transpose_mode:raise ValueError('Expected B,N,3')
            self.k=k
        def __call__(self,reference,query):
            assert reference.is_cuda and query.is_cuda
            if reference.data_ptr()!=CACHE['dense_ptr']:
                raise RuntimeError('GPU KD-tree must match the current full point cloud')
            torch.cuda.synchronize()
            queries=cp.from_dlpack(query[0].contiguous())
            def search():
                d,i=CACHE['tree'].query(queries,k=self.k,eps=0,p=2)
                return torch_view(d.astype(cp.float32))[None],torch_view(i)[None]
            return observed('coordinate_knn_gpu_kdtree',search,
                            f'CuPy exact GPU KDTree k={self.k}; query={query.shape[1]}')

    def get_features(self,unorganized_pc,voxel_size=.1):
        assert self.args.max_nn==20 and self.args.use_MSND and self.args.num_MSND==2
        xyz_cpu=unorganized_pc.squeeze(0).numpy()
        if xyz_cpu.ndim!=2 or xyz_cpu.shape[1]!=3 or len(xyz_cpu)<max(60,self.args.num_group):
            raise ValueError(f'Invalid full-resolution point cloud shape {xyz_cpu.shape}')
        if np.linalg.norm(xyz_cpu.max(0)-xyz_cpu.min(0))>=10000:
            raise RuntimeError('Nonbinding official radius assumption failed')
        legacy=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_cpu))
        run.observed('normal_estimation_including_neighbor_search',
            lambda:legacy.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=10000,max_nn=10)),
            'cpu','Official legacy CPU normals max_nn=10 radius=10000')
        CACHE.clear()
        xyz,normals=observed('positions_normals_upload',lambda:
            (cp.asarray(xyz_cpu,dtype=cp.float32),cp.asarray(np.asarray(legacy.normals).astype(np.float32))),
            'CPU coordinates and normals to CuPy GPU Float32')
        tree=observed('gpu_kdtree_build',lambda:KDTree(xyz),'CuPy GPU tree construction')
        def all_neighbors():
            d,i=tree.query(xyz,k=60,eps=0,p=2)
            return i.astype(cp.int32)
        indices=observed('gpu_kdtree_all_points_k60',all_neighbors,
                         f'Exact GPU KDTree full {len(xyz_cpu)} point query; reuse prefixes 20/40/60')
        scales=[]
        for k in (20,40,60):
            value=observed(f'fpfh_gpu_kdtree_scale_{k}',lambda k=k:descriptors(xyz,normals,indices,k),
                           f'Open3D-derived Float32 SPFH+FPFH CUDA; k={k}')
            scales.append(torch_view(value))
        concatenate=lambda:torch.cat([scales[0],scales[1],scales[1],scales[2]],dim=-1)
        features=observed('feature_scale_concatenation',concatenate,'GPU cat 20/40/40/60') if detail_tracing else concatenate()
        dense=torch_view(xyz)[None].contiguous()
        CACHE.update(tree=tree,dense_ptr=dense.data_ptr())
        centers,_=module.fps(dense, self.args.num_group)
        _,ix=TreeKNN(self.args.group_size)(dense,centers)
        gather=lambda:features[ix.reshape(-1)].reshape(1,self.args.num_group,self.args.group_size,132)
        neighborhoods=observed('neighborhood_feature_gather',gather,'GPU gather center-neighbor features') if detail_tracing else gather()
        aggregate=torch.mean(neighborhoods,-2).squeeze()
        return aggregate,torch.tensor(xyz_cpu),dense,centers

    run.base.FPFHFeatures.get_fpfh_features=get_features
    run.base.features_module.KNN=TreeKNN
