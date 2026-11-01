"""RTX 5080 compatibility adapters; no learned-feature or sampling-budget changes.

Replaces old Pointnet2/KNN_CUDA binaries with NVRTC FPS and exact scipy KD-tree
queries. Interpolation uses exact nearest-three topk instead of full sort.
These backends differ from the paper, so timing is local implementation timing.
"""
import sys
import types
import numpy as np
import torch
from scipy.spatial import cKDTree

FPS_CODE = r'''
extern "C" __global__ void fps_kernel(const float* xyz, float* temp, int* out,
                                      const int n, const int m) {
 __shared__ float best[512];
 __shared__ int inds[512];
 const int t=threadIdx.x, b=blockIdx.x;
 xyz += b*n*3; temp += b*n; out += b*m;
 int old=0;
 if(t==0) out[0]=0;
 for(int j=1;j<m;j++) {
   float x=xyz[old*3], y=xyz[old*3+1], z=xyz[old*3+2];
   float bd=-1.0f; int bi=0;
   for(int k=t;k<n;k+=512) {
     float a=xyz[k*3], c=xyz[k*3+1], e=xyz[k*3+2];
     if(a*a+c*c+e*e <= 1e-3f) continue;
     float dd=(a-x)*(a-x)+(c-y)*(c-y)+(e-z)*(e-z);
     dd=fminf(dd,temp[k]); temp[k]=dd;
     if(dd>bd) {bd=dd;bi=k;}
   }
   best[t]=bd; inds[t]=bi; __syncthreads();
   for(int step=256;step>0;step/=2) {
     if(t<step && best[t+step]>best[t]) {best[t]=best[t+step];inds[t]=inds[t+step];}
     __syncthreads();
   }
   old=inds[0]; if(t==0) out[j]=old; __syncthreads();
 }
}
'''

def furthest_point_sample(data, number):
    import cupy as cp
    if not hasattr(furthest_point_sample, 'kernel'):
        furthest_point_sample.kernel = cp.RawKernel(FPS_CODE, 'fps_kernel')
    b,n,_ = data.shape
    out=torch.empty((b,number),device=data.device,dtype=torch.int32)
    temp=torch.full((b,n),1e10,device=data.device,dtype=torch.float32)
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        furthest_point_sample.kernel((b,), (512,),
            (cp.from_dlpack(data),cp.from_dlpack(temp),cp.from_dlpack(out),np.int32(n),np.int32(number)))
    return out

def gather_operation(features, idx):
    return torch.gather(features,2,idx.long().unsqueeze(1).expand(-1,features.shape[1],-1))

class KNN:
    def __init__(self,k,transpose_mode=True): self.k=k;self.transpose_mode=transpose_mode
    def __call__(self,reference,query):
        ref=reference if self.transpose_mode else reference.transpose(1,2)
        qry=query if self.transpose_mode else query.transpose(1,2)
        ds=[]; ids=[]
        for r,q in zip(ref,qry):
            dist,idx=cKDTree(r.detach().cpu().numpy()).query(q.detach().cpu().numpy(),k=self.k,workers=4)
            if self.k==1: dist=dist[:,None];idx=idx[:,None]
            ds.append(dist);ids.append(idx)
        return (torch.as_tensor(np.array(ds),dtype=ref.dtype,device=ref.device),
                torch.as_tensor(np.array(ids),dtype=torch.long,device=ref.device))

def install():
    import os
    if os.environ.get('SIMPLE3D_NATIVE_OPS')=='1':
        # Import compiled native dependencies; no CPU kNN or NVRTC FPS substitution.
        from pointnet2_ops import pointnet2_utils
        from knn_cuda import KNN as NativeKNN
        quan=types.ModuleType('data.quan');quan.get_quan_loader=None;quan.quan_classes=lambda:[]
        sys.modules['data.quan']=quan
        return
    ops=types.ModuleType('pointnet2_ops'); sub=types.ModuleType('pointnet2_ops.pointnet2_utils')
    sub.furthest_point_sample=furthest_point_sample; sub.gather_operation=gather_operation
    ops.pointnet2_utils=sub
    sys.modules['pointnet2_ops']=ops;sys.modules['pointnet2_ops.pointnet2_utils']=sub
    backend=os.environ.get('SIMPLE3D_KNN_BACKEND','cpu')
    if backend not in ('cpu','gpu'):raise ValueError('Unknown SIMPLE3D_KNN_BACKEND')
    if backend=='gpu':
        from simple3d_gpu_knn import GPUKNN
        selected=GPUKNN
    else:selected=KNN
    knn=types.ModuleType('knn_cuda');knn.KNN=selected;sys.modules['knn_cuda']=knn
    # Upstream imports a dataset module absent from its own repository; unused here.
    quan=types.ModuleType('data.quan');quan.get_quan_loader=None;quan.quan_classes=lambda:[]
    sys.modules['data.quan']=quan

def fast_interpolation(xyz1,xyz2,points2,chunk_size=8192):
    from feature_extractors.pointnet2_utils import square_distance,index_points
    a,b,p=xyz1.transpose(1,2),xyz2.transpose(1,2),points2.transpose(1,2)
    chunks=[]
    for q in a.split(chunk_size,dim=1):
        dist,idx=square_distance(q,b).topk(3,dim=-1,largest=False,sorted=True)
        w=1/(dist+1e-8);w=w/w.sum(-1,keepdim=True)
        chunks.append((index_points(p,idx)*w.unsqueeze(-1)).sum(2).transpose(1,2))
    return torch.cat(chunks,dim=2)
