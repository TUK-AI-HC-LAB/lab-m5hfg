// FPFH formulas adapted from Open3D 0.20.0 FeatureImpl.h (MIT).
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// External GPU KD-tree neighbor lists; full-point Float32 SPFH/FPFH.
// NVRTC provides CUDA math builtins without host libc headers.
__device__ float dot3(const float* a,const float* b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
__device__ void cross3(const float* a,const float* b,float* c){
 c[0]=a[1]*b[2]-a[2]*b[1];c[1]=a[2]*b[0]-a[0]*b[2];c[2]=a[0]*b[1]-a[1]*b[0];}
__device__ void pair_feature(const float* p1,const float* n1,const float* p2,const float* n2,float* f){
 float dp[3]={p2[0]-p1[0],p2[1]-p1[1],p2[2]-p1[2]};
 f[3]=sqrtf(dot3(dp,dp));
 if(f[3]==0){f[0]=f[1]=f[2]=f[3]=0;return;}
 float a1=dot3(n1,dp)/f[3],a2=dot3(n2,dp)/f[3];
 const float *u=n1,*nn=n2;
 if(acosf(fabsf(a1))>acosf(fabsf(a2))){u=n2;nn=n1;dp[0]=-dp[0];dp[1]=-dp[1];dp[2]=-dp[2];f[2]=-a2;}
 else f[2]=a1;
 float v[3],w[3];cross3(dp,u,v);float norm=sqrtf(dot3(v,v));
 if(norm==0){f[0]=f[1]=f[2]=f[3]=0;return;}
 v[0]/=norm;v[1]/=norm;v[2]/=norm;cross3(u,v,w);
 f[1]=dot3(v,nn);f[0]=atan2f(dot3(w,nn),dot3(u,nn));
}
extern "C" __global__ void spfh(const float* xyz,const float* normals,const int* indices,
 int n,int k,int stride,float* hist){
 int p=blockDim.x*blockIdx.x+threadIdx.x;if(p>=n)return;
 float h[33]={0};float incr=100.0/(float)(k-1);
 for(int j=1;j<k;++j){int q=indices[(long long)p*stride+j];float f[4]={0};
  pair_feature(xyz+3*p,normals+3*p,xyz+3*q,normals+3*q,f);
  int a=(int)floor(11*(f[0]+3.14159265358979323846)/(2.0*3.14159265358979323846));
  int b=(int)floor(11*(f[1]+1.0)*0.5),c=(int)floor(11*(f[2]+1.0)*0.5);
  a=a>=11?10:max(0,a);b=b>=11?10:max(0,b);c=c>=11?10:max(0,c);
  h[a]+=incr;h[b+11]+=incr;h[c+22]+=incr;
 }
 for(int b=0;b<33;++b)hist[(long long)p*33+b]=h[b];
}
extern "C" __global__ void fpfh(const float* xyz,const int* indices,const float* s,
 int n,int k,int stride,float* out){
 int p=blockDim.x*blockIdx.x+threadIdx.x;if(p>=n)return;
 float h[33]={0},sum[3]={0};
 for(int j=1;j<k;++j){int q=indices[(long long)p*stride+j];
  float dx=xyz[3*p]-xyz[3*q],dy=xyz[3*p+1]-xyz[3*q+1],dz=xyz[3*p+2]-xyz[3*q+2];
  float d=dx*dx+dy*dy+dz*dz;if(d==0)continue;
  for(int b=0;b<33;++b){float val=s[(long long)q*33+b]/d;sum[b/11]+=val;h[b]+=val;}
 }
 for(int b=0;b<3;++b)sum[b]=sum[b]!=0?100.0/sum[b]:0;
 for(int b=0;b<33;++b)out[(long long)p*33+b]=h[b]*sum[b/11]+s[(long long)p*33+b];
}
