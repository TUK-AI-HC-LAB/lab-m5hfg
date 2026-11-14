// STEP-inspired exact 3D search. Wang and Ino, doi:10.1016/j.jpdc.2025.105187.
// Sort bounding-sphere lower bounds, traverse, prune, merge candidates.
// Independent implementation; not the paper's warp/kernel implementation.
__device__ bool after(double a,int i,double b,int j) {
    return a>b || (a==b && i>j);
}
__device__ void sort_pairs(double* d,int* ids,int n) {
    for (int size=2;size<=n;size*=2) {
        for (int stride=size/2;stride;stride/=2) {
            for (int i=threadIdx.x;i<n;i+=blockDim.x) {
                int j=i^stride;
                if (j>i && (after(d[i],ids[i],d[j],ids[j])==((i&size)==0))) {
                    double u=d[i]; d[i]=d[j]; d[j]=u;
                    int v=ids[i]; ids[i]=ids[j]; ids[j]=v;
                }
            }
            __syncthreads();
        }
    }
}
extern "C" __global__ void search(
    const float* x,const float* centers,const double* rad,const int* offsets,
    const int* members,int p,int n,int* result) {
    int q=blockIdx.x, t=threadIdx.x;
    if(q>=n) return;
    double X=x[3*q],Y=x[3*q+1],Z=x[3*q+2];
    __shared__ double bound[512], best[256];
    __shared__ int order[512], ids[256];
    __shared__ int needed;
    for(int c=t;c<p;c+=128){
        double a=X-centers[c*3],b=Y-centers[c*3+1],d=Z-centers[c*3+2];
        double r=sqrt(a*a+b*b+d*d)-rad[c];
        // Conservative slack for sqrt/subtraction and the radius reduction.
        r=fmax(0.,r-1e-12*(1.+fabs(r)+rad[c]+fabs(X)+fabs(Y)+fabs(Z)));
        bound[c]=r*r; order[c]=c;
    }
    best[t]=1.0/0.0; ids[t]=2147483647;
    __syncthreads();
    sort_pairs(bound,order,p);
    for(int c=0;c<p;++c) {
        if(bound[c]>best[119]) break; // strict, to include distance ties
        int g=order[c];
        for (int start=offsets[g];start<offsets[g+1];start+=128) {
            int at=start+t, id=2147483647;
            double d=1.0/0.0;
            if (at<offsets[g+1]) {
                id=members[at];
                double a=X-x[id*3],b=Y-x[id*3+1],v=Z-x[id*3+2];
                d=a*a+b*b+v*v;
            }
            if(t==0) needed=0;
            __syncthreads();
            if(after(best[119],ids[119],d,id)) atomicOr(&needed,1);
            best[128+t]=d; ids[128+t]=id;
            __syncthreads();
            if(needed) sort_pairs(best,ids,256);
            __syncthreads(); // keep needed and best[119] stable for all threads
        }
    }
    if(t<120) result[(long long)q*120+t]=ids[t];
}
