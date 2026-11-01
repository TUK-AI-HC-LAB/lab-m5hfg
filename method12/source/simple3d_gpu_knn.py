"""Exact GPU coordinate kNN, bounded memory, no FAISS dependency."""
import torch

class GPUKNN:
    def __init__(self,k,transpose_mode=True,query_chunk=128,reference_chunk=524288):
        self.k=k;self.transpose_mode=transpose_mode
        self.query_chunk=query_chunk;self.reference_chunk=reference_chunk

    @torch.no_grad()
    def __call__(self,reference,query):
        ref=reference if self.transpose_mode else reference.transpose(1,2)
        qry=query if self.transpose_mode else query.transpose(1,2)
        if not ref.is_cuda or not qry.is_cuda:raise ValueError('GPUKNN requires CUDA inputs')
        if self.k>ref.shape[1]:raise ValueError('k exceeds reference point count')
        batch_d=[];batch_i=[]
        for r,q in zip(ref,qry):
            all_d=[];all_i=[]
            for qc in q.split(self.query_chunk):
                best_d=best_i=None
                for start in range(0,len(r),self.reference_chunk):
                    # Direct differences avoid cancellation in ||a||²+||b||²-2a·b.
                    d=torch.cdist(qc[None].contiguous(),r[None,start:start+self.reference_chunk].contiguous(),
                                  p=2,compute_mode='donot_use_mm_for_euclid_dist')[0]
                    take=min(self.k,d.shape[1]);v,idx=d.topk(take,largest=False,sorted=True);idx+=start
                    if best_d is not None:
                        v=torch.cat((best_d,v),1);idx=torch.cat((best_i,idx),1)
                        v,pos=v.topk(min(self.k,v.shape[1]),largest=False,sorted=True);idx=idx.gather(1,pos)
                    best_d,best_i=v,idx
                all_d.append(best_d);all_i.append(best_i)
            batch_d.append(torch.cat(all_d));batch_i.append(torch.cat(all_i))
        return torch.stack(batch_d),torch.stack(batch_i)
