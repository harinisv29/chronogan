from preprocess import PatientDataset, read_patient, delta_from_mask
import torch

def artificial_mask(m,valid,rate=.2,block=False,generator=None):
    if block:
        batch,length,features=m.shape
        span=max(1,int(length*rate))
        starts=torch.randint(max(1,length-span+1),(batch,1,features),device=m.device,generator=generator)
        hide=(torch.arange(length,device=m.device)[None,:,None]>=starts)&(torch.arange(length,device=m.device)[None,:,None]<starts+span)
    else:
        hide=torch.rand(m.shape,device=m.device,generator=generator)<rate
    held=m*hide*valid[...,None]
    return m-held,held

def recompute_delta(m,original_delta):
    # Original elapsed deltas encode hourly gaps; never retain information about hidden observations.
    out=torch.zeros_like(m)
    for t in range(1,m.shape[1]):
        # Challenge rows are hourly. Recompute solely from the *visible* mask.
        out[:,t]=1+(1-m[:,t-1])*out[:,t-1]
    return out
