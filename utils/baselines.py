import numpy as np
import pandas as pd
import torch
from torch import nn

class RecurrentImputer(nn.Module):
    """Causal denoising GRU baseline, not an implementation of BRITS."""
    def __init__(self,features,hidden=64):
        super().__init__()
        self.gru=nn.GRU(features*3,hidden,batch_first=True)
        self.head=nn.Linear(hidden,features)
    def forward(self,x,m,d,valid):
        y=self.head(self.gru(torch.cat([x*m,m,torch.log1p(d)],-1))[0])
        return m*x+(1-m)*y

def simple_impute(x,m,method):
    a=x.detach().cpu().numpy().copy(); mask=m.cpu().numpy().astype(bool)
    a[~mask]=np.nan
    for i in range(len(a)):
        frame=pd.DataFrame(a[i])
        if method=='forward': frame=frame.ffill()
        elif method=='linear_retrospective': frame=frame.interpolate(limit_direction='both')
        elif method!='mean': raise ValueError(method)
        a[i]=frame.fillna(0).to_numpy()
    return torch.tensor(a,device=x.device,dtype=x.dtype)
