import torch
from torch import nn
from models.encoder import TemporalEncoder

class SepsisPredictor(nn.Module):
    def __init__(self,features,hidden=64,direct=False,matched=False):
        super().__init__()
        self.matched=matched
        self.direct=direct
        self.encoder=TemporalEncoder(features,hidden) if direct else nn.GRU(features,hidden,batch_first=True)
        if matched: self.encoder=nn.GRU(features*3,hidden,batch_first=True)
        self.head=nn.Linear(hidden,1)
    def forward(self,x,m,d,valid):
        if self.matched:
            return self.head(self.encoder(torch.cat([x,m,torch.log1p(d)],-1))[0]).squeeze(-1)
        h=self.encoder(x,m,d,valid) if self.direct else self.encoder(x)[0]
        return self.head(h).squeeze(-1)
