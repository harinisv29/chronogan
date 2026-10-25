import torch
from torch import nn
from models.encoder import TemporalEncoder

class Generator(nn.Module):
    def __init__(self,features,hidden=64,noise=16):
        super().__init__()
        self.noise=noise
        self.encoder=TemporalEncoder(features,hidden)
        self.gru=nn.GRU(hidden+noise,hidden,batch_first=True)
        self.head=nn.Linear(hidden,features)
    def forward(self,x,m,d,valid,z=None):
        if z is None: z=torch.randn(*x.shape[:2],self.noise,device=x.device)
        h=self.encoder(x,m,d if getattr(self,'use_delta',True) else torch.zeros_like(d),valid,use_mask=getattr(self,'use_mask',True))
        generated=self.head(self.gru(torch.cat([h,z],-1))[0])
        return m*x+(1-m)*generated
