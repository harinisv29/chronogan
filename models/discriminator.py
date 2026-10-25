import torch
from torch import nn
from torch.nn import functional as F

class Discriminator(nn.Module):
    """GAIN-style per-cell observation discrimination with partial mask hints."""
    def __init__(self,features,hidden=64):
        super().__init__()
        self.conv1=nn.Conv1d(features*2+1,hidden,3)
        self.conv2=nn.Conv1d(hidden,hidden,3)
        self.head=nn.Conv1d(hidden,features,1)
    def forward(self,x,hint,valid):
        h=torch.cat([x,hint,valid[...,None]],-1).transpose(1,2)
        h=F.leaky_relu(self.conv1(F.pad(h,(2,0))),.2)
        h=F.leaky_relu(self.conv2(F.pad(h,(2,0))),.2)
        return self.head(h).transpose(1,2)
