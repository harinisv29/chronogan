import torch
from torch import nn

class TemporalEncoder(nn.Module):
    def __init__(self, features, hidden):
        super().__init__()
        self.embedding=nn.Parameter(torch.zeros(features))
        self.decay=nn.Parameter(torch.zeros(features))
        self.hidden_decay=nn.Linear(features,hidden)
        self.cell=nn.GRUCell(features*3,hidden)
        self.hidden=hidden
    def forward(self,x,m,d,valid,use_mask=True):
        h=x.new_zeros(x.size(0),self.hidden); last=torch.zeros_like(x[:,0]); outputs=[]
        for t in range(x.size(1)):
            gamma=torch.exp(-torch.nn.functional.softplus(self.decay)*d[:,t])
            value=m[:,t]*x[:,t]+(1-m[:,t])*(gamma*last+(1-gamma)*self.embedding)
            if not use_mask: value=x[:,t]  # Zero-filled physiology; no mask-conditioned input decay.
            candidate=self.cell(torch.cat([value,m[:,t] if use_mask else torch.zeros_like(m[:,t]),torch.log1p(d[:,t])],-1),h*torch.exp(-torch.relu(self.hidden_decay(d[:,t]))))
            active=valid[:,t,None]
            h=active*candidate+(1-active)*h
            last=active*(m[:,t]*x[:,t]+(1-m[:,t])*last)+(1-active)*last
            outputs.append(h*active)
        return torch.stack(outputs,1)
