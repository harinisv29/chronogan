"""Experimental adapters only; not imported by production inference or training."""
import torch
from torch import nn
from torch.nn import functional as F
from utils.baselines import RecurrentImputer

class GapCandidate(nn.Module):
    def __init__(self,features,hidden,stats,nonnegative_indices,gap=False,domain=False):
        super().__init__()
        self.base=RecurrentImputer(features,hidden)
        self.gap=gap;self.domain=domain
        self.carry_weight=nn.Parameter(torch.zeros(features))
        self.decay_rate=nn.Parameter(torch.zeros(features))
        exact_lower=torch.tensor([-float(mu)/float(sd) for mu,sd in zip(stats['mean'],stats['std'])],dtype=torch.float32)
        # One representable step above the rounded lower bound prevents tiny
        # negative native values after the float64 affine inverse.
        self.register_buffer('lower',torch.nextafter(exact_lower,torch.full_like(exact_lower,float('inf'))))
        self.register_buffer('nonnegative',torch.tensor([j in nonnegative_indices for j in range(features)],dtype=torch.bool))

    def forward(self,x,m,d,valid):
        out=self.base(x,m,d,valid)
        if self.gap:
            last=torch.zeros_like(x[:,0]);seen=torch.zeros_like(last);values=[]
            for t in range(x.shape[1]):
                # convex gate, feature-specific rate, log-gap power-law decay;
                # no future observations or carry across independent windows.
                rate=torch.exp(-F.softplus(self.decay_rate)*torch.log1p(d[:,t]))
                alpha=self.carry_weight.clamp(0,1)*rate*seen
                value=(1-alpha)*out[:,t]+alpha*last
                values.append(m[:,t]*x[:,t]+(1-m[:,t])*value)
                active=valid[:,t,None]
                last=active*(m[:,t]*x[:,t]+(1-m[:,t])*last)+(1-active)*last
                seen=torch.maximum(seen,m[:,t]*active)
            out=torch.stack(values,1)
        self.prelink=out.detach()
        if self.domain:
            # Smooth lower-bound link in normalized space, trained end-to-end.
            # No clinical upper bounds or arbitrary normal ranges are imposed.
            linked=self.lower+F.softplus(out-self.lower,beta=20)
            out=torch.where(self.nonnegative,linked,out)
        return m*x+(1-m)*out
