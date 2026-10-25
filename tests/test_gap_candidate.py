import numpy as np
import torch
import json
from pathlib import Path
from models.gap_candidate import GapCandidate

def model(gap=False,domain=False):
    return GapCandidate(3,4,{'mean':[2.,0.,1.],'std':[5.,1.,2.]},[0,2],gap,domain)

def test_zero_gate_reproduces_base_and_preserves_observed_values():
    g=model(gap=True);x=torch.randn(2,5,3);m=torch.ones_like(x);m[:,1:3]=0;d=torch.ones_like(x);v=torch.ones(2,5)
    torch.testing.assert_close(g(x*m,m,d,v),g.base(x*m,m,d,v))
    out=g(x*m,m,d,v);torch.testing.assert_close(out[m.bool()],x[m.bool()])

def test_domain_link_has_positive_native_missing_values_and_finite_gradients():
    g=model(domain=True)
    with torch.no_grad():g.base.head.weight.zero_();g.base.head.bias.fill_(-5)
    x=torch.zeros(1,3,3);m=torch.zeros_like(x);v=torch.ones(1,3);out=g(x,m,torch.ones_like(x),v)
    native=out*torch.tensor([5.,1.,2.])+torch.tensor([2.,0.,1.])
    assert (native[:,:,[0,2]]>=-1e-6).all() and (out[:,:,1]<0).all()
    out.sum().backward();assert all(torch.isfinite(p.grad).all() for p in g.base.parameters() if p.grad is not None)

def test_gap_adapter_is_causal_and_does_not_carry_an_unobserved_feature():
    g=model(gap=True)
    with torch.no_grad():g.carry_weight.fill_(1)
    x=torch.zeros(1,4,3);x[:,0,0]=4;m=torch.zeros_like(x);m[:,0,0]=1;d=torch.ones_like(x);v=torch.ones(1,4)
    a=g(x,m,d,v);x[:,3]=100;m[:,3]=1;b=g(x,m,d,v)
    torch.testing.assert_close(a[:,:3],b[:,:3])
    baseline=g.base(x*m,m,d,v);torch.testing.assert_close(b[:,:3,1:],baseline[:,:3,1:])

def test_real_statistics_inverse_stays_nonnegative_even_after_link_underflow():
    stats=json.loads((Path(__file__).resolve().parents[1]/'data/full/statistics.json').read_text())
    g=GapCandidate(34,4,stats,list(range(34)),domain=True)
    with torch.no_grad():g.base.head.weight.zero_();g.base.head.bias.fill_(-1000.)
    x=torch.zeros(1,1,34);out=g(x,x,torch.ones_like(x),torch.ones(1,1)).detach().numpy().astype(float)
    native=out*np.array(stats['std'])+np.array(stats['mean'])
    assert np.isfinite(native).all() and (native>=0).all()
