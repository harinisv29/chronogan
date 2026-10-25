import torch
from utils.research import simple,hide
from utils.baselines import simple_impute
from models.generator import Generator

def test_vectorized_baselines_match_existing_implementations():
    x=torch.tensor([[[0.,0.],[2.,0.],[0.,0.],[8.,0.],[0.,0.]]])
    m=torch.tensor([[[0.,0.],[1.,0.],[0.,0.],[1.,0.],[0.,0.]]])
    for method in ['mean','forward','linear_retrospective']:
        torch.testing.assert_close(simple(x,m,method),simple_impute(x,m,method))

def test_structured_masks_only_hide_known_unpadded_measurements():
    m=(torch.rand(3,24,34)>.6).float(); v=torch.ones(3,24); v[:,-3:]=0
    for pattern in ['random','block','variable_specific','laboratory_panel_block']:
        rng=torch.Generator().manual_seed(42); visible,held=hide(m,v,pattern,.2,rng)
        assert torch.equal(visible+held,m) and not held[:,-3:].any()
        assert not held[m==0].any()
        if pattern=='laboratory_panel_block': assert not held[:,:,:8].any()

def test_ablations_preserve_observations():
    g=Generator(4,8,2); x=torch.randn(2,5,4); m=(torch.rand_like(x)>.5).float()
    for use_mask,use_delta in [(True,True),(False,True),(True,False)]:
        g.use_mask=use_mask; g.use_delta=use_delta
        out=g(x*m,m,torch.ones_like(x),torch.ones(2,5))
        assert torch.equal(out[m.bool()],x[m.bool()])
