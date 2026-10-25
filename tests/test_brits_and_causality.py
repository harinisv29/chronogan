import torch
from models.brits import BRITS
from models.generator import Generator
from models.sepsis_predictor import SepsisPredictor
from utils.dataset import recompute_delta


def test_brits_padding_and_backward():
    torch.manual_seed(42)
    model=BRITS(4,8)
    x=torch.randn(2,6,4); mask=(torch.rand_like(x)>.4).float()
    valid=torch.tensor([[1,1,1,0,0,0],[1,1,1,1,1,1.]],dtype=torch.float32)
    mask=mask*valid[...,None]; d=recompute_delta(mask,mask)
    output,loss=model(x,mask,d,valid,return_loss=True)
    assert torch.isfinite(output).all() and torch.isfinite(loss)
    assert torch.equal(output[mask.bool()],x[mask.bool()])
    short=model(x[:1,:3],mask[:1,:3],d[:1,:3],valid[:1,:3])
    torch.testing.assert_close(short,output[:1,:3])
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_generator_and_matched_prediction_use_no_future_values():
    torch.manual_seed(42)
    x=torch.randn(2,8,4); m=(torch.rand_like(x)>.3).float()
    d=recompute_delta(m,m); valid=torch.ones(2,8)
    generator=Generator(4,8,3).eval(); z=torch.randn(2,8,3)
    changed=x.clone(); changed[:,4:]+=100
    a=generator(x,m,d,valid,z); b=generator(changed,m,d,valid,z)
    torch.testing.assert_close(a[:,:4],b[:,:4])
    classifier=SepsisPredictor(4,8,matched=True).eval()
    pa=classifier(a,m,d,valid); pb=classifier(b,m,d,valid)
    torch.testing.assert_close(pa[:,:4],pb[:,:4])


def test_prediction_methods_share_identical_architecture():
    models=[SepsisPredictor(4,8,matched=True) for _ in range(4)]
    signatures=[[(k,tuple(v.shape)) for k,v in model.state_dict().items()] for model in models]
    assert all(signature==signatures[0] for signature in signatures)
