import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.cache import CachedWindows
from utils.performance import BatchedWindows,collate_cached_batch,DeferredMetrics
from utils.losses import generator_loss,prepare_physiological_bounds


def test_vectorized_physiological_loss_and_gradients_are_identical():
    torch.manual_seed(42)
    shape=(3,6,34)
    first=(torch.randn(shape)*10).requires_grad_(); second=first.detach().clone().requires_grad_()
    x=torch.randn(shape); m=(torch.rand(shape)>.4).float(); held=1-m
    valid=torch.tensor([[1,1,1,1,0,0],[1,1,1,1,1,1],[1,1,1,0,0,0.]],dtype=torch.float32)
    logits=torch.randn(shape); unknown=torch.ones(shape)
    stats={'mean':np.linspace(-2,100,34).tolist(),'std':np.linspace(.1,50,34).tolist()}
    config=dict(adversarial=.1,reconstruction=10.,temporal=.01,plausibility=.001)
    a,parts_a=generator_loss(first,x,held,m,valid,logits,unknown,config,stats,True)
    b,parts_b=generator_loss(second,x,held,m,valid,logits,unknown,config,stats,True,
                           prepared_bounds=prepare_physiological_bounds(stats,'cpu'))
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    for key in parts_a: torch.testing.assert_close(parts_a[key],parts_b[key],rtol=0,atol=0)
    a.backward(); b.backward()
    torch.testing.assert_close(first.grad,second.grad,rtol=0,atol=0)


def test_vectorized_loader_preserves_shuffle_and_values(tmp_path):
    for index,name in enumerate(['x','mask','delta','valid','labels']):
        np.save(tmp_path/f'{name}.npy',np.arange(11*3*4,dtype=np.float32).reshape(11,3,4)+index)
    standard=DataLoader(CachedWindows(tmp_path),batch_size=4,shuffle=True,generator=torch.Generator().manual_seed(42))
    batched=DataLoader(BatchedWindows(tmp_path),batch_size=4,shuffle=True,generator=torch.Generator().manual_seed(42),collate_fn=collate_cached_batch)
    for a,b in zip(standard,batched):
        for x,y in zip(a,b): torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_deferred_metrics_preserve_original_averages():
    rng=np.random.default_rng(42); values=rng.normal(size=(9,6)).astype(np.float32)
    meter=DeferredMetrics()
    for row in values:
        tensors=[torch.tensor(value) for value in row]
        meter.add(tensors[0],tensors[1],dict(zip(meter.keys[2:],tensors[2:])))
    assert meter.means()==dict(zip(meter.keys,values.astype(np.float64).mean(0).tolist()))


def test_complete_cpu_training_trajectory_is_preserved(tmp_path,monkeypatch):
    import train as trainer
    from preprocess import FEATURES
    from torch.utils.data import TensorDataset
    torch.manual_seed(7)
    x=torch.randn(7,4,34); m=(torch.rand_like(x)>.4).float()
    dataset=TensorDataset(x*m,m,torch.ones_like(x),torch.ones(7,4),torch.zeros(7,4))
    stats={'features':FEATURES,'mean':[0.]*34,'std':[1.]*34}
    splits={'train':['fixture'],'val':['fixture-val'],'test':['fixture-test']}
    monkeypatch.setattr(trainer,'prepare',lambda config:(splits,stats))
    monkeypatch.setattr(trainer,'windows',lambda *args:dataset)
    base={'seed':42,'device':'cpu','hidden':8,'noise':2,'sequence_length':4,'batch_size':3,'epochs':2,'patience':7,'learning_rate':.001,'hide_rate':.2,'adversarial':.1,'reconstruction':10.,'temporal':.01,'plausibility':.001}
    reference=trainer.train({**base,'run_dir':str(tmp_path/'reference')})
    candidate=trainer.train({**base,'run_dir':str(tmp_path/'candidate'),'performance_optimized':True})
    for a,b in zip(reference,candidate):
        for key in ['generator','discriminator','validation_normalized_mae','adversarial_loss','reconstruction_mse','temporal_loss','plausibility_loss']:
            assert a[key]==b[key]
    a=trainer.load_checkpoint(tmp_path/'reference/checkpoints/last.pt')
    b=trainer.load_checkpoint(tmp_path/'candidate/checkpoints/last.pt')
    for name in ['generator','discriminator']:
        assert all(torch.equal(a[name][key],b[name][key]) for key in a[name])
