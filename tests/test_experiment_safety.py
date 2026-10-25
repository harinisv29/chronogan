import pytest
import torch
from preprocess import FEATURES
from models.generator import Generator
from utils.cache import windows
from utils.experiment import generator_from_state
from utils.losses import generator_loss

def test_reconstruction_component_ignores_unscored_ground_truth():
    out=torch.zeros(1,3,2,requires_grad=True); x=torch.ones_like(out)
    held=torch.zeros_like(out); held[0,1,0]=1
    config=dict(adversarial=.1,reconstruction=10.,temporal=.01,plausibility=.001)
    args=(held,1-held,torch.ones(1,3),torch.zeros_like(out),torch.ones_like(out),config)
    _,first=generator_loss(out,x,*args,return_components=True)
    changed=x.clone(); changed[held==0]=999
    _,second=generator_loss(out,changed,*args,return_components=True)
    assert first['reconstruction_mse']==second['reconstruction_mse']

def test_ablation_flags_are_restored_with_weights():
    g=Generator(34,8,2)
    state={'stats':{'features':FEATURES},'config':{'hidden':8,'noise':2,'use_mask':False,'use_delta':False},'generator':g.state_dict()}
    restored=generator_from_state(state)
    assert not restored.use_mask and not restored.use_delta

def test_window_cache_rejects_changed_normalization(tmp_path):
    path=tmp_path/'real-parser-fixture.psv'; path.write_text('HR|ICULOS|SepsisLabel\n80|1|0\nNaN|2|0\n')
    stats={'mean':[0.]*34,'std':[1.]*34}
    cached=windows([str(path)],stats,4,tmp_path/'cache')
    assert len(cached)==1 and cached[0][3].tolist()==[1,1,0,0]
    changed={**stats,'mean':[1.]*34}
    with pytest.raises(AssertionError,match='mismatch'): windows([str(path)],changed,4,tmp_path/'cache')
