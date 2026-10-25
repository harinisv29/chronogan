import numpy as np
import pandas as pd
import torch
from preprocess import FEATURES
from predict import reconstruct
from utils.research_diagnostics import temporal_errors,stochastic_variation
from models.generator import Generator


def test_deterministic_baseline_preserves_observations_without_invented_intervals(tmp_path):
    path=tmp_path/'baseline.pt'
    torch.save({'version':'test-baseline','model_kind':'forward','stats':{'mean':[0.]*34,'std':[1.]*34,'features':FEATURES},'config':{'sequence_length':24,'deployment_windowed':True}},path)
    patient=pd.DataFrame({'HR':[80.,np.nan,90.]})
    result,mask,uncertainty,meta=reconstruct(patient,path,device='cpu',seed=42)
    assert result.HR.tolist()==[80.,80.,90.]
    assert not meta['uncertainty_available'] and meta['samples']==1
    assert np.isnan(uncertainty.HR__p05.iloc[1]) and uncertainty.HR__p05.iloc[0]==80
    assert mask.HR.tolist()==[False,True,False]


def test_stochastic_variant_flags_and_seed_are_respected(tmp_path):
    model=Generator(34,8,3)
    path=tmp_path/'variant.pt'
    torch.save({'version':'test-variant','model_kind':'generator','generator':model.state_dict(),
                'stats':{'mean':[0.]*34,'std':[1.]*34,'features':FEATURES},
                'config':{'hidden':8,'noise':3,'sequence_length':24,'use_mask':False,'use_delta':False,'deployment_windowed':True}},path)
    patient=pd.DataFrame({'HR':[80.,np.nan,90.]})
    first=reconstruct(patient,path,samples=3,device='cpu',seed=42)
    second=reconstruct(patient,path,samples=3,device='cpu',seed=42)
    np.testing.assert_array_equal(first[0][FEATURES],second[0][FEATURES])
    assert first[3]['uncertainty_available']


def test_temporal_diagnostic_excludes_unknown_truth_and_padding():
    truth=torch.tensor([[[1.],[3.],[9.],[30.]]]); output=truth+torch.tensor([[[0.],[1.],[5.],[20.]]])
    observed=torch.tensor([[[1.],[1.],[0.],[1.]]]); held=torch.tensor([[[0.],[1.],[0.],[1.]]]); valid=torch.tensor([[1.,1.,1.,0.]])
    error,count=temporal_errors(output,truth,observed,held,valid)
    assert error.item()==1 and count.item()==1


def test_stochastic_variation_detects_collapse():
    samples=torch.zeros(2,20,3,4); held=torch.ones(2,3,4)
    deviation,collapsed=stochastic_variation(samples,held)
    assert deviation.sum()==0 and torch.equal(collapsed,held.sum((0,1)))
