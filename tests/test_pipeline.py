"""Small fixture values test invariants only; never used as research results."""
import io
import numpy as np
import pandas as pd
import pytest
import torch
from preprocess import FEATURES,read_patient,tensors,delta_from_mask,make_splits,fit_stats,PatientDataset
from models.generator import Generator
from models.discriminator import Discriminator
from utils.dataset import artificial_mask,recompute_delta
from utils.baselines import simple_impute
from train import load_checkpoint

@pytest.fixture
def stats(): return dict(mean=[0.]*len(FEATURES),std=[1.]*len(FEATURES),features=FEATURES)

def test_parsing_and_masks(tmp_path,stats):
    path=tmp_path/'p.psv'; path.write_text('HR|Temp|ICULOS|SepsisLabel\n80|NaN|1|0\nNaN|37|2|1\n')
    f=read_patient(path); x,m,d=tensors(f,stats)
    assert pd.isna(f.HR.iloc[1]); assert m[0,0]==1 and m[1,0]==0
    assert x.shape==(2,len(FEATURES)); assert 'SepsisLabel' not in FEATURES
    assert np.array_equal(delta_from_mask(np.array([[0],[0],[1],[0]])),[[0],[1],[2],[1]])

def test_generator_preservation_shapes_and_causality():
    torch.manual_seed(9); model=Generator(4,12,3).eval()
    x=torch.randn(2,7,4); m=(torch.rand_like(x)>.5).float(); d=torch.ones_like(x); valid=torch.ones(2,7); z=torch.randn(2,7,3)
    a=model(x,m,d,valid,z)
    assert a.shape==x.shape; assert torch.equal(a[m.bool()],x[m.bool()])
    changed=x.clone(); changed[:,4:]+=100
    changed_m=m.clone(); changed_m[:,4:]=1-changed_m[:,4:]
    changed_d=d.clone(); changed_d[:,4:]=99
    b=model(changed,changed_m,changed_d,valid,z)
    assert torch.equal(a[:,:4],b[:,:4])
    logits=Discriminator(4,12)(a,m,valid)
    assert logits.shape==x.shape

def test_hidden_mask_and_padding():
    m=torch.tensor([[[1.,0.],[0.,1.],[0.,0.]]]); v=torch.tensor([[1.,1.,0.]])
    visible,held=artificial_mask(m,v,1.)
    assert torch.equal(held,m); assert visible.sum()==0
    assert torch.equal(recompute_delta(visible,torch.zeros_like(m)),recompute_delta(visible,torch.ones_like(m)*99))

def test_patient_splits_training_statistics_and_padding(tmp_path):
    paths=[]
    for i in range(10):
        p=tmp_path/f'{i}.psv'; p.write_text(f'HR|ICULOS\n{i}|1\nNaN|2\n'); paths.append(p)
    s=make_splits(paths); assert not set(s['train'])&set(s['test']); assert not set(s['val'])&set(s['test'])
    stats=fit_stats(s['train']); expected=np.mean([int(pd.read_csv(p,sep='|').HR.iloc[0]) for p in s['train']])
    assert stats['mean'][0]==expected
    batch=PatientDataset(s['test'],stats,4)[0]
    assert batch[0].shape==(4,len(FEATURES)); assert batch[3].tolist()==[1,1,0,0]
    assert np.isfinite(batch[0].numpy()).all()

def test_checkpoint_and_unseen_inference(tmp_path,stats):
    from predict import reconstruct
    cfg=dict(hidden=8,noise=3,sequence_length=24)
    model=Generator(len(FEATURES),8,3)
    path=tmp_path/'unit-test-only.pt'
    torch.save(dict(config=cfg,stats=stats,generator=model.state_dict(),version='unit-test-untrained'),path)
    assert load_checkpoint(path)['version']=='unit-test-untrained'
    f=pd.DataFrame({'HR':[80.,np.nan,90.],'Temp':[np.nan,37.,np.nan]})
    result,mask,uncertainty,meta=reconstruct(f,path,samples=2)
    assert result.HR.iloc[0]==80.; assert result.HR.iloc[2]==90.
    assert mask.HR.tolist()==[False,True,False]; assert len(uncertainty)==3
    assert pd.read_csv(io.StringIO(result.to_csv(index=False))).HR.iloc[0]==80.

def test_forward_baseline_no_future():
    x=torch.tensor([[[2.],[0.],[100.]]]); m=torch.tensor([[[1.],[0.],[1.]]])
    assert simple_impute(x,m,'forward')[0,1,0]==2

def test_api_validation_and_export():
    from app import app
    app.config.update(TESTING=True)
    with app.test_client() as client:
        assert client.post('/api/upload').status_code==400
        assert client.post('/api/reconstruct').status_code==400
        assert client.get('/api/export/csv').status_code==409
        assert client.post('/api/upload',data={'file':(io.BytesIO(b'HR\nwrong\n'),'p.csv')}).status_code==400
        assert client.post('/api/upload',data={'file':(io.BytesIO(b'HR|ICULOS\n80|2\n90|1\n'),'p.psv')}).status_code==400
        r=client.post('/api/upload',data={'file':(io.BytesIO(b'HR,Temp\n80,\n,37\n'),'p.csv')})
        assert r.status_code==200; assert r.json['rows']==2
        assert client.get('/').status_code==200

def test_real_held_out_patient():
    from download_dataset import ROOT
    from predict import reconstruct
    checkpoint=ROOT/'models'/'chronogan_best.pt'
    if not checkpoint.exists(): pytest.skip('Real-data training has not completed')
    state=load_checkpoint(checkpoint); path=state['splits']['test'][0]
    f=read_patient(path); result,mask,_,_=reconstruct(f,samples=2)
    for col in FEATURES:
        if col in f:
            known=f[col].notna(); assert np.array_equal(f.loc[known,col],result.loc[known,col])
    values=result[FEATURES].to_numpy();withheld=result[[f+'__withheld' for f in FEATURES]].to_numpy()
    assert (np.isfinite(values)|withheld).all() and not np.isinf(values).any()

def test_real_patient_web_workflow():
    from app import app
    from download_dataset import ROOT
    checkpoint=ROOT/'models'/'chronogan_best.pt'
    if not checkpoint.exists(): pytest.skip('Real-data training has not completed')
    state=load_checkpoint(checkpoint); path=state['splits']['test'][0]
    app.config.update(TESTING=True)
    with app.test_client() as client:
        with open(path,'rb') as source:
            r=client.post('/api/upload',data={'file':(source,'held-out.psv')})
        assert r.status_code==200
        r=client.post('/api/reconstruct'); assert r.status_code==200
        from app import active_artifacts
        assert r.json['metadata']['model_version']==load_checkpoint(active_artifacts()[0])['version']
        for kind in ['csv','mask','uncertainty','metadata','pdf']:
            response=client.get('/api/export/'+kind)
            assert response.status_code==200 and len(response.data)>0
        result=pd.read_csv(io.BytesIO(client.get('/api/export/csv').data))
        original=read_patient(path)
        for feature in FEATURES:
            known=original[feature].notna()
            np.testing.assert_allclose(result.loc[known,feature],original.loc[known,feature],rtol=0,atol=1e-12)
