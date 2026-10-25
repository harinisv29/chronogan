"""Synthetic fixtures test safety invariants, never research performance."""
import io
import numpy as np
import pandas as pd
import pytest
import torch
from models.generator import Generator
from preprocess import FEATURES,read_patient,tensors
from predict import reconstruct
from utils.clinical_validation import apply_policy,validate_upload_schema


@pytest.fixture
def checkpoint(tmp_path):
    model=Generator(34,8,3)
    with torch.no_grad():
        for parameter in model.parameters():parameter.zero_()
        model.head.bias.fill_(1.)
        model.head.bias[FEATURES.index('AST')]=-6.
        model.head.bias[FEATURES.index('TroponinI')]=-3.
    stats={'features':FEATURES,'mean':[0.]*34,'std':[1.]*34,'count':[100.]*34}
    stats['mean'][14]=100.;stats['std'][14]=20.
    stats['mean'][27]=2.;stats['std'][27]=5.
    path=tmp_path/'unit-only.pt'
    torch.save({'version':'synthetic-test-only','generator':model.state_dict(),'stats':stats,'config':{'hidden':8,'noise':3,'sequence_length':24}},path)
    return path,stats


def test_negative_predictions_are_withheld_not_replaced_with_zero(checkpoint):
    path,_=checkpoint
    patient=pd.DataFrame({'HR':[80.,np.nan,85.],'AST':[100.,np.nan,200.],'TroponinI':[2.,np.nan,3.]})
    output,mask,uncertainty,meta=reconstruct(patient,path,samples=2,device='cpu',seed=42)
    assert output.AST__raw.iloc[1]==-20. and output.TroponinI__raw.iloc[1]==-13.
    assert pd.isna(output.AST.iloc[1]) and pd.isna(output.TroponinI.iloc[1])
    assert output.AST__withheld.iloc[1] and output.AST__corrected.iloc[1] and mask.AST.iloc[1]
    assert 'outside_mathematical_domain' in output.AST__reason.iloc[1]
    assert pd.isna(uncertainty.AST__p05.iloc[1]) and uncertainty.AST__raw_p05.iloc[1]==-20.
    assert meta['output_validation']['observed_values_modified']==0
    for name in ['AST','TroponinI','HR']:
        known=patient[name].notna();np.testing.assert_array_equal(output.loc[known,name],patient.loc[known,name])


@pytest.mark.parametrize('length',[1,7,24,25,48])
def test_missing_blocks_all_missing_features_and_lengths(checkpoint,length):
    path,_=checkpoint
    patient=pd.DataFrame({'HR':np.full(length,80.),'AST':np.full(length,np.nan),'TroponinI':np.full(length,np.nan)})
    if length>1:patient.loc[1:min(4,length-1),'HR']=np.nan
    output,mask,uncertainty,meta=reconstruct(patient,path,samples=2,device='cpu',seed=9)
    assert len(output)==length and output.AST.isna().all() and output.AST__withheld.all()
    assert output.AST__reason.str.contains('no_prior_observation').all()
    for name in FEATURES:
        values=output[name].to_numpy();withheld=output[name+'__withheld'].to_numpy()
        assert (np.isfinite(values)|withheld).all()
        assert not np.isinf(values).any()
    assert np.array_equal(mask.AST,np.ones(length,dtype=bool))


def test_feature_order_and_normalization_roundtrip(checkpoint):
    path,stats=checkpoint
    patient=pd.DataFrame({'TroponinI':[2.,3.],'AST':[120.,140.],'HR':[80.,90.]})
    x,m,_=tensors(patient,stats)
    restored=x*np.array(stats['std'])+np.array(stats['mean'])
    raw=patient.reindex(columns=FEATURES).to_numpy()
    np.testing.assert_allclose(restored[m.astype(bool)],raw[m.astype(bool)],rtol=1e-6,atol=1e-6)
    first=reconstruct(patient,path,samples=2,device='cpu',seed=42)[0]
    second=reconstruct(patient[patient.columns[::-1]],path,samples=2,device='cpu',seed=42)[0]
    np.testing.assert_allclose(first[FEATURES],second[FEATURES],equal_nan=True)
    state=torch.load(path,weights_only=True);state['stats']['features']=FEATURES[::-1];torch.save(state,path)
    with pytest.raises(ValueError,match='feature names/order'):reconstruct(patient,path,device='cpu')


def test_extreme_and_invalid_observations_are_preserved(checkpoint):
    path,_=checkpoint
    patient=pd.DataFrame({'HR':[-2.,10000.],'AST':[-99.,1e7],'Temp':[98.6,37.]})
    output,_,_,meta=reconstruct(patient,path,samples=2,device='cpu')
    for name in patient:np.testing.assert_array_equal(output[name],patient[name])
    assert not output.AST__corrected.any()
    assert meta['output_validation']['per_variable']['AST']['observed_domain_warnings']==1
    assert any('shift' in message for message in meta['output_validation']['warnings'])


def test_interval_domain_violation_withholds_interval_only():
    shape=(2,34);original=np.ones(shape);original[1,0]=np.nan
    raw=np.ones(shape);raw[0]=original[0];lo=raw.copy();hi=raw.copy();lo[1,0]=-1.;hi[1,0]=2.
    stats={'mean':[0.]*34,'std':[1.]*34,'count':[10.]*34}
    point,lower,upper,_,withheld,_,interval_withheld,_,_=apply_policy(raw,lo,hi,np.zeros(shape),original,stats,True)
    assert point[1,0]==1 and not withheld[1,0]
    assert interval_withheld[1,0] and np.isnan(lower[1,0]) and np.isnan(upper[1,0])
    assert lo[1,0]==-1 and hi[1,0]==2


def test_no_future_anchor_authorizes_an_earlier_estimate(checkpoint):
    path,_=checkpoint
    patient=pd.DataFrame({'HR':[np.nan,80.,np.nan]})
    output,_,_,_=reconstruct(patient,path,samples=2,device='cpu')
    assert output.HR__withheld.iloc[0] and np.isnan(output.HR.iloc[0])
    assert output.HR.iloc[1]==80 and output.HR.iloc[2]==1


def test_csv_parsing_duplicate_headers_and_export(checkpoint,tmp_path,monkeypatch):
    path,_=checkpoint
    patient_path=tmp_path/'patient.csv';patient_path.write_text('TroponinI,HR,AST\n2,80,100\nNaN,,\n')
    patient=read_patient(patient_path);assert patient.iloc[1].isna().all()
    duplicate=tmp_path/'duplicate.csv';duplicate.write_text('HR,HR\n80,90\n')
    with pytest.raises(ValueError,match='duplicate'):validate_upload_schema(duplicate)
    import app as application
    monkeypatch.setattr(application,'active_artifacts',lambda:(path,tmp_path))
    with application.app.test_client() as client:
        assert client.post('/api/upload',data={'file':(io.BytesIO(patient_path.read_bytes()),'patient.csv')}).status_code==200
        response=client.post('/api/reconstruct');assert response.status_code==200
        assert response.json['result'][1]['TroponinI'] is None
        assert response.json['result'][1]['TroponinI__raw']==-13.
        for kind in ['csv','mask','uncertainty','metadata','raw','raw-uncertainty','pdf']:
            assert client.get('/api/export/'+kind).status_code==200
        exported=pd.read_csv(io.BytesIO(client.get('/api/export/csv').data))
        assert exported.HR.iloc[0]==80 and pd.isna(exported.AST.iloc[1]) and exported.AST__withheld.iloc[1]
        assert exported.AST__raw.iloc[1]==-20


def test_infinite_or_unrepresentable_inputs_fail_explicitly(checkpoint):
    path,_=checkpoint
    for value in [np.inf,1e308]:
        with pytest.raises(ValueError):reconstruct(pd.DataFrame({'HR':[value]}),path,device='cpu')


def test_random_missingness_flags_and_accepted_domains(checkpoint):
    path,_=checkpoint;rng=np.random.default_rng(42)
    patient=pd.DataFrame({'HR':rng.uniform(60,100,31),'O2Sat':rng.uniform(90,100,31),'AST':rng.uniform(10,100,31)})
    hidden=rng.random(patient.shape)<.3;patient=patient.mask(hidden)
    result,mask,_,_=reconstruct(patient,path,samples=3,device='cpu',seed=42)
    for name in patient:
        np.testing.assert_array_equal(mask[name],patient[name].isna())
        known=patient[name].notna();np.testing.assert_array_equal(patient.loc[known,name],result.loc[known,name])
        accepted=~known&~result[name+'__withheld']
        assert (result.loc[accepted,name]>=0).all()
        if name=='O2Sat':assert (result.loc[accepted,name]<=100).all()
