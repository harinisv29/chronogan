import numpy as np
from pathlib import Path
import pandas as pd
from preprocess import FEATURES,read_patient
from utils.clinical_validation import domain_invalid,registry
from reconstruction_experiment import masking,baselines,metrics,aggregate

def test_masks_only_known_values_and_feature_mask_is_whole_feature():
    known=np.ones((36,34),dtype=bool);known[0]=False
    for pattern in ['random','block','feature']:
        held=masking(known,pattern,.4,np.random.default_rng(7))
        assert not (held&~known).any()
        np.testing.assert_array_equal(held,masking(known,pattern,.4,np.random.default_rng(7)))
        if pattern=='feature':assert (held[1:]==held[1]).all()

def test_forward_is_causal_and_resets_at_training_window():
    values=np.full((25,34),np.nan);values[0]=10;values[23]=20
    output=baselines(values,np.ones(34),24)
    np.testing.assert_array_equal(output['forward'][22],np.full(34,10))
    np.testing.assert_array_equal(output['forward'][24],np.ones(34))
    np.testing.assert_array_equal(output['forward'][23],values[23])

def test_aggregate_separates_withheld_ground_truth_error():
    truth=np.ones((2,34));pred=truth.copy();pred[0,0]=3;pred[1,0]=5
    held=np.zeros_like(truth,dtype=bool);held[:,0]=True
    released=np.ones_like(held);released[1,0]=False
    row=metrics(pred,truth,held,np.ones(34),np.ones(34),released)
    result=aggregate([row],np.ones(34),np.ones(34))
    assert result['raw']['standardized_mae']==3
    assert result['released']['standardized_mae']==2
    assert result['withheld']['standardized_mae']==4
    assert result['coverage']==.5

def test_supplied_patient_and_historical_export_regression():
    folder=Path(__file__).resolve().parents[1]/'runs/inference-audit/20261022'
    patient=read_patient(folder/'synthetic-input.csv').reindex(columns=FEATURES)
    historic=pd.read_csv(folder/'downloaded-before.csv')
    assert patient.shape==(36,34) and patient.isna().sum().sum()==934
    known=patient.notna()
    for name in FEATURES:
        np.testing.assert_allclose(patient.loc[known[name],name],historic.loc[known[name],name],rtol=0,atol=1e-10)
    specs=registry()['variables']
    invalid=sum((domain_invalid(historic[f].to_numpy(),specs[f])&~known[f]).sum() for f in FEATURES)
    assert invalid==74
