"""Reconstruct one unseen real patient with the separately saved GPU checkpoint."""
import json
import numpy as np
from download_dataset import ROOT
from preprocess import FEATURES,read_patient
from predict import reconstruct
from train import load_checkpoint

checkpoint=ROOT/'runs'/'gpu-smoke'/'models'/'chronogan_best.pt'
state=load_checkpoint(checkpoint)
path=state['splits']['test'][0]
assert path not in state['splits']['train']
frame=read_patient(path)
result,mask,uncertainty,metadata=reconstruct(frame,checkpoint=checkpoint,samples=20)
for feature in FEATURES:
    observed=frame[feature].notna()
    assert np.array_equal(frame.loc[observed,feature],result.loc[observed,feature])
assert np.isfinite(result[FEATURES].to_numpy()).all()
output=ROOT/'runs'/'gpu-smoke'/'inference'; output.mkdir(exist_ok=True)
result.to_csv(output/'reconstructed.csv',index=False)
mask.to_csv(output/'mask.csv',index=False)
uncertainty.to_csv(output/'uncertainty.csv',index=False)
metadata['held_out_patient']=path
metadata['observed_measurements_preserved']=True
(output/'metadata.json').write_text(json.dumps(metadata,indent=2))
print(f'GPU checkpoint inference passed on {len(frame)} unseen real hourly rows; {int(mask.sum().sum())} estimates exported.')
