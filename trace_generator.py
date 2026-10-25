"""Read-only generator-head trace for the exact supplied synthetic record."""
from pathlib import Path
import importlib.util
import json
import numpy as np
import torch
from train import load_checkpoint
from preprocess import FEATURES,read_patient

root=Path(__file__).resolve().parent
out=root/'runs/inference-audit/20261022'
checkpoint=root/'runs/full-training/models/chronogan_best.pt'
state=load_checkpoint(checkpoint,'cpu');torch.set_num_threads(2)
spec=importlib.util.spec_from_file_location('legacy',out/'predict-before.py');legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
head=[];original_factory=legacy.generator_from_state

def traced_factory(*args,**kwargs):
    model=original_factory(*args,**kwargs)
    model.head.register_forward_hook(lambda module,inputs,output:head.append(output.detach().cpu().numpy()))
    return model

legacy.generator_from_state=traced_factory
patient=read_patient(out/'synthetic-input.csv')
result=legacy.reconstruct(patient,checkpoint,samples=20,device='cpu',seed=42)[0]
normalized=np.stack(head).astype(np.float64).mean(0)[0]
trace={}
for name in ['TroponinI','AST']:
    j=FEATURES.index(name);native=normalized[:,j]*state['stats']['std'][j]+state['stats']['mean'][j]
    error=float(np.max(np.abs(native-result[name].to_numpy())))
    assert error<1e-9
    trace[name]={'feature_index':j,'normalized_generator_head_minimum':float(normalized[:,j].min()),'training_mean':state['stats']['mean'][j],
                 'training_std':state['stats']['std'][j],'inverse_transformed_minimum':float(native.min()),'inverse_formula_maximum_error':error,
                 'seed':42,'samples':20,'mode':'Original full-sequence replay, separate from corrected training-window inference.'}
(out/'generator-trace.json').write_text(json.dumps(trace,indent=2))
print(json.dumps(trace,indent=2))
