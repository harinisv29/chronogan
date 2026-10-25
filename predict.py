import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import hashlib
import os
from datetime import datetime,timezone
from download_dataset import ROOT
from preprocess import FEATURES,read_patient,tensors
from models.generator import Generator
from train import load_checkpoint
from utils.experiment import generator_from_state
from utils.baselines import RecurrentImputer
from utils.dataset import recompute_delta
from utils.research import simple
from utils.clinical_validation import validate_stats,apply_policy,registry

def reconstruct(frame,checkpoint=None,samples=20,device=None,seed=None,output_policy=None,sequence_mode='training'):
    if not 2<=samples<=100: raise ValueError('Use 2 to 100 noise samples')
    checkpoint=Path(checkpoint) if checkpoint is not None else ROOT/'models'/'chronogan_best.pt'
    if not checkpoint.exists(): raise FileNotFoundError('Train ChronoGAN first; no checkpoint is available.')
    device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA inference requested but unavailable')
    state=load_checkpoint(checkpoint,device); cfg=state['config']; stats=state['stats']
    validate_stats(stats)
    if not len(frame):raise ValueError('Patient record is empty')
    raw_input=frame.reindex(columns=FEATURES).to_numpy(dtype=float)
    if np.isinf(raw_input).any():raise ValueError('Infinite observed measurements are not missing-value markers')
    with np.errstate(over='ignore',invalid='ignore'):
        cast_input=raw_input.astype(np.float32)
    if (np.isfinite(raw_input)&~np.isfinite(cast_input)).any():raise ValueError('An observed measurement cannot be represented safely in model float32 inputs; verify values and units')
    kind=state.get('model_kind','generator'); model=None
    if kind=='generator': model=generator_from_state(state,device).eval()
    elif kind=='recurrent':
        model=RecurrentImputer(len(FEATURES),cfg['hidden']).to(device).eval(); model.load_state_dict(state['weights'])
    elif kind not in ['mean','forward']: raise ValueError('Unsupported reconstruction model kind')
    x,m,d=tensors(frame,stats)
    if not np.isfinite(x).all():raise ValueError('Normalization produced nonfinite model inputs; verify units and magnitudes')
    inputs=[torch.tensor(value,device=device)[None] for value in (x,m,d)]
    rng=torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    if sequence_mode not in ['training','legacy_full_sequence']:raise ValueError('Unknown inference sequence mode')
    length=cfg['sequence_length'] if sequence_mode=='training' else len(x)
    chunks=[]
    with torch.no_grad():
        for start in range(0,len(x),length):
            xx,mm,dd=[value[:,start:start+length] for value in inputs]
            valid=torch.ones(1,xx.shape[1],device=device)
            if sequence_mode=='training': dd=recompute_delta(mm,dd)
            if kind=='generator':
                generated=[]
                for _ in range(samples):
                    z=torch.randn(*xx.shape[:2],model.noise,device=device,generator=rng) if rng is not None else None
                    generated.append(model(xx,mm,dd,valid,z)[0].cpu().numpy())
                chunks.append(np.stack(generated))
            else:
                output=model(xx*mm,mm,dd,valid) if kind=='recurrent' else simple(xx,mm,kind)
                chunks.append(output[0].cpu().numpy()[None])
        draws=np.concatenate(chunks,axis=1)
    draws=draws*np.array(stats['std'])+np.array(stats['mean'])
    original=frame.reindex(columns=FEATURES).to_numpy(dtype=float)
    observed=np.isfinite(original)
    mean=np.where(observed,original,draws.mean(0))
    lo=np.where(observed,original,np.quantile(draws,.05,axis=0))
    hi=np.where(observed,original,np.quantile(draws,.95,axis=0))
    spread=np.where(observed,0,draws.std(0))
    stochastic=kind=='generator'
    if not stochastic:
        lo=np.where(observed,original,np.nan); hi=lo.copy(); spread=lo.copy()
    raw_mean=mean.copy();raw_lo=lo.copy();raw_hi=hi.copy();raw_spread=spread.copy()
    mode=output_policy or os.environ.get('CHRONOGAN_OUTPUT_POLICY','withhold')
    mean,lo,hi,spread,withheld,corrected,interval_withheld,reasons,validation=apply_policy(raw_mean,raw_lo,raw_hi,raw_spread,original,stats,stochastic,mode,window_length=length)
    result=frame.copy()
    extras={}
    for j,f in enumerate(FEATURES):
        result[f]=mean[:,j]
        extras[f+'__imputed']=~observed[:,j]
        extras[f+'__raw']=np.where(np.isfinite(raw_mean[:,j]),raw_mean[:,j],np.nan)
        extras[f+'__withheld']=withheld[:,j]
        extras[f+'__corrected']=corrected[:,j]
        extras[f+'__reason']=reasons[:,j]
    result=pd.concat([result,pd.DataFrame(extras,index=frame.index)],axis=1)
    result['model_version']=state['version']
    uncertainty_values={f+'__'+key:arr[:,j] for j,f in enumerate(FEATURES) for key,arr in [('p05',lo),('p95',hi),('std',spread)]}
    for j,f in enumerate(FEATURES):
        for key,values in [('raw_p05',raw_lo),('raw_p95',raw_hi),('raw_std',raw_spread)]:
            uncertainty_values[f+'__'+key]=np.where(np.isfinite(values[:,j]),values[:,j],np.nan)
        uncertainty_values[f+'__interval_withheld']=interval_withheld[:,j]
    uncertainty=pd.DataFrame(uncertainty_values)
    metadata={'model_version':state['version'],'model_kind':kind,'deployment_variant':state.get('deployment_variant',kind),
              'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'processed_utc':datetime.now(timezone.utc).isoformat(),
              'samples':samples if stochastic else 1,'seed':seed,'uncertainty_available':stochastic,'inference_device':str(device),
              'interval':'5th–95th noise-sample quantiles; not clinically validated confidence intervals. Invalid or unsupported intervals are withheld; raw intervals remain in diagnostic exports.' if stochastic else 'Unavailable: deterministic baseline; missing-value interval columns are empty, not zero-width confidence intervals.',
              'mode':'causal training-window reset inference' if sequence_mode=='training' else 'legacy full-sequence diagnostic replay',
              'training_window_hours':cfg['sequence_length'],'features_unobserved_in_training':[f for f,n in zip(FEATURES,stats.get('count',[1]*len(FEATURES))) if n==0],
              'output_validation':validation,'units':{name:spec['unit'] for name,spec in registry()['variables'].items()},
              'raw_prediction_columns':'<variable>__raw; raw_p05/raw_p95/raw_std in uncertainty export',
              'warning':'Research prototype; no diagnostic capability or demonstrated clinical effectiveness. Estimates are not verified clinical observations. Blank withheld outputs are intentional abstentions, not reconstructed zeros.'}
    return result,pd.DataFrame(~observed,columns=FEATURES),uncertainty,metadata

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('patient'); p.add_argument('--samples',type=int,default=20)
    p.add_argument('--checkpoint',default=str(ROOT/'models'/'chronogan_best.pt'))
    p.add_argument('--output-dir',default=str(ROOT/'results')); args=p.parse_args()
    frame=read_patient(args.patient); result,mask,uncertainty,meta=reconstruct(frame,checkpoint=args.checkpoint,samples=args.samples)
    destination=Path(args.output_dir); destination.mkdir(parents=True,exist_ok=True)
    result.to_csv(destination/'reconstructed.csv',index=False); mask.to_csv(destination/'mask.csv',index=False)
    uncertainty.to_csv(destination/'uncertainty.csv',index=False)
    (destination/'inference.json').write_text(json.dumps(meta,indent=2))
