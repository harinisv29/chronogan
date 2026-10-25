"""Validate an existing zero-adversarial-weight checkpoint, then audit on fixed masks.

This isolated diagnostic never changes the research controller's deployment choice.
"""
import argparse
import json
import time
import numpy as np
import torch
from preprocess import FEATURES,read_patient,tensors
from train import load_checkpoint
from utils.dataset import recompute_delta
from utils.experiment import generator_from_state
from utils.clinical_validation import apply_policy
from reconstruction_experiment import ROOT,write,digest,metrics,aggregate

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['cuda'],required=True);parser.add_argument('--reference',default='runs/reconstruction-improvement/cuda-v1');args=parser.parse_args()
    assert torch.cuda.is_available();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.12);torch.cuda.reset_peak_memory_stats()
    folder=ROOT/args.reference;out=folder/'archived-no-adversarial';out.mkdir(exist_ok=True)
    assert not (out/'results.json').exists(),'Completed results must not be overwritten'
    reference=json.loads((folder/'results.json').read_text());protocol=reference['protocol'];masks=np.load(folder/'masks.npz');clock=time.perf_counter()
    path=ROOT/'runs/full-training/ablations/no_adversarial/models/chronogan_best.pt';state=load_checkpoint(path,'cpu');stats=state['stats'];length=state['config']['sequence_length'];samples=protocol['samples']
    assert stats==json.loads((ROOT/'data/full/statistics.json').read_text())
    assert state['splits']['test'][:len(protocol['test'])]==protocol['test']
    candidate_hash=digest(path);model=generator_from_state(state,'cuda').eval();assert next(model.parameters()).is_cuda
    training=json.loads((folder/'training-only-statistics.json').read_text());std=np.array(stats['std']);scale=np.array(training['iqr_scale'])
    write(out/'protocol.json',{'reference_protocol_sha256':digest(folder/'protocol.json'),'reference_masks_sha256':digest(folder/'masks.npz'),'candidate_sha256':candidate_hash,
        'rationale':'Test archived single-factor zero-adversarial-weight ablation. Architecture unchanged. No training or test-based selection.',
        'selection':'Prefer candidate only if validation random20 standardized MAE is lower than the reference. Evaluate its test outcomes for diagnosis only; frozen deployment selection is untouched.'})
    def prediction(path,held,seed):
        frame=read_patient(path);truth=frame.reindex(columns=FEATURES).to_numpy(dtype=float);values=truth.copy();values[held]=np.nan;frame[FEATURES]=values
        x,m,d=tensors(frame,stats);chunks=[];rng=torch.Generator(device='cuda').manual_seed(seed)
        with torch.no_grad():
            for start in range(0,len(x),length):
                xx,mm,dd=[torch.tensor(a[start:start+length],device='cuda')[None] for a in [x,m,d]];dd=recompute_delta(mm,dd);v=torch.ones(samples,xx.shape[1],device='cuda')
                z=torch.randn(samples,xx.shape[1],model.noise,device='cuda',generator=rng)
                chunks.append(model(xx.repeat(samples,1,1),mm.repeat(samples,1,1),dd.repeat(samples,1,1),v,z).mean(0).cpu().numpy())
        pred=np.concatenate(chunks)*std+stats['mean'];pred=np.where(np.isfinite(values),values,pred)
        return pred,values,truth
    error=0.;count=0
    for i,path in enumerate(protocol['val']):
        held=masks['val_'+str(i)];pred,values,truth=prediction(path,held,31022+i);error+=float((np.abs(pred-truth)/std)[held].sum());count+=int(held.sum())
    baseline=reference['selection']['validation_standardized_mae']['1.0'];candidate=error/count
    selection={'baseline_validation_mae':baseline,'candidate_validation_mae':candidate,'candidate_preferred_on_validation':candidate<baseline,'test_used':False};write(out/'selection.json',selection);print(selection,flush=True)
    results={}
    for scenario,(pattern,rate) in enumerate(protocol['scenarios']):
        rows=[]
        for i,path in enumerate(protocol['test']):
            held=masks[f'{pattern}_{rate}_{i}'];pred,values,truth=prediction(path,held,41022+i)
            guarded=apply_policy(pred,pred,pred,np.zeros_like(pred),values,stats,False,'withhold',length)
            rows.append(metrics(pred,truth,held,std,scale,~guarded[4]))
        results[pattern+'_'+str(rate)]=aggregate(rows,std,scale);print(pattern,rate,results[pattern+'_'+str(rate)]['raw']['standardized_mae'],flush=True)
    assert candidate_hash==digest(ROOT/'runs/full-training/ablations/no_adversarial/models/chronogan_best.pt')
    assert all(digest(ROOT/f)==h for f,h in protocol['frozen_hashes'].items())
    write(out/'results.json',{'selection':selection,'results':results,'seconds':time.perf_counter()-clock,'device':'cuda','peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'checkpoint_unchanged':True,'deployment_changed':False})

if __name__=='__main__':main()
