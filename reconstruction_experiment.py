"""Bounded, validation-selected CUDA inference experiment; existing study is read-only."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from preprocess import FEATURES,read_patient,tensors
from train import load_checkpoint
from utils.experiment import generator_from_state
from utils.baselines import RecurrentImputer
from utils.dataset import recompute_delta
from utils.clinical_validation import apply_policy,domain_invalid,registry

ROOT=Path(__file__).resolve().parent
SCENARIOS=[('random',.1),('random',.2),('random',.4),('block',.2),('feature',.2)]

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False))

def masking(known,pattern,rate,rng):
    if pattern=='random':return known&(rng.random(known.shape)<rate)
    if pattern=='feature':return known&(rng.random(known.shape[1])<rate)[None,:]
    span=max(1,int(len(known)*rate));start=rng.integers(0,len(known)-span+1,size=known.shape[1])
    t=np.arange(len(known))[:,None]
    return known&(t>=start)&(t<start+span)

def baselines(values,median,length):
    # Independent windows match the generator. Linear uses future measurements
    # and is explicitly retrospective, excluded from causal candidate selection.
    forward=np.empty_like(values);linear=np.empty_like(values)
    for start in range(0,len(values),length):
        chunk=pd.DataFrame(values[start:start+length])
        forward[start:start+length]=chunk.ffill().fillna(pd.Series(median)).to_numpy()
        linear[start:start+length]=chunk.interpolate(limit_direction='both').fillna(pd.Series(median)).to_numpy()
    return {'median':np.where(np.isfinite(values),values,median),'forward':forward,'linear_retrospective':linear}

def metrics(pred,truth,held,std,scale,released):
    output={}
    specs=registry()['variables']
    invalid=np.column_stack([domain_invalid(pred[:,j],specs[f]) for j,f in enumerate(FEATURES)])
    for name,subset in [('raw',held),('released',held&released),('withheld',held&~released)]:
        valid=subset&np.isfinite(pred);n=int(valid.sum());delta=np.where(valid,pred-truth,0.)
        absolute=np.abs(delta).sum(0);square=np.square(delta).sum(0);count=valid.sum(0)
        output[name]={'n':n,'standardized_mae':float((absolute/std).sum()/n) if n else None,
            'standardized_rmse':float(np.sqrt((square/std**2).sum()/n)) if n else None,
            'normalized_mae':float((absolute/scale).sum()/n) if n else None,
            'invalid_count':int((invalid&valid).sum()),
            'per_feature':{f:{'absolute_sum':float(absolute[j]),'squared_sum':float(square[j]),'n':int(count[j])} for j,f in enumerate(FEATURES)}}
    output['eligible']=int(held.sum());return output

def aggregate(rows,std,scale):
    result={};eligible=sum(r['eligible'] for r in rows)
    for group in ['raw','released','withheld']:
        counts=np.array([sum(r[group]['per_feature'][f]['n'] for r in rows) for f in FEATURES])
        absolute=np.array([sum(r[group]['per_feature'][f]['absolute_sum'] for r in rows) for f in FEATURES])
        square=np.array([sum(r[group]['per_feature'][f]['squared_sum'] for r in rows) for f in FEATURES]);n=int(counts.sum())
        result[group]={'n':n,'standardized_mae':float((absolute/std).sum()/n) if n else None,'standardized_rmse':float(np.sqrt((square/std**2).sum()/n)) if n else None,
            'normalized_mae':float((absolute/scale).sum()/n) if n else None,'invalid_rate':sum(r[group]['invalid_count'] for r in rows)/n if n else None,
            'per_feature':{f:{'mae':float(absolute[j]/counts[j]) if counts[j] else None,'rmse':float(np.sqrt(square[j]/counts[j])) if counts[j] else None,'n':int(counts[j])} for j,f in enumerate(FEATURES)}}
    result['coverage']=result['released']['n']/eligible if eligible else None
    result['withholding_rate']=1-result['coverage'] if eligible else None
    result['eligible']=eligible;return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cuda'],required=True);p.add_argument('--patients',type=int,default=64);p.add_argument('--validation-patients',type=int,default=64);p.add_argument('--training-patients',type=int,default=512);p.add_argument('--samples',type=int,default=20);p.add_argument('--output',default='runs/reconstruction-improvement/cuda-v1');args=p.parse_args()
    assert torch.cuda.is_available(),'CUDA is required; no CPU fallback'
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.12);torch.cuda.reset_peak_memory_stats()
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'results.json').exists(),'Use a fresh output directory; completed experiments are immutable'
    checkpoint=ROOT/'runs/full-training/models/chronogan_best.pt';state=load_checkpoint(checkpoint,'cpu');stats=state['stats'];splits=state['splits'];length=state['config']['sequence_length']
    sets=[set(splits[k]) for k in ['train','val','test']];assert not any(sets[i]&sets[j] for i in range(3) for j in range(i))
    frozen=[checkpoint,ROOT/'data/full/statistics.json',ROOT/'data/full/splits.json',ROOT/'runs/full-training/results/deployment-policy.json']
    hashes={str(f.relative_to(ROOT)):digest(f) for f in frozen}
    protocol={'device':'cuda','gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,'samples':args.samples,'seed':31022,
        'train':splits['train'][:args.training_patients],'val':splits['val'][:args.validation_patients],'test':splits['test'][:args.patients],
        'scenarios':SCENARIOS,'frozen_hashes':hashes,'selection':'Scalar GAN/causal-forward blend selected on validation random20 standardized MAE; candidates .25,.5,.75,1. No test selection.',
        'normalization':'Standardized errors divide by frozen training standard deviations; normalized MAE divides by training-subsample IQR, falling back to std for zero/absent IQR. Native pooled error omitted because units differ.',
        'policy':'Existing withholding unchanged; common support checks on every method. No deployment or training checkpoint modification.'}
    protocol_path=out/'protocol.json'
    if protocol_path.exists():assert json.loads(protocol_path.read_text())==json.loads(json.dumps(protocol))
    else:write(protocol_path,protocol)
    train=np.concatenate([read_patient(f).reindex(columns=FEATURES).to_numpy(dtype=float) for f in protocol['train']])
    median=np.array([np.nanmedian(train[:,j]) if np.isfinite(train[:,j]).any() else stats['mean'][j] for j in range(34)])
    std=np.array(stats['std']);q=np.nanpercentile(train,[25,75],axis=0);scale=q[1]-q[0];scale=np.where(np.isfinite(scale)&(scale>0),scale,std)
    write(out/'training-only-statistics.json',{'median':median.tolist(),'iqr_scale':scale.tolist(),'patients':len(protocol['train'])})
    model=generator_from_state(state,'cuda').eval();assert next(model.parameters()).is_cuda
    gru_state=load_checkpoint(ROOT/'runs/full-training/baselines/missingness_aware_gru/best.pt','cpu')
    gru=RecurrentImputer(34,gru_state['config']['hidden']).cuda().eval();gru.load_state_dict(gru_state['weights'])
    def predict(frame,seed):
        x,m,d=tensors(frame,stats);parts=[];recurrent=[];rng=torch.Generator(device='cuda').manual_seed(seed)
        with torch.no_grad():
            for start in range(0,len(x),length):
                xx,mm,dd=[torch.tensor(v[start:start+length],device='cuda')[None] for v in [x,m,d]];dd=recompute_delta(mm,dd);v=torch.ones_like(xx[:,:,0]);repeat=lambda a:a.repeat(args.samples,1,1)
                z=torch.randn(args.samples,xx.shape[1],model.noise,device='cuda',generator=rng)
                parts.append(model(repeat(xx),repeat(mm),repeat(dd),v.repeat(args.samples,1),z).mean(0).cpu().numpy())
                recurrent.append(gru(xx*mm,mm,dd,v)[0].cpu().numpy())
        return np.concatenate(parts)*std+stats['mean'],np.concatenate(recurrent)*std+stats['mean']
    candidate_scores={a:[0.,0] for a in [.25,.5,.75,1.]};mask_arrays={};clock=time.perf_counter()
    for i,path in enumerate(protocol['val']):
        frame=read_patient(path);truth=frame.reindex(columns=FEATURES).to_numpy(dtype=float);held=masking(np.isfinite(truth),'random',.2,np.random.default_rng(31022+i));values=truth.copy();values[held]=np.nan
        masked=frame.copy();masked[FEATURES]=values;gan,_=predict(masked,31022+i);forward=baselines(values,median,length)['forward']
        for a in candidate_scores:
            prediction=a*gan+(1-a)*forward;candidate_scores[a][0]+=float((np.abs(prediction-truth)/std)[held].sum());candidate_scores[a][1]+=int(held.sum())
        mask_arrays['val_'+str(i)]=held
    scores={str(a):total/n for a,(total,n) in candidate_scores.items()};alpha=float(min(scores,key=scores.get));selection={'alpha':alpha,'validation_standardized_mae':scores,'test_used':False}
    selection_path=out/'selection.json'
    if selection_path.exists():assert json.loads(selection_path.read_text())==selection
    else:write(selection_path,selection)
    print('FROZEN VALIDATION SELECTION',selection,flush=True)
    results={};detail=[];reason_counts={}
    for scenario,(pattern,rate) in enumerate(SCENARIOS):
        accum={name:[] for name in ['chronogan','median','forward','linear_retrospective','mask_aware_gru','selected_blend']}
        for i,path in enumerate(protocol['test']):
            frame=read_patient(path);truth=frame.reindex(columns=FEATURES).to_numpy(dtype=float);held=masking(np.isfinite(truth),pattern,rate,np.random.default_rng(41022+scenario*1000+i));values=truth.copy();values[held]=np.nan
            masked=frame.copy();masked[FEATURES]=values;gan,recurrent=predict(masked,41022+i)
            predictions=baselines(values,median,length);predictions.update(chronogan=gan,mask_aware_gru=recurrent);predictions['selected_blend']=alpha*gan+(1-alpha)*predictions['forward']
            mask_arrays[f'{pattern}_{rate}_{i}']=held
            for method,prediction in predictions.items():
                # MC interval calibration is not claimed for a blend; deterministic policy only.
                prediction=np.where(np.isfinite(values),values,prediction)
                checked=apply_policy(prediction,prediction,prediction,np.zeros_like(prediction),values,stats,False,'withhold',length)
                released=~checked[4];row=metrics(prediction,truth,held,std,scale,released);accum[method].append(row)
                if method=='chronogan':
                    for reason in checked[7][~np.isfinite(values)]:
                        for item in str(reason).split('; '):
                            if item:reason_counts[item]=reason_counts.get(item,0)+1
                # Gap bins count consecutive unavailable rows within model windows.
                gaps=np.zeros_like(values,dtype=int)
                for t in range(len(values)):gaps[t]=np.where(np.isfinite(values[t]),0,1+(gaps[t-1] if t%length else 0))
                for label,subset in [('all',held),('gap_1_3',held&(gaps<=3)),('gap_4_12',held&(gaps>=4)&(gaps<=12)),('gap_13_plus',held&(gaps>=13))]:
                    if subset.any():detail.append({'scenario':pattern+str(rate),'patient':i,'method':method,'gap':label,'input_missing_fraction':float(np.isnan(values).mean()),'n':int(subset.sum()),'standardized_absolute_sum':float((np.abs(prediction-truth)/std)[subset].sum())})
            if (i+1)%16==0:print(pattern,rate,i+1,'patients',flush=True)
        results[pattern+'_'+str(rate)]={method:aggregate(rows,std,scale) for method,rows in accum.items()}
        write(out/f'{pattern}-{rate}.json',results[pattern+'_'+str(rate)])
    assert all(digest(ROOT/f)==h for f,h in hashes.items())
    np.savez_compressed(out/'masks.npz',**mask_arrays);pd.DataFrame(detail).to_csv(out/'gap-and-missingness.csv',index=False)
    report={'protocol':protocol,'selection':selection,'results':results,'withholding_reason_counts_overlapping':reason_counts,'seconds':time.perf_counter()-clock,
        'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_cuda_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'preserved_hashes':True,
        'deployment_changed':False,'limitations':['Limited fixed patient subsets, one split and mask seed; no confidence intervals.','Previously audited test cohort is not a pristine final confirmation set.','Validation selects an inference blend, not new GAN weights.','Wholly unobserved features remain withheld.','No calibrated uncertainty for a blend; not installed into the demo.']}
    write(out/'results.json',report)
    rows=[]
    for scenario,methods in results.items():
        for method,value in methods.items():rows.append({'scenario':scenario,'method':method,'mae':value['raw']['standardized_mae'],'rmse':value['raw']['standardized_rmse'],'coverage':value['coverage']})
    pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False)
    print(json.dumps({'selection':selection,'seconds':report['seconds'],'peak_cuda_mib':report['peak_cuda_allocated_mib'],'comparison':rows},indent=2),flush=True)

if __name__=='__main__':main()
