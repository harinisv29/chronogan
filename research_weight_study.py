"""Isolated CUDA warm-start loss-weight study with crossed masking seeds.

Never writes the existing study, dashboard model, or frozen deployment choice.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse
import copy
import hashlib
import json
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from preprocess import FEATURES,PatientDataset,read_patient,tensors
from models.discriminator import Discriminator
from utils.baselines import RecurrentImputer
from utils.dataset import artificial_mask,recompute_delta
from utils.experiment import generator_from_state,atomic_json,atomic_save
from utils.losses import generator_loss,masked_mean,prepare_physiological_bounds
from utils.clinical_validation import apply_policy,domain_invalid,registry
from train import load_checkpoint
from reconstruction_experiment import masking

ROOT=Path(__file__).resolve().parent
SCENARIOS=[('random',.1),('random',.2),('random',.4),('random',.6),('block',.2),('block',.4)]
WEIGHTS=[('rec10_adv0',10.,0.),('rec10_adv001',10.,.01),('rec10_adv005',10.,.05),('rec10_adv01',10.,.1),('rec20_adv01',20.,.1)]
TRAIN_SEEDS=[42,43,44]
MASK_SEEDS=[1101,2202,3303]

def hashfile(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def patient_arrays(paths,stats,length,scenario=None,seed=0):
    records=[];windows=[];masks=[]
    for i,path in enumerate(paths):
        frame=read_patient(path);truth=frame.reindex(columns=FEATURES).to_numpy(dtype=float)
        held=np.zeros_like(truth,dtype=bool) if scenario is None else masking(np.isfinite(truth),*scenario,np.random.default_rng(seed+i))
        frame=frame.copy();values=truth.copy();values[held]=np.nan;frame[FEATURES]=values
        x,m,d=tensors(frame,stats);begin=len(windows)
        for start in range(0,len(x),length):
            n=min(length,len(x)-start);pad=lambda a:np.pad(a,((0,length-n),(0,0)))
            windows.append((pad(x[start:start+length]),pad(m[start:start+length]),pad(d[start:start+length]),np.r_[np.ones(n),np.zeros(length-n)].astype(np.float32)))
        records.append((truth,values,held,begin,len(windows)));masks.append(held)
    arrays=tuple(torch.from_numpy(np.stack([w[k] for w in windows])).float() for k in range(4))
    return arrays,records,masks

def infer(model,kind,arrays,records,stats,samples,seed,device,batch=16,pause=.025):
    outputs=[];rng=torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        for start in range(0,len(arrays[0]),batch):
            x,m,d,v=[a[start:start+batch].to(device) for a in arrays];d=recompute_delta(m,d)
            if kind=='generator':
                repeat=lambda a:a.repeat_interleave(samples,0)
                z=torch.randn(len(x)*samples,x.shape[1],model.noise,device=device,generator=rng)
                result=model(repeat(x*m),repeat(m),repeat(d),repeat(v),z).reshape(len(x),samples,*x.shape[1:]).mean(1)
            else:result=model(x*m,m,d,v)
            outputs.append(result.cpu().numpy())
            if pause:time.sleep(pause)
    normalized=np.concatenate(outputs);std=np.array(stats['std']);mean=np.array(stats['mean']);out=[]
    for truth,values,held,begin,end in records:
        raw=normalized[begin:end].reshape(-1,len(FEATURES))[:len(truth)]*std+mean
        raw=np.where(np.isfinite(values),values,raw)
        np.testing.assert_array_equal(raw[np.isfinite(values)],values[np.isfinite(values)])
        out.append(raw)
    return out

def score(predictions,records,stats,length):
    """Per patient/subset/metric/feature sufficient stats, not pooled native units.

    subset raw/released/withheld; metric absolute/squared/count/invalid/eligible.
    """
    result=np.zeros((len(records),3,5,len(FEATURES)));specs=registry()['variables']
    for i,(pred,(truth,values,held,_,_)) in enumerate(zip(predictions,records)):
        guarded=apply_policy(pred,pred,pred,np.zeros_like(pred),values,stats,False,'withhold',length)
        released=~guarded[4];invalid=np.column_stack([domain_invalid(pred[:,j],specs[f]) for j,f in enumerate(FEATURES)])
        for k,subset in enumerate([held,held&released,held&~released]):
            valid=subset&np.isfinite(pred);delta=np.where(valid,pred-truth,0.)
            result[i,k]=np.array([np.abs(delta).sum(0),np.square(delta).sum(0),valid.sum(0),(invalid&valid).sum(0),subset.sum(0)])
        assert not (invalid&held&released).any(),'Invalid prediction was released'
    return result

def summarize(a,std):
    # Sum every leading replicate/patient dimension; retain subset/metric/feature.
    total=a.sum(axis=tuple(range(a.ndim-3)));output={}
    for k,name in enumerate(['raw','released','withheld']):
        absolute,square,count,invalid,eligible=total[k];n=count.sum()
        output[name]={'standardized_mae':float((absolute/std).sum()/n) if n else None,'standardized_rmse':float(np.sqrt((square/std**2).sum()/n)) if n else None,'n':int(n),'nonfinite_count':int(eligible.sum()-n),'invalid_rate':float(invalid.sum()/n) if n else None,
            'per_feature':{f:{'mae_native':float(absolute[j]/count[j]) if count[j] else None,'rmse_native':float(np.sqrt(square[j]/count[j])) if count[j] else None,'n':int(count[j]),'invalid':int(invalid[j])} for j,f in enumerate(FEATURES)}}
    eligible=total[0,4].sum();output['coverage']=float(total[1,2].sum()/eligible) if eligible else None;output['withholding_rate']=1-output['coverage'] if eligible else None
    return output

def bootstrap(a,reference,std,iterations=1000):
    """Paired crossed resampling: patient clusters, mask seeds, adaptation seeds.

    Arrays training-replicate,mask-replicate,patient,subset,metric,feature.
    Archives have one training replicate: no fabricated training-seed replication.
    """
    rng=np.random.default_rng(96022);values=[];differences=[];rmse=[];coverage=[]
    def compact(x):
        return np.stack([(x[:,:,:,0,0]/std).sum(-1),(x[:,:,:,0,1]/std**2).sum(-1),x[:,:,:,0,2].sum(-1),x[:,:,:,1,2].sum(-1),x[:,:,:,0,4].sum(-1)],axis=-1)
    a=compact(a);reference=compact(reference)
    def measure(x):
        total=x.sum(axis=(0,1,2));return total[0]/total[2],np.sqrt(total[1]/total[2]),total[3]/total[4]
    for _ in range(iterations):
        patients=rng.integers(a.shape[2],size=a.shape[2]);masks=rng.integers(a.shape[1],size=a.shape[1]);runs=rng.integers(a.shape[0],size=a.shape[0])
        sampled=a[runs][:,masks][:,:,patients]
        base=(reference[runs] if reference.shape[0]==a.shape[0] else reference)[:,masks][:,:,patients]
        value,error,cov=measure(sampled);baseline=measure(base)[0]
        values.append(value);differences.append(value-baseline);rmse.append(error);coverage.append(cov)
    interval=lambda x:np.quantile(x,[.025,.975]).tolist()
    return {'mae_95_ci':interval(values),'rmse_95_ci':interval(rmse),'coverage_95_ci':interval(coverage),'paired_mae_difference_vs_original_95_ci':interval(differences),'bootstrap_iterations':iterations}

def train_variant(base,train_data,val_data,val_records,name,rec,adv,seed,out,args):
    folder=out/'adaptations'/f'{name}-seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    best_path=folder/'best.pt'
    if (folder/'complete.json').exists():return best_path
    assert not (folder/'history.json').exists(),'Partial run: use a fresh output directory rather than silently restarting'
    torch.manual_seed(seed);cfg={**base['config'],'reconstruction':rec,'adversarial':adv,'learning_rate':.0001,'seed':seed}
    g=generator_from_state(base,'cuda');d=Discriminator(len(FEATURES),cfg['hidden']).cuda();d.load_state_dict(base['discriminator'])
    go=torch.optim.Adam(g.parameters(),lr=cfg['learning_rate']);do=torch.optim.Adam(d.parameters(),lr=cfg['learning_rate']);bounds=prepare_physiological_bounds(base['stats'],'cuda')
    best=float('inf');history=[];std=np.array(base['stats']['std'])
    for epoch in range(args.epochs):
        clock=time.perf_counter();g.train();d.train();order=torch.randperm(len(train_data));sums=np.zeros(6);batches=0
        for start in range(0,len(order),32):
            x,m,delta,v,_=[a.cuda() for a in next(iter(DataLoader(torch.utils.data.Subset(train_data,order[start:start+32].tolist()),batch_size=32)))]
            visible,held=artificial_mask(m,v,.2);dd=recompute_delta(visible,delta);reveal=(torch.rand_like(m)<.9).float();hint=reveal*visible+.5*(1-reveal);unknown=1-reveal
            generated=g(x*visible,visible,dd,v);do.zero_grad(set_to_none=True)
            logits=d(generated.detach(),hint,v);dl=masked_mean(F.binary_cross_entropy_with_logits(logits,visible,reduction='none'),unknown*v[...,None])
            dl.backward();dn=torch.nn.utils.clip_grad_norm_(d.parameters(),5,error_if_nonfinite=True);do.step()
            for parameter in d.parameters():parameter.requires_grad_(False)
            go.zero_grad(set_to_none=True);gl,parts=generator_loss(generated,x,held,visible,v,d(generated,hint,v),unknown,cfg,base['stats'],True,bounds)
            if not torch.isfinite(gl):raise FloatingPointError('Nonfinite generator loss')
            gl.backward();gn=torch.nn.utils.clip_grad_norm_(g.parameters(),5,error_if_nonfinite=True);go.step()
            for parameter in d.parameters():parameter.requires_grad_(True)
            sums+=np.array([gl.item(),dl.item(),parts['reconstruction_mse'].item(),parts['adversarial_loss'].item(),gn.item(),dn.item()]);batches+=1
            time.sleep(args.pause)
        g.eval();pred=infer(g,'generator',val_data,val_records,base['stats'],args.samples,77777,'cuda',pause=args.pause)
        result=summarize(score(pred,val_records,base['stats'],cfg['sequence_length']),std);mae=result['raw']['standardized_mae']
        row={'epoch':epoch+1,'validation_mae':mae,'seconds':time.perf_counter()-clock,**dict(zip(['generator_loss','discriminator_loss','reconstruction_mse','adversarial_loss','generator_gradient_norm_before_clip','discriminator_gradient_norm_before_clip'],(sums/batches).tolist()))};history.append(row)
        if mae<best:
            best=mae;atomic_save(best_path,{'generator':g.state_dict(),'discriminator':d.state_dict(),'config':cfg,'stats':base['stats'],'splits':base['splits'],'version':f'isolated-warmstart-{name}-seed{seed}','epoch':epoch+1,'best':best})
        atomic_json(folder/'history.json',history);print(name,seed,row,flush=True)
    atomic_json(folder/'complete.json',{'best_validation_mae':best,'epochs':args.epochs,'selection':'fixed random20 validation only','warm_start':True})
    del g,d,go,do;torch.cuda.empty_cache();return best_path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['cuda'],required=True);parser.add_argument('--output',default='runs/loss-weight-study/cuda-v1');parser.add_argument('--train-patients',type=int,default=512);parser.add_argument('--val-patients',type=int,default=64);parser.add_argument('--test-patients',type=int,default=64);parser.add_argument('--epochs',type=int,default=3);parser.add_argument('--samples',type=int,default=20);parser.add_argument('--pause',type=float,default=.05);args=parser.parse_args()
    assert torch.cuda.is_available(),'CUDA is required';torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.12);torch.cuda.reset_peak_memory_stats()
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True);assert not (out/'results.json').exists(),'Completed runs are immutable'
    checkpoint=ROOT/'runs/full-training/models/chronogan_best.pt';base=load_checkpoint(checkpoint);stats=base['stats'];length=base['config']['sequence_length'];std=np.array(stats['std'])
    archived={'chronogan':checkpoint,'reconstruction_only':ROOT/'runs/full-training/ablations/no_adversarial/models/chronogan_best.pt','mask_aware_gru':ROOT/'runs/full-training/baselines/missingness_aware_gru/best.pt','no_mask':ROOT/'runs/full-training/ablations/no_mask/models/chronogan_best.pt','no_delta':ROOT/'runs/full-training/ablations/no_delta/models/chronogan_best.pt'}
    frozen={str(p.relative_to(ROOT)):hashfile(p) for p in list(archived.values())+[ROOT/'data/full/statistics.json',ROOT/'data/full/splits.json',ROOT/'runs/full-training/results/deployment-policy.json',ROOT/'clinical_variables.json']}
    splits=base['splits'];sets=[set(splits[k]) for k in ['train','val','test']];assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
    protocol={'train':splits['train'][:args.train_patients],'val':splits['val'][:args.val_patients],'test':splits['test'][:args.test_patients],'train_seeds':TRAIN_SEEDS,'mask_seeds':MASK_SEEDS,'scenarios':SCENARIOS,'weights':WEIGHTS,'epochs':args.epochs,'samples':args.samples,'learning_rate':.0001,'frozen':frozen,
        'design':'Same full-trained GAN warm start and discriminator, fresh Adam per adaptation, shared shuffles/masks/noise within training seed. Fixed random20 validation and noise; best of three epochs. Predeclared test comparisons, no test selection. Archives have ONE original training seed. Adaptations have THREE seeds, each crossed with THREE mask/noise seeds.',
        'limitations':'Warm-start short adaptation, not from-scratch weight optimization; test patients previously inspected; no claim of pristine confirmation. Confidence intervals describe patient/seed resampling, not clinical validity.',
        'environment':{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'device':'cuda','pause_seconds_per_batch':args.pause,'memory_fraction_cap':.12}}
    normalized=json.loads(json.dumps(protocol));protocol_path=out/'protocol.json'
    if protocol_path.exists():assert json.loads(protocol_path.read_text())==normalized
    else:atomic_json(protocol_path,protocol)
    started=time.perf_counter();train_data=PatientDataset(protocol['train'],stats,length);val_data,val_records,_=patient_arrays(protocol['val'],stats,length,('random',.2),77000)
    models={name:[path] for name,path in archived.items()}
    for name,rec,adv in WEIGHTS:
        models[name]=[]
        for seed in TRAIN_SEEDS:
            atomic_json(out/'progress.json',{'stage':'adaptation','variant':name,'seed':seed})
            models[name].append(train_variant(base,train_data,val_data,val_records,name,rec,adv,seed,out,args))
    # Freeze all validation choices and model hashes before opening this study's test masks.
    atomic_json(out/'evaluated-models.json',{name:[{'path':str(p.relative_to(ROOT)),'sha256':hashfile(p)} for p in paths] for name,paths in models.items()})
    evaluations={};mask_hashes={}
    for scenario_index,scenario in enumerate(SCENARIOS):
        key=scenario[0]+'_'+str(scenario[1]);by_model={name:[] for name in models}
        for mask_seed in MASK_SEEDS:
            arrays,records,masks=patient_arrays(protocol['test'],stats,length,scenario,mask_seed+scenario_index*10000)
            mask_hashes[key+'_'+str(mask_seed)]=hashlib.sha256(b''.join(m.tobytes() for m in masks)).hexdigest()
            mask_file=out/f'masks-{key}-{mask_seed}.npz';np.savez_compressed(mask_file,**{str(i):m for i,m in enumerate(masks)})
            for name,paths in models.items():
                replicates=[]
                for path in paths:
                    state=load_checkpoint(path);assert state.get('stats',stats)==stats
                    if name=='mask_aware_gru':
                        model=RecurrentImputer(34,state['config']['hidden']).cuda();model.load_state_dict(state['weights']);kind='recurrent'
                    else:model=generator_from_state(state,'cuda');kind='generator'
                    assert next(model.parameters()).is_cuda;model.eval()
                    pred=infer(model,kind,arrays,records,stats,args.samples,mask_seed+90000,'cuda',pause=args.pause)
                    replicates.append(score(pred,records,stats,length));del model;torch.cuda.empty_cache()
                by_model[name].append(np.stack(replicates))
            atomic_json(out/'progress.json',{'stage':'evaluation','scenario':key,'mask_seed':mask_seed});print('EVALUATED',key,mask_seed,flush=True)
        evaluations[key]={name:np.stack(values,axis=1) for name,values in by_model.items()}
        np.savez_compressed(out/f'patient-metrics-{key}.npz',**evaluations[key])
    results={};comparison=[];feature_rows=[]
    for scenario,methods in evaluations.items():
        results[scenario]={}
        for name,values in methods.items():
            result=summarize(values,std);result.update(bootstrap(values,methods['chronogan'],std));results[scenario][name]=result
            comparison.append({'scenario':scenario,'model':name,'mae':result['raw']['standardized_mae'],'rmse':result['raw']['standardized_rmse'],'coverage':result['coverage'],'invalid_rate':result['raw']['invalid_rate'],**{k:json.dumps(result[k]) for k in ['mae_95_ci','rmse_95_ci','coverage_95_ci','paired_mae_difference_vs_original_95_ci']}})
            for subset in ['raw','released','withheld']:
                for feature,value in result[subset]['per_feature'].items():feature_rows.append({'scenario':scenario,'model':name,'subset':subset,'feature':feature,**value})
    assert all(hashfile(ROOT/f)==h for f,h in frozen.items()),'A preserved artifact changed'
    atomic_json(out/'results.json',{'protocol':protocol,'results':results,'mask_sha256':mask_hashes,'seconds':time.perf_counter()-started,'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'frozen_artifacts_preserved':True,'dashboard_changed':False})
    pd.DataFrame(comparison).to_csv(out/'comparison.csv',index=False);pd.DataFrame(feature_rows).to_csv(out/'per-feature.csv',index=False)
    atomic_json(out/'progress.json',{'stage':'complete','seconds':time.perf_counter()-started});print('COMPLETE',out,flush=True)

if __name__=='__main__':main()
