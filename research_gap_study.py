"""CUDA pilot: GRU control, learned feature gap carry, block training, domain link."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from models.gap_candidate import GapCandidate
from preprocess import FEATURES,PatientDataset
from train import load_checkpoint
from utils.dataset import artificial_mask,recompute_delta
from utils.losses import masked_mean
from utils.clinical_validation import registry,domain_invalid
from utils.experiment import atomic_json,atomic_save
from research_weight_study import ROOT,patient_arrays,infer,score,summarize,bootstrap,hashfile

SEEDS=[42,43,44]
VARIANTS=['control','gap_carry','block_training','nonnegative_link']

def build(base,variant):
    indices=[j for j,f in enumerate(FEATURES) if registry()['variables'][f]['minimum']==0.]
    model=GapCandidate(34,base['config']['hidden'],base['stats'],indices,variant=='gap_carry',variant=='nonnegative_link').cuda()
    model.base.load_state_dict(base['weights']);return model

def train_one(base,variant,seed,dataset,val_arrays,val_records,out,args):
    folder=out/'models'/f'{variant}-{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return folder/'best.pt'
    assert not (folder/'history.json').exists(),'Partial adaptation: use fresh output directory'
    torch.manual_seed(seed);model=build(base,variant);optimizer=torch.optim.Adam(model.parameters(),lr=.0001);history=[];best=float('inf')
    std=np.array(base['stats']['std']);length=base['config']['sequence_length']
    for epoch in range(args.epochs):
        clock=time.perf_counter();model.train();order=torch.randperm(len(dataset));losses=[];norms=[]
        for index,start in enumerate(range(0,len(order),32)):
            batch=next(iter(DataLoader(torch.utils.data.Subset(dataset,order[start:start+32].tolist()),batch_size=32)))
            x,m,d,v,_=[a.cuda() for a in batch]
            # Generate both mask types for EVERY variant so RNG advancement,
            # data order and subsequent random draws stay exactly matched.
            random_visible,random_held=artificial_mask(m,v,.2)
            block_visible,block_held=artificial_mask(m,v,.2,block=True)
            use_block=variant=='block_training' and (index+epoch)%2==1
            visible,held=(block_visible,block_held) if use_block else (random_visible,random_held)
            dd=recompute_delta(visible,d);optimizer.zero_grad(set_to_none=True)
            pred=model(x*visible,visible,dd,v);loss=masked_mean((pred-x)**2,held)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite masked training loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step()
            if variant=='gap_carry':
                with torch.no_grad():model.carry_weight.clamp_(0,1)
            losses.append(loss.item());norms.append(norm.item());time.sleep(args.pause)
        model.eval();predictions=infer(model,'recurrent',val_arrays,val_records,base['stats'],1,77777,'cuda',pause=args.pause)
        validation=summarize(score(predictions,val_records,base['stats'],length),std)['raw']['standardized_mae']
        row={'epoch':epoch+1,'validation_mae':validation,'training_masked_mse':float(np.mean(losses)),'gradient_norm_before_clip':float(np.mean(norms)),'seconds':time.perf_counter()-clock};history.append(row)
        if validation<best:
            best=validation;atomic_save(folder/'best.pt',{'weights':model.state_dict(),'variant':variant,'seed':seed,'stats':base['stats'],'config':{**base['config'],'run_dir':str(folder.relative_to(ROOT))},'epoch':epoch+1,'best_validation_mae':best,'version':f'gap-pilot-{variant}-{seed}'})
        atomic_json(folder/'history.json',history);print(variant,seed,row,flush=True)
    atomic_json(folder/'complete.json',{'best_validation_mae':best,'epochs':args.epochs,'saved_as_inference_only':True})
    del model,optimizer;torch.cuda.empty_cache();return folder/'best.pt'

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cuda'],required=True);p.add_argument('--output',default='runs/gap-feature-study/cuda-v1');p.add_argument('--epochs',type=int,default=3);p.add_argument('--pause',type=float,default=.05);args=p.parse_args()
    assert torch.cuda.is_available();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.12);torch.cuda.reset_peak_memory_stats()
    free,total=torch.cuda.mem_get_info();assert free>2*1024**3,'Wait for at least 2 GiB free VRAM'
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True);assert not (out/'results.json').exists(),'Completed results are immutable'
    reference=ROOT/'runs/loss-weight-study/cuda-v1';reference_results=json.loads((reference/'results.json').read_text());prior=reference_results['protocol']
    for f,h in prior['frozen'].items():assert hashfile(ROOT/f)==h
    path=ROOT/'runs/full-training/baselines/missingness_aware_gru/best.pt';base=load_checkpoint(path)
    stats=base['stats']=json.loads((ROOT/'data/full/statistics.json').read_text());std=np.array(stats['std']);length=base['config']['sequence_length']
    protocol={'train':prior['train'],'val':prior['val'],'test':prior['test'],'training_seeds':SEEDS,'mask_seeds':prior['mask_seeds'],'scenarios':prior['scenarios'],'variants':VARIANTS,'epochs':args.epochs,'learning_rate':.0001,'batch_size':32,'pause_seconds':args.pause,
        'initial_checkpoint_sha256':hashfile(path),'reference_results_sha256':hashfile(reference/'results.json'),'protected_files':prior['frozen'],
        'selection':'Best epoch per seed using fixed random20 validation masks; candidate eligible for retention only if validation improves for all 3 seeds vs matched control and mean validation improves vs unchanged GRU. Final support additionally requires paired test MAE CI below zero vs GRU in random20 and block20, and no overall six-scenario mean MAE or RMSE regression. No deployment change regardless.',
        'design':'Same pretrained GRU and matched seeds. Four isolated interventions: unchanged fine-tuning; feature-specific learned causal carry with log-gap decay; alternating random/block artificial masks; smooth softplus lower-bound link only for registry minimum=0. All original observations and withholding rules preserved.',
        'scope':'64 previously audited patients; exploratory warm-start pilot, not independent clinical or from-scratch validation. Candidate deterministic: no calibrated uncertainty claims.',
        'environment':{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'free_vram_mib_at_start':free/2**20}}
    protocol_path=out/'protocol.json'
    if protocol_path.exists():assert json.loads(protocol_path.read_text())==protocol
    else:atomic_json(protocol_path,protocol)
    clock=time.perf_counter();train_data=PatientDataset(protocol['train'],stats,length)
    va,vr,_=patient_arrays(protocol['val'],stats,length,('random',.2),77000)
    original=build(base,'control').eval();original_pred=infer(original,'recurrent',va,vr,stats,1,77777,'cuda',pause=args.pause)
    baseline_val=summarize(score(original_pred,vr,stats,length),std)['raw']['standardized_mae'];del original;torch.cuda.empty_cache()
    models={}
    for variant in VARIANTS:
        models[variant]=[]
        for seed in SEEDS:
            atomic_json(out/'progress.json',{'stage':'training','variant':variant,'seed':seed})
            models[variant].append(train_one(base,variant,seed,train_data,va,vr,out,args))
    validation={v:[json.loads(p.with_name('complete.json').read_text())['best_validation_mae'] for p in paths] for v,paths in models.items()}
    eligible={v:all(a<b for a,b in zip(validation[v],validation['control'])) and float(np.mean(validation[v]))<baseline_val for v in VARIANTS if v!='control'}
    atomic_json(out/'validation-selection.json',{'original_gru_validation_mae':baseline_val,'validation_by_seed':validation,'eligible_before_test':eligible,'test_used':False})
    atomic_json(out/'model-hashes.json',{v:[{'path':str(p.relative_to(ROOT)),'sha256':hashfile(p)} for p in paths] for v,paths in models.items()})
    results={};gap_rows=[];prelink_counts={}
    for si,scenario in enumerate(protocol['scenarios']):
        key=scenario[0]+'_'+str(scenario[1]);evaluated={v:[] for v in VARIANTS};prelink_counts[key]={}
        for mask_seed in protocol['mask_seeds']:
            arrays,records,masks=patient_arrays(protocol['test'],stats,length,scenario,mask_seed+si*10000)
            known_masks=np.load(reference/f'masks-{key}-{mask_seed}.npz')
            assert all(np.array_equal(mask,known_masks[str(i)]) for i,mask in enumerate(masks))
            for variant,paths in models.items():
                replicas=[]
                for seed,path in zip(SEEDS,paths):
                    state=load_checkpoint(path);model=build(base,variant);model.load_state_dict(state['weights']);model.eval()
                    pred=infer(model,'recurrent',arrays,records,stats,1,mask_seed+90000,'cuda',pause=args.pause)
                    replicas.append(score(pred,records,stats,length))
                    for i,(prediction,(truth,values,held,_,_)) in enumerate(zip(pred,records)):
                        gap=np.zeros_like(values,dtype=int)
                        for t in range(len(values)):gap[t]=np.where(np.isfinite(values[t]),0,1+(gap[t-1] if t%length else 0))
                        for label,condition in [('1-3',gap<=3),('4-12',(gap>=4)&(gap<=12)),('13+',gap>=13)]:
                            subset=held&condition;n=int(subset.sum())
                            if n:gap_rows.append({'scenario':key,'model':variant,'train_seed':seed,'mask_seed':mask_seed,'patient':i,'gap':label,'n':n,'absolute_standardized_sum':float((np.abs(prediction-truth)/std)[subset].sum()),'squared_standardized_sum':float(((prediction-truth)/std)[subset].__pow__(2).sum())})
                    # Preserve unconstrained latent predictions for a full batch.
                    # Patient metrics refer to the trained linked output, not logits.
                    if variant=='nonnegative_link':
                        collected=[]
                        with torch.no_grad():
                            for start in range(0,len(arrays[0]),16):
                                xx,mm,dd,vv=[a[start:start+16].cuda() for a in arrays];model(xx*mm,mm,recompute_delta(mm,dd),vv);collected.append(model.prelink.cpu().numpy())
                        np.savez_compressed(out/f'prelink-{key}-{mask_seed}-{seed}.npz',normalized_windows=np.concatenate(collected))
                    del model;torch.cuda.empty_cache()
                evaluated[variant].append(np.stack(replicas))
            atomic_json(out/'progress.json',{'stage':'evaluation','scenario':key,'mask_seed':mask_seed});print('EVALUATED',key,mask_seed,flush=True)
        metrics={v:np.stack(a,axis=1) for v,a in evaluated.items()}
        archives=np.load(reference/f'patient-metrics-{key}.npz')
        for name in ['chronogan','reconstruction_only','mask_aware_gru']:metrics[name]=archives[name]
        np.savez_compressed(out/f'patient-metrics-{key}.npz',**metrics)
        results[key]={}
        for name,a in metrics.items():
            summary=summarize(a,std);summary['vs_gru']=bootstrap(a,metrics['mask_aware_gru'],std);summary['vs_control']=bootstrap(a,metrics['control'],std);results[key][name]=summary
    retained={}
    for variant in eligible:
        supported=eligible[variant] and all(results[s][variant]['vs_gru']['paired_mae_difference_vs_original_95_ci'][1]<0 for s in ['random_0.2','block_0.2'])
        for metric in ['standardized_mae','standardized_rmse']:
            supported=supported and np.mean([v[variant]['raw'][metric] for v in results.values()])<=np.mean([v['mask_aware_gru']['raw'][metric] for v in results.values()])
        retained[variant]=bool(supported)
    assert all(hashfile(ROOT/f)==h for f,h in protocol['protected_files'].items())
    pd.DataFrame(gap_rows).to_csv(out/'gap-errors.csv',index=False)
    atomic_json(out/'results.json',{'protocol':protocol,'validation':validation,'eligible_before_test':eligible,'retained_for_further_research':retained,'results':results,'seconds':time.perf_counter()-clock,'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'dashboard_modified':False,'protected_files_preserved':True})
    atomic_json(out/'progress.json',{'stage':'complete'});print('COMPLETE',out,flush=True)

if __name__=='__main__':main()
