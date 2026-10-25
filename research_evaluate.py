"""One frozen, matched-mask test evaluation after validation-selected training."""
import gc
import hashlib
import json
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from download_dataset import ROOT
from preprocess import FEATURES
from utils.baselines import RecurrentImputer
from models.brits import BRITS
from utils.dataset import recompute_delta
from utils.experiment import atomic_json,generator_from_state
from utils.research import STUDY,SCENARIOS,cached_data,hide,simple,draws,hashfile
from train import load_checkpoint
from utils.research_diagnostics import temporal_errors,stochastic_variation

def reconstruction_example(x,visible,held,output,samples,valid,state,target):
    """Prespecified first test window, showing ground truth only where observed."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import csv
    features=state['stats']['features']
    names=['HR','O2Sat','Temp','SBP']
    lower=samples.quantile(.05,dim=1); upper=samples.quantile(.95,dim=1)
    length=int(valid[0].sum()); times=np.arange(length)
    fig,axes=plt.subplots(4,1,figsize=(11,10),sharex=True)
    records=[]
    for ax,name in zip(axes,names):
        j=features.index(name); sd=state['stats']['std'][j]; mean=state['stats']['mean'][j]
        native=lambda value:value[0,:length,j].float().cpu().numpy()*sd+mean
        truth=native(x); estimate=native(output); lo=native(lower); hi=native(upper)
        known=visible[0,:length,j].bool().cpu().numpy(); scored=held[0,:length,j].bool().cpu().numpy()
        ax.plot(times,estimate,label='Reconstruction'); ax.fill_between(times,lo,hi,alpha=.2,label='90% noise interval')
        ax.scatter(times[known],truth[known],s=15,label='Visible observations')
        ax.scatter(times[scored],truth[scored],s=25,marker='x',label='Artificially hidden truth')
        ax.set_ylabel(name+' (native unit)')
        for t in times:
            records.append([int(t),name,float(truth[t]) if known[t] or scored[t] else '',bool(known[t]),bool(scored[t]),float(estimate[t]),float(lo[t]),float(hi[t])])
    axes[0].legend(fontsize=8,ncol=2); axes[-1].set_xlabel('Hour within first frozen test window')
    fig.tight_layout(); fig.savefig(target/'reconstruction-example.png'); plt.close(fig)
    with (target/'reconstruction-example.csv').open('w',newline='') as file:
        writer=csv.writer(file); writer.writerow(['hour','variable','observed_ground_truth','visible','held_out','estimate','noise_q05','noise_q95']); writer.writerows(records)


def evaluate():
    target=STUDY/'results'; complete=target/'reconstruction-complete.json'
    if complete.exists(): return
    from research_selection import select
    select()
    state=load_checkpoint(STUDY/'models'/'chronogan_best.pt')
    assert len(state['splits']['test'])==6051
    torch.set_num_threads(4); torch.manual_seed(42)
    loader=DataLoader(cached_data(state,'test'),batch_size=64,shuffle=False)
    generators={'chronogan':generator_from_state(state,'cuda').eval()}
    for name in ['no_mask','no_delta','no_adversarial']:
        saved=load_checkpoint(STUDY/'ablations'/name/'models'/'chronogan_best.pt')
        assert saved['splits']==state['splits'] and saved['stats']==state['stats']
        generators[name]=generator_from_state(saved,'cuda').eval()
    recurrent=RecurrentImputer(len(FEATURES),state['config']['hidden']).cuda().eval()
    recurrent.load_state_dict(load_checkpoint(STUDY/'baselines'/'missingness_aware_gru'/'best.pt')['weights'])
    brits=BRITS(len(FEATURES),state['config']['hidden']).cuda().eval()
    brits.load_state_dict(load_checkpoint(STUDY/'baselines'/'brits'/'best.pt')['weights'])
    methods=['mean','forward','linear_retrospective','recurrent','brits',*generators]
    patient_ids=[]
    for i,path in enumerate(state['splits']['test']):
        with open(path) as source: hours=sum(1 for _ in source)-1
        patient_ids.extend([i]*math.ceil(hours/24))
    assert len(patient_ids)==len(loader.dataset)
    patient_ids=np.array(patient_ids)
    rows=[]; uncertainty=[]; paired={}
    scale=np.array(state['stats']['std'])
    for pattern,rate in SCENARIOS:
        stem=f'{pattern}-{rate}'
        file=target/f'evaluation-{stem}.json'
        if file.exists():
            part=json.loads(file.read_text()); rows.extend(part['experiments']); uncertainty.extend(part['uncertainty']); continue
        accum={method:np.zeros((3,len(FEATURES))) for method in methods}
        temporal={method:np.zeros((2,len(FEATURES))) for method in methods}
        variation={method:np.zeros((2,len(FEATURES))) for method in generators}
        covers={method:np.zeros((4,len(FEATURES))) for method in generators}
        widths={method:np.zeros((4,len(FEATURES))) for method in generators}
        rng=torch.Generator(device='cuda').manual_seed(2026)
        digest=hashlib.sha256(); observed=0; position=0
        patient_errors={method:np.zeros((2,6051)) for method in methods} if pattern=='random' and rate==.2 else None
        with torch.no_grad():
            for batch_index,(x,m,d,v,_) in enumerate(loader):
                x,m,d,v=[a.cuda(non_blocking=True) for a in (x,m,d,v)]
                visible,held=hide(m,v,pattern,rate,rng); dd=recompute_delta(visible,d)
                digest.update(held.cpu().numpy().tobytes()); observed+=int(m.sum())
                counts=held.sum((0,1)).cpu().numpy()
                for method in methods:
                    if method in generators:
                        samples=draws(generators[method],x,visible,dd,v,samples=20,seed=9001+batch_index)
                        output=samples.mean(1)
                        variation_values=stochastic_variation(samples,held)
                        for j,value in enumerate(variation_values): variation[method][j]+=value.cpu().numpy()
                        if method=='chronogan' and pattern=='random' and rate==.2 and batch_index==0:
                            reconstruction_example(x,visible,held,output,samples,v,state,target)
                        for j,level in enumerate([.5,.8,.9,.95]):
                            lower=samples.quantile((1-level)/2,dim=1); upper=samples.quantile(1-(1-level)/2,dim=1)
                            covers[method][j]+=(((x>=lower)&(x<=upper))*held).sum((0,1)).cpu().numpy()
                            widths[method][j]+=((upper-lower)*held).sum((0,1)).cpu().numpy()
                        del samples
                    elif method=='recurrent':
                        with torch.autocast('cuda',dtype=torch.bfloat16): output=recurrent(x*visible,visible,dd,v).float()
                    elif method=='brits':
                        with torch.autocast('cuda',dtype=torch.bfloat16): output=brits(x*visible,visible,dd,v).float()
                    else: output=simple(x,visible,method)
                    error=(output-x)*held
                    temporal_values=temporal_errors(output,x,m,held,v)
                    for j,value in enumerate(temporal_values): temporal[method][j]+=value.cpu().numpy()
                    assert torch.isfinite(error).all()
                    accum[method][0]+=error.abs().sum((0,1)).cpu().numpy()
                    accum[method][1]+=error.square().sum((0,1)).cpu().numpy()
                    accum[method][2]+=counts
                    if patient_errors is not None:
                        ids=patient_ids[position:position+len(x)]
                        np.add.at(patient_errors[method][0],ids,error.abs().sum((1,2)).cpu().numpy())
                        np.add.at(patient_errors[method][1],ids,held.sum((1,2)).cpu().numpy())
                position+=len(x)
                if (batch_index+1)%25==0: print('Test evaluation',pattern,rate,batch_index+1,'/',len(loader),flush=True)
        part_rows=[]; part_uncertainty=[]
        for method,(absolute,squared,count) in accum.items():
            per={f:{'mae':float(absolute[j]/count[j]*scale[j]),'rmse':float(np.sqrt(squared[j]/count[j])*scale[j]),'n':int(count[j])} if count[j] else None for j,f in enumerate(FEATURES)}
            part_rows.append({'method':method,'pattern':pattern,'rate':rate,'normalized_mae':float(absolute.sum()/count.sum()),'normalized_rmse':float(np.sqrt(squared.sum()/count.sum())),'per_variable':per,'held_out_cells':int(count.sum()),'observed_cells':observed,'actual_hiding_fraction':float(count.sum()/observed),'mask_sha256':digest.hexdigest()})
            change_error,change_count=temporal[method]
            part_rows[-1]['temporal_normalized_mae']=float(change_error.sum()/change_count.sum()) if change_count.sum() else None
            part_rows[-1]['temporal_scored_pairs']=int(change_count.sum())
            part_rows[-1]['per_variable_temporal_mae_original_units_per_hour']={f:float(change_error[k]/change_count[k]*scale[k]) if change_count[k] else None for k,f in enumerate(FEATURES)}
            if method in generators:
                for j,level in enumerate([.5,.8,.9,.95]):
                    part_uncertainty.append({'method':method,'pattern':pattern,'rate':rate,'nominal_coverage':level,'empirical_coverage':float(covers[method][j].sum()/count.sum()),'mean_normalized_width':float(widths[method][j].sum()/count.sum()),'per_variable':{f:{'coverage':float(covers[method][j,k]/count[k]),'width_original_units':float(widths[method][j,k]/count[k]*scale[k])} if count[k] else None for k,f in enumerate(FEATURES)}})
                    part_uncertainty[-1]['mean_normalized_sample_std']=float(variation[method][0].sum()/count.sum())
                    part_uncertainty[-1]['fraction_near_zero_variation']=float(variation[method][1].sum()/count.sum())
                    for k,f in enumerate(FEATURES):
                        if count[k]:
                            part_uncertainty[-1]['per_variable'][f]['mean_sample_std_original_units']=float(variation[method][0,k]/count[k]*scale[k])
                            part_uncertainty[-1]['per_variable'][f]['fraction_near_zero_variation']=float(variation[method][1,k]/count[k])
        if patient_errors is not None:
            np.savez(target/'paired-patient-errors.npz',**patient_errors)
        atomic_json(file,{'experiments':part_rows,'uncertainty':part_uncertainty})
        rows.extend(part_rows); uncertainty.extend(part_uncertainty)
        print('Completed test protocol',stem,flush=True)
    # Approximate familywise-95% paired patient-cluster bootstrap for three prespecified contrasts.
    errors=np.load(target/'paired-patient-errors.npz'); rng=np.random.default_rng(42)
    deltas={name:[] for name in ['no_mask','no_delta','no_adversarial']}
    for _ in range(2000):
        ids=rng.integers(0,6051,6051); denominator=errors['chronogan'][1,ids].sum()
        for name in deltas: deltas[name].append(float((errors[name][0,ids]-errors['chronogan'][0,ids]).sum()/denominator))
    ablations=[]
    for name,values in deltas.items():
        point=float((errors[name][0]-errors['chronogan'][0]).sum()/errors['chronogan'][1].sum())
        ablations.append({'variant':name,'ablated_minus_full_normalized_mae':point,'patient_bootstrap_bonferroni_interval':np.quantile(values,[.05/6,1-.05/6]).tolist(),'interpretation':'Positive difference favors full architecture; one training seed, approximate patient-cluster intervals; no clinical-benefit conclusion.'})
    atomic_json(target/'ablations.json',{'contrasts':ablations,'protocol':'Same seed, optimizer, maximum epochs, validation early stopping, frozen patients and matched test masks. no_mask removes explicit encoder mask channels and mask-conditioned input decay; masks remain for compositing and GAN targets. no_delta zeros all elapsed-time encoding.'})
    atomic_json(target/'evaluation.json',{'checkpoint_sha256':hashfile(STUDY/'models'/'chronogan_best.pt'),'patients':6051,'experiments':rows,'protocol':'Matched, reproducible originally-observed-only masks; per-variable native-unit errors, pooled standardized errors; linear interpolation is retrospective only; laboratory grouping is an acquisition stress scenario, not a validated clinical missingness mechanism. BRITS is a reconstruction-only adaptation with bidirectional RITS and consistency, excluded from prospective prediction.'})
    atomic_json(target/'uncertainty.json',{'samples':20,'results':uncertainty,'interpretation':'Uncalibrated generative noise quantiles tested against hidden known measurements. Empirical coverage diagnostics do not establish clinical confidence intervals.'})
    from utils.visualization import comparison_plot
    comparison_plot(rows,target)
    atomic_json(complete,{'test_patients':6051,'scenarios':len(SCENARIOS),'methods':methods,'test_used_for_model_selection':False})

if __name__=='__main__': evaluate()
