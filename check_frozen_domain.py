"""Secondary predeclared validation-first check: frozen GRU, fixed domain link.

No training, output clipping, checkpoint writing, or policy changes.
"""
import json
import argparse
import time
import numpy as np
import torch
from models.gap_candidate import GapCandidate
from train import load_checkpoint
from utils.clinical_validation import registry
from utils.experiment import atomic_json
from research_weight_study import ROOT,patient_arrays,infer,score,summarize,bootstrap,hashfile

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='runs/gap-feature-study/cuda-v1');args=parser.parse_args()
    assert torch.cuda.is_available();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.12);torch.cuda.reset_peak_memory_stats()
    free,_=torch.cuda.mem_get_info();assert free>2*1024**3
    reference=ROOT/'runs/loss-weight-study/cuda-v1';parent=ROOT/args.output;out=parent/'frozen-domain';out.mkdir(exist_ok=True)
    assert not (out/'results.json').exists(),'Completed comparison is immutable'
    prior=json.loads((reference/'protocol.json').read_text());stats=json.loads((ROOT/'data/full/statistics.json').read_text());std=np.array(stats['std']);base=load_checkpoint(ROOT/'runs/full-training/baselines/missingness_aware_gru/best.pt');length=base['config']['sequence_length']
    indices=[i for i,f in enumerate(stats['features']) if registry()['variables'][f]['minimum']==0.]
    model=GapCandidate(34,base['config']['hidden'],stats,indices,domain=True).cuda().eval();model.base.load_state_dict(base['weights'])
    baseline=GapCandidate(34,base['config']['hidden'],stats,indices).cuda().eval();baseline.base.load_state_dict(base['weights'])
    protocol={'source_checkpoint_sha256':hashfile(ROOT/'runs/full-training/baselines/missingness_aware_gru/best.pt'),'reference_protocol_sha256':hashfile(reference/'protocol.json'),'validation_mask_seeds':[77000,78000,79000],
        'link':'Fixed beta20 softplus lower-bound link in training-standardized units, minimum=0 variables only; numerical one-ULP lower-bound guard. No new upper caps. Original values and original withholding unchanged.',
        'decision':'Validation MAE lower on all three fixed masks, then require paired test MAE CI upper<0 in random20 and block20 and no six-scenario mean MAE/RMSE regression vs original GRU. No deployment or clinical-validation claim.',
        'rationale':'Fine-tuned variants failed validation. Isolate fixed, previously specified mathematical-domain link without changing pretrained weights or tuning its temperature. This is secondary exploratory analysis; test patients are not independent confirmation.'}
    atomic_json(out/'protocol.json',protocol);validations=[];clock=time.perf_counter()
    for seed in protocol['validation_mask_seeds']:
        arrays,records,_=patient_arrays(prior['val'],stats,length,('random',.2),seed)
        row={}
        for name,network in [('baseline',baseline),('linked',model)]:
            predictions=infer(network,'recurrent',arrays,records,stats,1,0,'cuda',pause=.05);row[name]=summarize(score(predictions,records,stats,length),std)['raw']['standardized_mae']
        validations.append(row)
    validation_pass=all(v['linked']<v['baseline'] for v in validations);atomic_json(out/'validation-selection.json',{'scores':validations,'passed':validation_pass,'test_used':False});print('FROZEN LINK VALIDATION',validations,validation_pass,flush=True)
    results={}
    for si,scenario in enumerate(prior['scenarios']):
        key=scenario[0]+'_'+str(scenario[1]);replicates=[]
        for seed in prior['mask_seeds']:
            arrays,records,masks=patient_arrays(prior['test'],stats,length,scenario,seed+si*10000);saved=np.load(reference/f'masks-{key}-{seed}.npz');assert all(np.array_equal(m,saved[str(i)]) for i,m in enumerate(masks))
            predictions=infer(model,'recurrent',arrays,records,stats,1,0,'cuda',pause=.05);replicates.append(score(predictions,records,stats,length))
            latent=[]
            with torch.no_grad():
                from utils.dataset import recompute_delta
                for start in range(0,len(arrays[0]),16):
                    x,m,d,v=[a[start:start+16].cuda() for a in arrays];model(x*m,m,recompute_delta(m,d),v);latent.append(model.prelink.cpu().numpy())
            np.savez_compressed(out/f'prelink-{key}-{seed}.npz',normalized_windows=np.concatenate(latent))
        a=np.stack(replicates)[None];baseline_metrics=np.load(reference/f'patient-metrics-{key}.npz')['mask_aware_gru']
        np.savez_compressed(out/f'patient-metrics-{key}.npz',linked=a,baseline=baseline_metrics)
        summary=summarize(a,std);summary['vs_gru']=bootstrap(a,baseline_metrics,std);results[key]=summary;print('LINK EVALUATED',key,summary['raw']['standardized_mae'],flush=True)
    baseline_results=json.loads((reference/'results.json').read_text())['results']
    retained=validation_pass and all(results[s]['vs_gru']['paired_mae_difference_vs_original_95_ci'][1]<0 for s in ['random_0.2','block_0.2'])
    for metric in ['standardized_mae','standardized_rmse']:retained=retained and np.mean([v['raw'][metric] for v in results.values()])<=np.mean([v['mask_aware_gru']['raw'][metric] for v in baseline_results.values()])
    assert all(hashfile(ROOT/f)==h for f,h in prior['frozen'].items())
    atomic_json(out/'results.json',{'protocol':protocol,'validation':validations,'results':results,'retained_for_further_research':bool(retained),'seconds':time.perf_counter()-clock,'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'weights_unchanged':True,'dashboard_modified':False})
    print('FROZEN DOMAIN COMPLETE',bool(retained),flush=True)

if __name__=='__main__':main()
