"""Validation-only causal deployment selection, frozen before test evaluation."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.research import STUDY,cached_data,simple,draws,hashfile
from utils.dataset import artificial_mask,recompute_delta
from utils.experiment import atomic_json,atomic_save,generator_from_state
from utils.baselines import RecurrentImputer
from train import load_checkpoint

POLICY={
    'scope':'Lowest validation MAE among causal models, including baselines; explicitly chosen by the user before test evaluation.',
    'candidates':['mean','forward','recurrent','chronogan','no_mask','no_delta','no_adversarial'],
    'metric':'Pooled normalized MAE on originally observed validation values hidden at 20%; all candidates use the same mask.',
    'mask_seed':12345,'noise_seed':54321,'samples':20,'batch_size':64,
    'selection_inference':'Stochastic candidates use a 20-draw mean, matching the intended displayed point estimate. This is separate from the single-draw validation criterion used to select training checkpoints.',
    'tie_break':'Lowest MAE, then listed candidate order; no test metrics are read.',
    'exclusions':'BRITS and linear interpolation use future measurements; prediction heads are not imputers.',
    'uncertainty':'Deterministic winners expose no intervals for missing values; stochastic ChronoGAN remains a separately labelled comparison.',
}


def freeze_policy():
    path=STUDY/'results'/'deployment-policy.json'
    if path.exists():
        assert json.loads(path.read_text())==POLICY,'Frozen deployment policy changed'
    else:
        assert not list((STUDY/'results').glob('evaluation-*.json')),'Freeze the policy before test evaluation'
        atomic_json(path,POLICY)
    return path


def select():
    freeze_policy(); path=STUDY/'results'/'deployment-selection.json'
    if path.exists(): return json.loads(path.read_text())
    assert not list((STUDY/'results').glob('evaluation-*.json')),'Cannot select after looking at test results'
    state=load_checkpoint(STUDY/'models'/'chronogan_best.pt')
    loader=DataLoader(cached_data(state,'val'),batch_size=POLICY['batch_size'],shuffle=False)
    models={}; sources={}; states={}
    for name in ['chronogan','no_mask','no_delta','no_adversarial']:
        source=STUDY/'models/chronogan_best.pt' if name=='chronogan' else STUDY/'ablations'/name/'models/chronogan_best.pt'
        saved=load_checkpoint(source); assert saved['splits']==state['splits'] and saved['stats']==state['stats']
        models[name]=generator_from_state(saved,'cuda').eval(); sources[name]=source; states[name]=saved
    source=STUDY/'baselines/missingness_aware_gru/best.pt'; saved=load_checkpoint(source)
    recurrent=RecurrentImputer(34,state['config']['hidden']).cuda().eval(); recurrent.load_state_dict(saved['weights'])
    models['recurrent']=recurrent; sources['recurrent']=source; states['recurrent']=saved
    totals={name:0. for name in POLICY['candidates']}; count=0; digest=hashlib.sha256()
    rng=torch.Generator(device='cuda').manual_seed(POLICY['mask_seed'])
    with torch.no_grad():
        for batch_index,(x,m,d,v,_) in enumerate(loader):
            x,m,d,v=[a.cuda() for a in (x,m,d,v)]
            visible,held=artificial_mask(m,v,.2,generator=rng); dd=recompute_delta(visible,d)
            digest.update(held.cpu().numpy().tobytes()); count+=int(held.sum())
            for name in POLICY['candidates']:
                if name in ['mean','forward']: output=simple(x,visible,name)
                elif name=='recurrent':
                    with torch.autocast('cuda',dtype=torch.bfloat16): output=recurrent(x*visible,visible,dd,v).float()
                else: output=draws(models[name],x,visible,dd,v,samples=POLICY['samples'],seed=POLICY['noise_seed']+batch_index).mean(1)
                assert torch.isfinite(output).all()
                totals[name]+=float(((output-x).abs()*held).sum())
    assert count>0
    scores={name:value/count for name,value in totals.items()}; winner=min(POLICY['candidates'],key=lambda name:scores[name])
    target=STUDY/'deployment'/'models'/'selected.pt'; target.parent.mkdir(parents=True,exist_ok=True)
    assert not target.exists(),'Do not overwrite an existing deployment checkpoint'
    if winner in ['mean','forward']:
        packaged={'version':'ChronoGAN-research-'+winner,'model_kind':winner,'stats':state['stats'],'config':state['config'],'splits':state['splits']}
    elif winner=='recurrent':
        packaged={**states[winner],'version':'ChronoGAN-research-missingness-aware-GRU','model_kind':'recurrent','stats':state['stats'],'splits':state['splits']}
    else:
        packaged={**states[winner],'model_kind':'generator','deployment_variant':winner}
    packaged['config']={**packaged['config'],'deployment_windowed':True}
    atomic_save(target,packaged)
    result={'selected_method':winner,'checkpoint':str(target.relative_to(STUDY)),'checkpoint_sha256':hashfile(target),
            'validation_normalized_mae':scores[winner],'validation_scores':scores,'mask_sha256':digest.hexdigest(),
            'validation_scored_cells':count,'test_used_for_selection':False,'policy':POLICY,
            'source_checkpoint':str(sources[winner]) if winner in sources else 'No learned weights: frozen training normalization and causal rule.',
            'stochastic_uncertainty_available':winner not in ['mean','forward','recurrent']}
    atomic_json(path,result)
    print('Validation-only deployment selection',json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--freeze-only',action='store_true');args=parser.parse_args()
    if args.freeze_only: print(freeze_policy())
    else: print(json.dumps(select(),indent=2))
