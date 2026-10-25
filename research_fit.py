"""Validation-selected full-cohort baselines and causal sepsis models."""
import gc
import json
import time
import torch
from torch.nn import functional as F
from download_dataset import ROOT
from models.sepsis_predictor import SepsisPredictor
from utils.baselines import RecurrentImputer
from models.brits import BRITS
from utils.dataset import artificial_mask,recompute_delta
from utils.losses import masked_mean
from utils.research import STUDY,cached_data,simple,draws
from utils.experiment import atomic_json,atomic_save,generator_from_state
from utils.metrics import classification_metrics
from train import load_checkpoint

def tensors(state,split):
    source=cached_data(state,split)
    return tuple(torch.from_numpy(a).cuda() for a in source.arrays)

def train_model(kind,state,method=None,max_epochs=50):
    torch.set_num_threads(4); torch.manual_seed(42)
    folder=STUDY/('baselines' if kind in ('imputer','brits') else 'sepsis')/(method or ('brits' if kind=='brits' else 'missingness_aware_gru'))
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists(): return folder
    c=state['config']; f=len(state['stats']['features'])
    model=(BRITS(f,c['hidden']) if kind=='brits' else RecurrentImputer(f,c['hidden']) if kind=='imputer' else SepsisPredictor(f,c['hidden'],matched=True)).cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=c['learning_rate'])
    train=tensors(state,'train'); val=tensors(state,'val')
    generator=None
    if method=='chronogan': generator=generator_from_state(state,'cuda').eval()
    # Impute causally in batches, once. Labels are not passed to reconstruction.
    if kind=='sepsis' and method!='direct':
        for split,data in [('train',train),('val',val)]:
            x,m,d,v,y=data; filled=torch.empty_like(x)
            with torch.no_grad():
                for start in range(0,len(x),c['batch_size']):
                    sl=slice(start,start+c['batch_size'])
                    filled[sl]=draws(generator,x[sl],m[sl],d[sl],v[sl],samples=10,seed=7042+start).mean(1) if method=='chronogan' else simple(x[sl],m[sl],method)
            if split=='train': train=(filled,m,d,v,y)
            else: val=(filled,m,d,v,y)
        del generator; gc.collect(); torch.cuda.empty_cache()
    # Identical optimization shuffles across heads despite different preprocessing.
    torch.manual_seed(42)
    history=[]; best=float('inf'); stale=0; start_epoch=0
    last=folder/'last.pt'
    if last.exists():
        old=load_checkpoint(last,'cuda'); model.load_state_dict(old['weights']); optimizer.load_state_dict(old['optimizer'])
        history=old['history']; best=old['best']; stale=old['stale']; start_epoch=old['epoch']+1
        torch.set_rng_state(old['rng'].cpu()); torch.cuda.set_rng_state_all([s.cpu() for s in old['cuda_rng']])
    batch=c['batch_size']
    for epoch in range(start_epoch,max_epochs):
        if stale>=c['patience']: break
        clock=time.perf_counter(); model.train(); losses=[]
        order=torch.randperm(len(train[0]),device='cuda')
        for offset in range(0,len(order),batch):
            idx=order[offset:offset+batch]; x,m,d,v,y=[a[idx] for a in train]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                if kind in ('imputer','brits'):
                    visible,held=artificial_mask(m,v,c['hide_rate'])
                    if kind=='brits':
                        output,loss=model(x*visible,visible,recompute_delta(visible,d),v,return_loss=True)
                    else:
                        output=model(x*visible,visible,recompute_delta(visible,d),v)
                        loss=masked_mean((output.float()-x)**2,held)
                else:
                    known=v.bool()&torch.isfinite(y)
                    loss=F.binary_cross_entropy_with_logits(model(x,m,d,v).float()[known],y[known])
            if not torch.isfinite(loss): raise FloatingPointError(f'{kind}/{method}: nonfinite training loss')
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True); optimizer.step()
            losses.append(loss.item())
        model.eval(); error=0.; count=0
        rng=torch.Generator(device='cuda').manual_seed(12345)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for offset in range(0,len(val[0]),batch):
                x,m,d,v,y=[a[offset:offset+batch] for a in val]
                if kind in ('imputer','brits'):
                    visible,held=artificial_mask(m,v,c['hide_rate'],generator=rng)
                    output=model(x*visible,visible,recompute_delta(visible,d),v).float()
                    error+=((output-x).abs()*held).sum().item(); count+=int(held.sum())
                else:
                    known=v.bool()&torch.isfinite(y)
                    error+=F.binary_cross_entropy_with_logits(model(x,m,d,v).float()[known],y[known],reduction='sum').item(); count+=int(known.sum())
        metric=error/max(count,1)
        if not torch.isfinite(torch.tensor(metric)): raise FloatingPointError('Nonfinite validation metric')
        improved=metric<best
        if improved: best=metric; stale=0
        else: stale+=1
        torch.cuda.synchronize()
        row={'epoch':epoch+1,'training_loss':sum(losses)/len(losses),'validation_mae' if kind in ('imputer','brits') else 'validation_bce':metric,'seconds':time.perf_counter()-clock}
        history.append(row); print(kind,method,row,flush=True)
        saved={'weights':model.state_dict(),'optimizer':optimizer.state_dict(),'config':c,'history':history,'epoch':epoch,'best':best,'stale':stale,'method':method,'kind':kind,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
        atomic_save(last,saved)
        if improved: atomic_save(folder/'best.pt',saved)
        atomic_json(folder/'history.json',history)
    atomic_json(folder/'complete.json',{'epochs':len(history),'best_validation':best,'selection':'validation MAE' if kind in ('imputer','brits') else 'validation BCE','test_used_for_selection':False})
    return folder

def evaluate_sepsis(state):
    c=state['config']; output={}
    test=tensors(state,'test')
    for method in ['direct','mean','forward','chronogan']:
        folder=STUDY/'sepsis'/method
        if (folder/'test-metrics.json').exists():
            output[method]=json.loads((folder/'test-metrics.json').read_text()); continue
        saved=load_checkpoint(folder/'best.pt','cuda')
        model=SepsisPredictor(len(state['stats']['features']),c['hidden'],matched=True).cuda().eval(); model.load_state_dict(saved['weights'])
        g=generator_from_state(state,'cuda').eval() if method=='chronogan' else None
        ys=[]; ps=[]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for start in range(0,len(test[0]),c['batch_size']):
                x,m,d,v,y=[a[start:start+c['batch_size']] for a in test]
                filled=x if method=='direct' else draws(g,x,m,d,v,samples=10,seed=7042+start).mean(1) if method=='chronogan' else simple(x,m,method)
                known=v.bool()&torch.isfinite(y)
                ps.extend(model(filled,m,d,v).float().sigmoid()[known].cpu().tolist()); ys.extend(y[known].cpu().tolist())
        result=classification_metrics(ys,ps)
        result['test_patients']=len(state['splits']['test']); result['training_epochs']=len(saved['history'])
        atomic_json(folder/'test-metrics.json',result); output[method]=result
        print('Held-out sepsis',method,result,flush=True)
        del model,g; gc.collect(); torch.cuda.empty_cache()
    atomic_json(STUDY/'results'/'sepsis.json',{'methods':output,'task':'Provided six-hour-shifted Challenge labels, no additional shifting; causal 24-hour windows; pooled hourly metrics; fixed 0.5 decision threshold','checkpoint_sha256':__import__('utils.research',fromlist=['hashfile']).hashfile(STUDY/'models'/'chronogan_best.pt'),'improvement_claim':'Measured differences in one frozen split do not establish clinical benefit.'})
    from utils.visualization import calibration_plot
    calibration_plot(output,STUDY/'results')
