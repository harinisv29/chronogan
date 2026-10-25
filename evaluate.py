"""Real held-out masking experiments and separate causal downstream models."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from download_dataset import ROOT
from preprocess import PatientDataset,FEATURES
from models.generator import Generator
from models.sepsis_predictor import SepsisPredictor
from utils.baselines import RecurrentImputer,simple_impute
from utils.dataset import artificial_mask,recompute_delta
from utils.losses import masked_mean
from utils.metrics import classification_metrics
from train import load_checkpoint

def evaluate(checkpoint,epochs=10,output_dir=None):
    output=Path(output_dir) if output_dir is not None else ROOT
    for folder in ['models','results']: (output/folder).mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(min(4,torch.get_num_threads())); torch.manual_seed(42)
    state=load_checkpoint(checkpoint,device); cfg=state['config']; features=len(FEATURES)
    checkpoint_hash=hashlib.sha256(open(checkpoint,'rb').read()).hexdigest()
    loaders={k:DataLoader(PatientDataset(v,state['stats'],cfg['sequence_length']),batch_size=cfg['batch_size'],shuffle=k=='train') for k,v in state['splits'].items()}
    g=Generator(features,cfg['hidden'],cfg['noise']).to(device); g.load_state_dict(state['generator']); g.eval()
    recurrent=RecurrentImputer(features,cfg['hidden']).to(device)
    opt=torch.optim.Adam(recurrent.parameters(),lr=cfg['learning_rate'])
    best=float('inf'); best_weights=None
    for epoch in range(epochs):
        recurrent.train()
        for x,m,d,v,_ in loaders['train']:
            x,m,d,v=[a.to(device) for a in (x,m,d,v)]
            visible,held=artificial_mask(m,v,cfg['hide_rate'])
            out=recurrent(x*visible,visible,recompute_delta(visible,d),v)
            loss=masked_mean((out-x)**2,held); opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(recurrent.parameters(),5); opt.step()
        recurrent.eval(); err=0.; n=0.
        rng=torch.Generator(device=device).manual_seed(12345)
        with torch.no_grad():
            for x,m,d,v,_ in loaders['val']:
                x,m,d,v=[a.to(device) for a in (x,m,d,v)]
                visible,held=artificial_mask(m,v,cfg['hide_rate'],generator=rng)
                out=recurrent(x*visible,visible,recompute_delta(visible,d),v)
                err+=((out-x).abs()*held).sum().item(); n+=held.sum().item()
        if err/max(n,1)<best:
            best=err/max(n,1); best_weights={k:v.detach().clone() for k,v in recurrent.state_dict().items()}
    recurrent.load_state_dict(best_weights); recurrent.eval()
    torch.save({'weights':recurrent.state_dict(),'hidden':cfg['hidden']},output/'models'/'recurrent_baseline.pt')
    methods=['mean','forward','linear_retrospective','recurrent','chronogan']
    def impute(method,x,m,d,v):
        if method=='chronogan': return torch.stack([g(x*m,m,d,v) for _ in range(10)]).mean(0)
        if method=='recurrent': return recurrent(x*m,m,d,v)
        return simple_impute(x,m,method)
    rows=[]
    for rate in [.1,.2,.4,.6]:
        for pattern in ['random','block']:
            sums={method:np.zeros((3,features)) for method in methods}
            rng=torch.Generator(device=device).manual_seed(2026)
            with torch.no_grad():
                for x,m,d,v,_ in loaders['test']:
                    x,m,d,v=[a.to(device) for a in (x,m,d,v)]
                    visible,held=artificial_mask(m,v,rate,pattern=='block',rng); dd=recompute_delta(visible,d)
                    for method in methods:
                        error=(impute(method,x,visible,dd,v)-x)*held
                        sums[method][0]+=error.abs().sum((0,1)).cpu().numpy()
                        sums[method][1]+=(error**2).sum((0,1)).cpu().numpy()
                        sums[method][2]+=held.sum((0,1)).cpu().numpy()
            scale=np.array(state['stats']['std'])
            for method,(abs_sum,sq_sum,count) in sums.items():
                per={f:dict(mae=float(abs_sum[j]/count[j]*scale[j]),rmse=float(np.sqrt(sq_sum[j]/count[j])*scale[j]),n=int(count[j])) if count[j] else None for j,f in enumerate(FEATURES)}
                rows.append(dict(method=method,rate=rate,pattern=pattern,normalized_mae=float(abs_sum.sum()/max(count.sum(),1)),normalized_rmse=float(np.sqrt(sq_sum.sum()/max(count.sum(),1))),per_variable=per,held_out_cells=int(count.sum())))
    (output/'results'/'evaluation.json').write_text(json.dumps({'checkpoint_sha256':checkpoint_hash,'baseline_epochs':epochs,'experiments':rows,'patients':len(state['splits']['test']),'protocol':'Same held-out real observations and patient splits; native-unit errors per variable, standardized aggregate errors. Block masks hide contiguous time spans per feature.'},indent=2))
    from utils.visualization import comparison_plot
    comparison_plot(rows,output/'results')
    # Linear interpolation excluded: it reads future observations.
    downstream={}
    for method in ['mean','forward','recurrent','chronogan','direct']:
        model=SepsisPredictor(features,cfg['hidden'],direct=method=='direct').to(device)
        opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
        best=float('inf'); best_weights=None
        # Cache causal reconstructed records once; never expose SepsisLabel to imputers.
        cache={}
        with torch.no_grad():
            for split,loader in loaders.items():
                cache[split]=[]
                for x,m,d,v,y in loader:
                    x,m,d,v,y=[a.to(device) for a in (x,m,d,v,y)]
                    filled=x if method=='direct' else impute(method,x,m,d,v)
                    cache[split].append((filled,m,d,v,y))
        for epoch in range(epochs):
            model.train()
            for x,m,d,v,y in cache['train']:
                valid=v.bool()&torch.isfinite(y)
                if not valid.any(): continue
                loss=F.binary_cross_entropy_with_logits(model(x,m,d,v)[valid],y[valid])
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
            model.eval(); total=0.; n=0
            with torch.no_grad():
                for x,m,d,v,y in cache['val']:
                    valid=v.bool()&torch.isfinite(y)
                    if valid.any():
                        total+=F.binary_cross_entropy_with_logits(model(x,m,d,v)[valid],y[valid],reduction='sum').item(); n+=int(valid.sum())
            if n and total/n<best:
                best=total/n; best_weights={k:v.detach().clone() for k,v in model.state_dict().items()}
        if best_weights is None:
            downstream[method]={'unavailable':'No validation labels'}; continue
        model.load_state_dict(best_weights); model.eval(); ys=[]; ps=[]
        with torch.no_grad():
            for x,m,d,v,y in cache['test']:
                valid=v.bool()&torch.isfinite(y)
                ys.extend(y[valid].cpu().tolist()); ps.extend(model(x,m,d,v)[valid].sigmoid().cpu().tolist())
        downstream[method]=classification_metrics(ys,ps) if ys else {'unavailable':'No test labels'}
        torch.save({'weights':model.state_dict(),'method':method,'hidden':cfg['hidden']},output/'models'/f'sepsis_{method}.pt')
        print(method,downstream[method],flush=True)
    (output/'results'/'sepsis.json').write_text(json.dumps({'checkpoint_sha256':checkpoint_hash,'methods':downstream,'task':'Provided challenge SepsisLabel at each hour (already shifted six hours before onset); patient-disjoint, causal, hourly pooled metrics','epochs':epochs,'improvement_claim':'No superiority claim; inspect results and use repeated seeds and patient-bootstrap intervals for research conclusions.'},indent=2))
    from utils.visualization import calibration_plot
    calibration_plot(downstream,output/'results')
    print(f'Saved actual held-out experiments to {output / "results"}',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=str,default=str(ROOT/'models'/'chronogan_best.pt')); p.add_argument('--baseline-epochs',type=int,default=10)
    p.add_argument('--output-dir',default=str(ROOT))
    args=p.parse_args()
    if args.baseline_epochs<1: p.error('baseline epochs must be positive')
    evaluate(args.checkpoint,args.baseline_epochs,args.output_dir)
