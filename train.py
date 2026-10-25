import argparse
import json
import random
import time
import numpy as np
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader
from download_dataset import ROOT
from preprocess import prepare,PatientDataset,FEATURES
from models.generator import Generator
from models.discriminator import Discriminator
from utils.dataset import artificial_mask,recompute_delta
from utils.losses import masked_mean,generator_loss,prepare_physiological_bounds
from utils.cache import windows
from utils.experiment import atomic_json,atomic_save
from utils.performance import BatchedWindows,collate_cached_batch,DeferredMetrics

def load_checkpoint(path,device='cpu'):
    return torch.load(path,map_location=device,weights_only=True)

def train(config,resume=None):
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    torch.set_num_threads(min(4,torch.get_num_threads()))
    requested=config.get('device','auto')
    if requested=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA training requested, but CUDA-enabled PyTorch and a compatible GPU are not available')
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if requested=='auto' else requested)
    device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU'
    print(f'Training device: {device} ({device_name}); PyTorch {torch.__version__}',flush=True)
    old=load_checkpoint(resume,device) if resume else None
    if old:
        config=old['config']; splits,stats=old['splits'],old['stats']
    elif config.get('cohort_checkpoint'):
        cohort=load_checkpoint(ROOT/config['cohort_checkpoint'])
        splits,stats=cohort['splits'],cohort['stats']
    else: splits,stats=prepare(config)
    output=ROOT/config.get('run_dir','.')
    for folder in ['models','checkpoints','results']:
        (output/folder).mkdir(parents=True,exist_ok=True)
    (output/'results'/'patient-splits.json').write_text(json.dumps(splits,indent=2))
    (output/'results'/'normalization.json').write_text(json.dumps(stats,indent=2))
    if (output/'checkpoints'/'last.pt').exists() and not resume:
        raise FileExistsError('Existing run detected; explicitly resume it instead of overwriting')
    cache_dir=ROOT/config['cache_dir'] if config.get('cache_dir') else None
    optimized=bool(config.get('performance_optimized',False))
    loaders={}
    for k,v in splits.items():
        if k=='test': continue
        dataset=windows(v,stats,config['sequence_length'],cache_dir,k)
        options={}
        if optimized and cache_dir is not None:
            dataset=BatchedWindows(cache_dir/k); options['collate_fn']=collate_cached_batch
        loaders[k]=DataLoader(dataset,batch_size=config['batch_size'],shuffle=k=='train',num_workers=0,pin_memory=device.type=='cuda',**options)
    amp=bool(config.get('amp',False)) and device.type=='cuda'
    amp_dtype=torch.bfloat16 if config.get('amp_dtype','bfloat16')=='bfloat16' else torch.float16
    if amp and amp_dtype==torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 mixed precision requested but not supported by this GPU')
    gscaler=torch.amp.GradScaler('cuda',enabled=amp and amp_dtype==torch.float16)
    dscaler=torch.amp.GradScaler('cuda',enabled=amp and amp_dtype==torch.float16)
    g=Generator(len(FEATURES),config['hidden'],config['noise']).to(device)
    g.use_mask=config.get('use_mask',True); g.use_delta=config.get('use_delta',True)
    d=Discriminator(len(FEATURES),config['hidden']).to(device)
    generator_device=str(next(g.parameters()).device)
    discriminator_device=str(next(d.parameters()).device)
    assert all(p.device.type==device.type for p in g.parameters())
    assert all(p.device.type==device.type for p in d.parameters())
    print(f'Generator: {generator_device}; discriminator: {discriminator_device}; patients: '+str({k:len(v) for k,v in splits.items()}),flush=True)
    go=torch.optim.Adam(g.parameters(),lr=config['learning_rate']); do=torch.optim.Adam(d.parameters(),lr=config['learning_rate'])
    prepared_bounds=prepare_physiological_bounds(stats,device) if optimized else None
    history=[]; best=float('inf'); stale=0; start=0
    if old:
        g.load_state_dict(old['generator']); d.load_state_dict(old['discriminator'])
        go.load_state_dict(old['generator_optimizer']); do.load_state_dict(old['discriminator_optimizer'])
        if 'generator_scaler' in old: gscaler.load_state_dict(old['generator_scaler'])
        if 'discriminator_scaler' in old: dscaler.load_state_dict(old['discriminator_scaler'])
        history=old['history']; best=old['best']; start=old['epoch']+1; stale=old.get('stale',0)
        torch.set_rng_state(old['rng'].cpu())
        if device.type=='cuda' and 'cuda_rng' in old:
            torch.cuda.set_rng_state_all([value.cpu() for value in old['cuda_rng']])
    for epoch in range(start,config['epochs']):
        if device.type=='cuda':
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(device)
        epoch_started=time.perf_counter()
        g.train(); d.train(); totals=[]
        components=[]
        metric_buffer=DeferredMetrics() if optimized else None
        for batch_index,(x,m,delta,valid,_) in enumerate(loaders['train']):
            x,m,delta,valid=[v.to(device,non_blocking=True) for v in (x,m,delta,valid)]
            visible,held=artificial_mask(m,valid,config['hide_rate'])
            xd=x*visible; dd=recompute_delta(visible,delta)
            reveal=(torch.rand_like(m)<.9).float()
            hint=reveal*visible+.5*(1-reveal); unknown=1-reveal
            with torch.autocast(device_type=device.type,dtype=amp_dtype,enabled=amp):
                out=g(xd,visible,dd,valid)
            do.zero_grad()
            with torch.autocast(device_type=device.type,dtype=amp_dtype,enabled=amp):
                logits=d(out.detach(),hint,valid)
            dl=masked_mean(F.binary_cross_entropy_with_logits(logits.float(),visible,reduction='none'),unknown*valid[...,None])
            if not torch.isfinite(dl): raise FloatingPointError('Non-finite discriminator loss')
            dscaler.scale(dl).backward(); dscaler.unscale_(do)
            torch.nn.utils.clip_grad_norm_(d.parameters(),5,error_if_nonfinite=not dscaler.is_enabled()); dscaler.step(do); dscaler.update()
            for p in d.parameters(): p.requires_grad_(False)
            go.zero_grad()
            with torch.autocast(device_type=device.type,dtype=amp_dtype,enabled=amp):
                logits=d(out,hint,valid)
            gl,parts=generator_loss(out,x,held,visible,valid,logits.float(),unknown,config,stats,return_components=True,prepared_bounds=prepared_bounds)
            if not torch.isfinite(gl): raise FloatingPointError('Non-finite generator loss')
            gscaler.scale(gl).backward(); gscaler.unscale_(go)
            torch.nn.utils.clip_grad_norm_(g.parameters(),5,error_if_nonfinite=not gscaler.is_enabled()); gscaler.step(go); gscaler.update()
            for p in d.parameters(): p.requires_grad_(True)
            if optimized: metric_buffer.add(gl,dl,parts)
            else:
                totals.append((gl.item(),dl.item()))
                components.append({key:float(value.detach()) for key,value in parts.items()})
            if (batch_index+1)%100==0:
                progress={'stage':'training','epoch':epoch+1,'completed_epoch':epoch,'batch':batch_index+1,'batches':len(loaders['train'])}
                atomic_json(output/'results'/'progress.json',progress)
                print(progress,flush=True)
        g.eval(); error=0.; count=0.
        rng=torch.Generator(device=device).manual_seed(12345)
        noise_rng=torch.Generator(device=device).manual_seed(54321)
        with torch.no_grad():
            for x,m,delta,valid,_ in loaders['val']:
                x,m,delta,valid=[v.to(device,non_blocking=optimized) for v in (x,m,delta,valid)]
                visible,held=artificial_mask(m,valid,config['hide_rate'],generator=rng)
                z=torch.randn(*x.shape[:2],g.noise,device=device,generator=noise_rng)
                with torch.autocast(device_type=device.type,dtype=amp_dtype,enabled=amp):
                    out=g(x*visible,visible,recompute_delta(visible,delta),valid,z)
                error+=((out-x).abs()*held).sum().item(); count+=held.sum().item()
        if count==0: raise ValueError('No held-out observations in validation split')
        val=error/count
        if not np.isfinite(val): raise FloatingPointError('Non-finite validation error')
        if optimized: row=dict(epoch=epoch+1,validation_normalized_mae=val,**metric_buffer.means())
        else:
            row=dict(epoch=epoch+1,generator=float(np.mean(totals,0)[0]),discriminator=float(np.mean(totals,0)[1]),validation_normalized_mae=val)
            row.update({key:float(np.mean([p[key] for p in components])) for key in components[0]})
        if device.type=='cuda': torch.cuda.synchronize()
        row['epoch_seconds']=time.perf_counter()-epoch_started
        row['peak_cuda_allocated_mib']=torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else 0
        row['peak_cuda_reserved_mib']=torch.cuda.max_memory_reserved(device)/1024**2 if device.type=='cuda' else 0
        history.append(row); print(row,flush=True)
        improved=val<best
        if improved: best=val; stale=0
        else: stale+=1
        state=dict(version='ChronoGAN-1',config=config,stats=stats,splits=splits,generator=g.state_dict(),discriminator=d.state_dict(),generator_optimizer=go.state_dict(),discriminator_optimizer=do.state_dict(),history=history,best=best,epoch=epoch,rng=torch.get_rng_state())
        state['training_device']=str(device)
        state['training_device_name']=device_name
        state['torch_version']=str(torch.__version__)
        state['generator_device']=generator_device
        state['discriminator_device']=discriminator_device
        state['cuda_runtime']=torch.version.cuda
        state['amp_enabled']=amp
        state['stale']=stale
        state['generator_scaler']=gscaler.state_dict()
        state['discriminator_scaler']=dscaler.state_dict()
        if device.type=='cuda': state['cuda_rng']=torch.cuda.get_rng_state_all()
        atomic_save(output/'checkpoints'/'last.pt',state)
        if improved: atomic_save(output/'models'/'chronogan_best.pt',state)
        atomic_json(output/'results'/'training.json',history)
        atomic_json(output/'results'/'progress.json',{'stage':'epoch_completed','completed_epoch':epoch+1,'best_validation_mae':best,'stale_epochs':stale})
        if stale>=config['patience']: break
    from utils.visualization import training_plot
    training_plot(history,output/'results'/'training.png')
    return history

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--config',default='config.yaml'); parser.add_argument('--resume'); parser.add_argument('--epochs',type=int); parser.add_argument('--device',choices=['auto','cpu','cuda'])
    args=parser.parse_args(); config=yaml.safe_load(open(args.config))
    if args.resume:
        config=load_checkpoint(args.resume)['config']
    if args.epochs: config['epochs']=args.epochs
    if args.device: config['device']=args.device
    from utils.performance_gate import before_training
    config=before_training(config,args.resume)
    # Explicit epoch override is also respected when resuming.
    if args.resume:
        state=load_checkpoint(args.resume); state['config']=config
        temporary=ROOT/config.get('run_dir','.')/'checkpoints'/'resume_config.pt'; torch.save(state,temporary)
        train(config,temporary)
    else: train(config)
