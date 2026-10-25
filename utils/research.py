"""Frozen study protocols. No outcome labels enter reconstruction or test masking."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from download_dataset import ROOT
from utils.cache import windows
from utils.dataset import artificial_mask,recompute_delta

STUDY=ROOT/'runs'/'full-training'
SCENARIOS=[(p,r) for p in ['random','block'] for r in [.1,.2,.4,.6]]+[('variable_specific',.2),('laboratory_panel_block',.2)]

def cached_data(state,split):
    return windows(state['splits'][split],state['stats'],state['config']['sequence_length'],STUDY/'cache',split)

def hide(m,v,pattern,rate,rng):
    if pattern in ['random','block']: return artificial_mask(m,v,rate,pattern=='block',rng)
    if pattern=='variable_specific':
        # Explicit, prespecified variable-dependent stress protocol; not estimated from outcomes.
        rates=torch.full((m.shape[-1],),.4,device=m.device); rates[:8]=.1
        selected=torch.rand(m.shape,device=m.device,generator=rng)<rates
    elif pattern=='laboratory_panel_block':
        length=m.shape[1]; span=max(1,int(length*rate))
        starts=torch.randint(length-span+1,(len(m),1,1),device=m.device,generator=rng)
        t=torch.arange(length,device=m.device)[None,:,None]
        lab=(torch.arange(m.shape[-1],device=m.device)>=8)[None,None,:]
        selected=(t>=starts)&(t<starts+span)&lab
    else: raise ValueError(pattern)
    held=m*selected*v[...,None]
    return m-held,held

def simple(x,m,method):
    if method=='mean': return x*m
    batch,length,features=x.shape
    t=torch.arange(length,device=x.device)[None,:,None].expand_as(x)
    before=torch.where(m.bool(),t,torch.full_like(t,-1)).cummax(1).values
    previous=x.gather(1,before.clamp(min=0)); previous=torch.where(before>=0,previous,0.)
    if method=='forward': return previous
    if method!='linear_retrospective': raise ValueError(method)
    after=torch.where(m.bool(),t,torch.full_like(t,length)).flip(1).cummin(1).values.flip(1)
    following=x.gather(1,after.clamp(max=length-1))
    alpha=(t-before)/(after-before).clamp(min=1)
    value=previous+(following-previous)*alpha
    value=torch.where(before<0,following,value)
    value=torch.where(after>=length,previous,value)
    return torch.where((before<0)&(after>=length),0.,value)

def draws(model,x,m,d,v,samples=20,seed=901):
    # Vectorize independent MC draws to avoid serial encoder launches; inference only.
    rng=torch.Generator(device=x.device).manual_seed(seed)
    repeat=lambda a:a.repeat_interleave(samples,dim=0)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        z=torch.randn(len(x)*samples,x.shape[1],model.noise,device=x.device,generator=rng)
        result=model(repeat(x*m),repeat(m),repeat(d),repeat(v),z)
    return result.float().reshape(len(x),samples,*x.shape[1:])

def hashfile(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
