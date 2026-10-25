import torch
from torch.nn import functional as F

def masked_mean(value,mask):
    return (value*mask).sum()/mask.sum().clamp(min=1)

def prepare_physiological_bounds(stats,device,dtype=torch.float32):
    from preprocess import FEATURES
    bounds={'HR':(0,350),'O2Sat':(0,100),'Temp':(15,50),'SBP':(0,400),'MAP':(0,350),'DBP':(0,300),'Resp':(0,120),'pH':(6,9),'Glucose':(0,2000),'Creatinine':(0,50),'WBC':(0,500),'Platelets':(0,2500)}
    indices=torch.tensor([FEATURES.index(name) for name in bounds],device=device)
    mean=torch.tensor(stats['mean'],device=device,dtype=dtype)[indices]
    std=torch.tensor(stats['std'],device=device,dtype=dtype)[indices]
    lo=torch.tensor([value[0] for value in bounds.values()],device=device,dtype=dtype)
    hi=torch.tensor([value[1] for value in bounds.values()],device=device,dtype=dtype)
    return indices,(lo-mean)/std,(hi-mean)/std


def generator_loss(out,x,held,m,valid,logits,unknown,config,stats=None,return_components=False,prepared_bounds=None):
    adv=masked_mean(F.binary_cross_entropy_with_logits(logits,torch.ones_like(logits),reduction='none'),(1-m)*unknown*valid[...,None])
    rec=masked_mean((out-x)**2,held)
    pairs=valid[:,1:,None]*valid[:,:-1,None]*((1-m[:,1:])+(1-m[:,:-1])).clamp(max=1)
    smooth=masked_mean((out[:,1:]-out[:,:-1]).abs(),pairs.expand_as(out[:,1:]))
    # Broad training-normalized soft tails, never clamp real measurements.
    plausible=masked_mean(F.relu(out.abs()-8)**2,(1-m)*valid[...,None])
    if prepared_bounds is not None:
        indices,lower,upper=prepared_bounds
        selected=out[...,indices]
        penalty=torch.zeros_like(out)
        penalty[...,indices]=F.relu(lower-selected)**2+F.relu(selected-upper)**2
        plausible=plausible+masked_mean(penalty,(1-m)*valid[...,None])
    elif stats is not None:
        from preprocess import FEATURES
        mean=out.new_tensor(stats['mean']); std=out.new_tensor(stats['std'])
        # Broad, soft physiological bounds in native units. No hard clipping.
        bounds={'HR':(0,350),'O2Sat':(0,100),'Temp':(15,50),'SBP':(0,400),'MAP':(0,350),'DBP':(0,300),'Resp':(0,120),'pH':(6,9),'Glucose':(0,2000),'Creatinine':(0,50),'WBC':(0,500),'Platelets':(0,2500)}
        penalty=torch.zeros_like(out)
        for name,(lo,hi) in bounds.items():
            j=FEATURES.index(name)
            penalty[...,j]=F.relu((lo-mean[j])/std[j]-out[...,j])**2+F.relu(out[...,j]-(hi-mean[j])/std[j])**2
        plausible=plausible+masked_mean(penalty,(1-m)*valid[...,None])
    total=config['adversarial']*adv+config['reconstruction']*rec+config['temporal']*smooth+config['plausibility']*plausible
    return (total,{'adversarial_loss':adv,'reconstruction_mse':rec,'temporal_loss':smooth,'plausibility_loss':plausible}) if return_components else total
