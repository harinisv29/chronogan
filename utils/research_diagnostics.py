"""Secondary reconstruction diagnostics; no fitting, labels or mask changes."""
import torch


def temporal_errors(output,truth,observed,held,valid):
    pairs=observed[:,1:]*observed[:,:-1]*valid[:,1:,None]*valid[:,:-1,None]
    pairs=pairs*((held[:,1:]+held[:,:-1])>0)
    difference=(output[:,1:]-output[:,:-1])-(truth[:,1:]-truth[:,:-1])
    return (difference.abs()*pairs).sum((0,1)),pairs.sum((0,1))


def stochastic_variation(samples,held):
    deviation=samples.float().std(dim=1,unbiased=False)
    return (deviation*held).sum((0,1)),((deviation<1e-6)*held).sum((0,1))
