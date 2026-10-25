"""BRITS reconstruction-only adaptation with variable-length padding support.

Reference: Cao et al., NeurIPS 2018, https://arxiv.org/abs/1805.10572
Independent implementation: bidirectional RITS, feature/history estimates,
learned temporal decay and consistency. No supervised classification objective.
"""
import torch
from torch import nn
from torch.nn import functional as F
from utils.dataset import recompute_delta


class RITS(nn.Module):
    def __init__(self, features, hidden):
        super().__init__()
        self.hidden = hidden
        self.hidden_decay = nn.Linear(features, hidden)
        self.input_decay = nn.Parameter(torch.empty(features).uniform_(-features**-.5,features**-.5))
        self.input_bias = nn.Parameter(torch.empty(features).uniform_(-features**-.5,features**-.5))
        self.history = nn.Linear(hidden, features)
        self.feature = nn.Linear(features, features)
        self.register_buffer('off_diagonal', 1-torch.eye(features))
        self.combine = nn.Linear(features*2, features)
        self.cell = nn.LSTMCell(features*2, hidden)

    def forward(self, x, mask, delta, valid):
        h = x.new_zeros(len(x), self.hidden)
        cell = torch.zeros_like(h)
        estimates = []
        numerator = x.new_zeros(())
        denominator = (mask*valid[..., None]).sum().clamp_min(1)
        for t in range(x.shape[1]):
            active = valid[:, t, None]
            observed = mask[:, t]*active
            decayed = h*torch.exp(-F.relu(self.hidden_decay(delta[:, t])))
            history = self.history(decayed)
            completed = observed*x[:, t]+(1-observed)*history
            cross = F.linear(completed, self.feature.weight*self.off_diagonal, self.feature.bias)
            gamma = torch.exp(-F.relu(delta[:, t]*self.input_decay+self.input_bias))
            # BRITS uses an unconstrained learned linear combination.
            weight = self.combine(torch.cat([gamma, observed], -1))
            estimate = weight*cross+(1-weight)*history
            for prediction in (history, cross, estimate):
                numerator = numerator+((prediction.float()-x[:, t]).abs()*observed).sum()
            output = observed*x[:, t]+(1-observed)*estimate
            candidate, next_cell = self.cell(torch.cat([output, observed], -1), (decayed, cell))
            h = active*candidate+(1-active)*h
            cell = active*next_cell+(1-active)*cell
            estimates.append(output*active)
        return torch.stack(estimates, 1), numerator/denominator


class BRITS(nn.Module):
    def __init__(self, features, hidden=64):
        super().__init__()
        self.forward_rits = RITS(features, hidden)
        self.backward_rits = RITS(features, hidden)

    def forward(self, x, mask, delta, valid, return_loss=False):
        # Reverse each valid prefix, leaving padding at the end in both directions.
        lengths = valid.sum(1).long()
        positions = torch.arange(x.shape[1], device=x.device)[None].expand(len(x), -1)
        indices = torch.where(positions < lengths[:, None], lengths[:, None]-1-positions, positions)
        def reverse(value):
            return value.gather(1, indices[..., None].expand_as(value))
        forward, loss_f = self.forward_rits(x*mask, mask, delta, valid)
        backwards_mask = reverse(mask)
        backward, loss_b = self.backward_rits(reverse(x*mask), backwards_mask,
                                            recompute_delta(backwards_mask, delta), valid)
        backward = reverse(backward)
        output = mask*x+(1-mask)*(forward+backward)/2
        consistency = ((forward.float()-backward.float()).abs()*valid[..., None]).sum()
        consistency = consistency/(valid.sum()*x.shape[-1]).clamp_min(1)
        loss = loss_f+loss_b+0.1*consistency
        return (output, loss) if return_loss else output
