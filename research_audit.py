"""Read-only checkpoint/history audit; safe while the atomic trainer is running."""
import json
import math
from pathlib import Path
import torch
from models.brits import BRITS
from train import load_checkpoint
from utils.dataset import recompute_delta
from utils.experiment import atomic_json
from utils.research import STUDY,hashfile,ROOT


def audit():
    history=json.loads((STUDY/'results'/'training.json').read_text())
    assert history and all(math.isfinite(value) for row in history for value in row.values() if isinstance(value,(float,int)))
    best=load_checkpoint(STUDY/'models'/'chronogan_best.pt')
    last=load_checkpoint(STUDY/'checkpoints'/'last.pt')
    assert best['best']==min(r['validation_normalized_mae'] for r in best['history'])
    assert last['epoch']+1==len(last['history'])
    assert last['best']==best['best']==min(r['validation_normalized_mae'] for r in last['history'])
    assert last['training_device']=='cuda'
    assert all(torch.isfinite(tensor).all() for name in ['generator','discriminator'] for tensor in last[name].values())
    for name in ['generator_optimizer','discriminator_optimizer','rng','cuda_rng','stale']:
        assert name in last
    prior=json.loads((STUDY/'readiness.json').read_text())['preserved_files']
    assert all(hashfile(ROOT/name)==expected for name,expected in prior.items())
    model=BRITS(34,64).cuda()
    torch.manual_seed(42)
    x=torch.randn(4,24,34,device='cuda'); m=(torch.rand_like(x)>.5).float(); valid=torch.ones(4,24,device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        output,loss=model(x,m,recompute_delta(m,m),valid,return_loss=True)
    assert torch.isfinite(loss) and torch.isfinite(output).all()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    report={'python_version':__import__('platform').python_version(),'torch_version':torch.__version__,
            'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
            'gpu_total_mib':torch.cuda.get_device_properties(0).total_memory/2**20,
            'frozen_split_sha256':hashfile(ROOT/'data'/'full'/'splits.json'),
            'frozen_statistics_sha256':hashfile(ROOT/'data'/'full'/'statistics.json'),
            'completed_epochs_at_audit':len(history),'best_epoch':best['epoch']+1,'best_validation_mae':best['best'],
            'latest_losses':history[-1],'checkpoint_weights_finite':True,'resumable_checkpoint_fields_present':True,
            'prior_artifacts_preserved':True,'brits_cuda_bf16_backward_passed':True,
            'interpretation':'No nonfinite losses/weights detected. Validation fluctuates; adversarial balance is diagnostic, not proof of stability or clinical utility.'}
    atomic_json(STUDY/'results'/'training-audit.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__': audit()
