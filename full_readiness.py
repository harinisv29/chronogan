"""Read-only cohort/checkpoint checks plus a real CUDA/BF16 backward test."""
import hashlib
import json
import shutil
import torch
from download_dataset import ROOT,DATA
from download_full_dataset import valid
from preprocess import PatientDataset,FEATURES
from models.generator import Generator
from models.discriminator import Discriminator
from utils.dataset import artificial_mask,recompute_delta

out=ROOT/'runs'/'full-training'; out.mkdir(parents=True,exist_ok=True)
report_path=out/'readiness.json'
if report_path.exists():
    raise SystemExit('Readiness record already exists; inspect it before replacing provenance')
manifest=json.loads((DATA/'official-s3-manifest.json').read_text())
assert len(manifest)==40336
for i,row in enumerate(manifest,1):
    assert valid(DATA/row['relative'],row),row['relative']
    if i%10000==0: print('Intact patients:',i,flush=True)
splits=json.loads((DATA/'full'/'splits.json').read_text())
groups=[set(splits[k]) for k in ['train','val','test']]
assert [len(g) for g in groups]==[28235,6050,6051]
assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
assert set.union(*groups)=={str((DATA/r['relative']).resolve()) for r in manifest}
preserved={}
for folder in [ROOT/'models',ROOT/'checkpoints',ROOT/'results',ROOT/'runs'/'gpu-smoke',ROOT/'runs'/'gpu-amp']:
    for path in folder.rglob('*'):
        if path.is_file() and path.suffix in ['.pt','.json','.png','.md','.csv'] and 'pytest-tmp' not in path.parts and 'gpu' not in path.relative_to(ROOT/'results').parts if folder==ROOT/'results' else path.is_file() and path.suffix in ['.pt','.json','.png','.md','.csv']:
            preserved[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
stats=json.loads((DATA/'full'/'statistics.json').read_text())
sample=PatientDataset(splits['train'][:4],stats,24)
x,m,d,v,_=[torch.stack([sample[i][j] for i in range(min(4,len(sample)))]).cuda() for j in range(5)]
visible,held=artificial_mask(m,v,.2); g=Generator(len(FEATURES),64,16).cuda(); disc=Discriminator(len(FEATURES),64).cuda()
with torch.autocast('cuda',dtype=torch.bfloat16):
    result=g(x*visible,visible,recompute_delta(visible,d),v)
    logits=disc(result,torch.full_like(m,.5),v)
loss=((result-x).square()*held).sum()/held.sum().clamp(min=1)+logits.float().square().mean()
loss.backward(); torch.cuda.synchronize()
assert torch.isfinite(loss) and torch.equal(result[visible.bool()],x[visible.bool()])
assert all(p.grad is None or torch.isfinite(p.grad).all() for p in list(g.parameters())+list(disc.parameters()))
free,total=torch.cuda.mem_get_info(); disk=shutil.disk_usage(ROOT)
assert free>1024**3 and disk.free>10*1024**3
report={'verified_patients':40336,'splits':{k:len(v) for k,v in splits.items()},'cuda':torch.version.cuda,'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name(0),'gpu_free_mib':free/1024**2,'disk_free_gib':disk.free/1024**3,'bf16_forward_backward_finite':True,'observations_preserved':True,'preserved_files':preserved,'split_sha256':hashlib.sha256((DATA/'full'/'splits.json').read_bytes()).hexdigest(),'statistics_sha256':hashlib.sha256((DATA/'full'/'statistics.json').read_bytes()).hexdigest()}
report_path.write_text(json.dumps(report,indent=2)); print({k:v for k,v in report.items() if k!='preserved_files'})
