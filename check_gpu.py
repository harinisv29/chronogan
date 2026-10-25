"""Require real CUDA computation, backpropagation, and allocator release."""
import gc
import json
import sys
import torch
from download_dataset import ROOT

if not torch.cuda.is_available():
    raise SystemExit(f'CUDA is not available in PyTorch {torch.__version__}')
torch.cuda.init(); torch.cuda.empty_cache()
# Initialize persistent BLAS workspaces before measuring releasable tensor memory.
warm=torch.ones(16,16,device='cuda',requires_grad=True)
(warm@warm.T).sum().backward()
del warm
gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
baseline=torch.cuda.memory_allocated()
torch.cuda.reset_peak_memory_stats()
x=torch.randn(1024,1024,device='cuda',requires_grad=True)
y=x@x.T
loss=y.square().mean(); loss.backward()
torch.cuda.synchronize()
assert x.device.type=='cuda' and y.device.type=='cuda'
assert x.grad is not None and torch.isfinite(x.grad).all().item()
allocated=torch.cuda.memory_allocated(); peak=torch.cuda.max_memory_allocated()
assert allocated>baseline
del x,y,loss
gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
released=torch.cuda.memory_allocated()
assert released<=baseline, f'GPU tensor memory was not released: {released} > {baseline}'
report={'python':sys.version,'executable':sys.executable,'torch':str(torch.__version__),'torch_origin':torch.__file__,'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'vram_mib':torch.cuda.get_device_properties(0).total_memory/1024**2,'cuda_available':True,'tensor_device':'cuda:0','matrix_multiplication_succeeded':True,'backward_pass_succeeded':True,'allocation_release_succeeded':True,'baseline_allocated_mib':baseline/1024**2,'allocated_mib':allocated/1024**2,'peak_allocated_mib':peak/1024**2,'released_allocated_mib':released/1024**2}
destination=ROOT/'results'/'gpu'; destination.mkdir(exist_ok=True)
(destination/'cuda-verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
