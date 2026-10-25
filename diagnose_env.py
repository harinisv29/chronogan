"""Read-only environment audit with sanitized pip configuration."""
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import torch

ROOT=Path(__file__).resolve().parent
def command(args):
    result=subprocess.run(args,capture_output=True,text=True,timeout=45)
    output=result.stdout+result.stderr
    return re.sub(r'(https?://)[^\s/@]+:[^\s/@]+@',r'\1[redacted]@',output).strip()

candidates={Path(sys.executable)}
home=Path.home()
for base,pattern in [(home/'AppData/Local/Programs','Python*/python.exe'),(Path('C:/Program Files'),'Python*/python.exe'),(home,'.conda/envs/*/python.exe'),(home,'miniconda3/python.exe'),(home,'anaconda3/python.exe'),(ROOT.parent,'*/.venv/Scripts/python.exe'),(ROOT.parent,'*/.tools/python/python.exe')]:
    if base.exists(): candidates.update(base.glob(pattern))
for base,pattern in [(Path('C:/'),'Python*/python.exe'),(home,'miniconda3/envs/*/python.exe'),(home,'anaconda3/envs/*/python.exe'),(home/'AppData/Roaming/uv/python','*/python.exe'),(Path('C:/ProgramData'),'miniconda3/python.exe')]:
    if base.exists(): candidates.update(base.glob(pattern))
environments=[]
for python in sorted(candidates):
    try:
        output=command([str(python),'-c','import sys,importlib.util,json; s=importlib.util.find_spec("torch"); print(json.dumps({"python":sys.version,"torch_origin":s.origin if s else None})); exec("import torch; print(torch.__version__, torch.cuda.is_available())") if s else None'])
        environments.append({'executable':str(python),'probe':output})
    except (OSError,subprocess.TimeoutExpired) as exc: environments.append({'executable':str(python),'error':str(exc)})
disk=shutil.disk_usage(ROOT)
out={'project':str(ROOT),'executable':sys.executable,'python':sys.version,'prefix':sys.prefix,'architecture':platform.machine(),'torch':str(torch.__version__),'torch_origin':torch.__file__,'cuda_available':torch.cuda.is_available(),'cuda_runtime':torch.version.cuda,'gpu':command(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.free,memory.used,utilization.gpu','--format=csv,noheader']),'pip':command([sys.executable,'-m','pip','--version']),'pip_config':command([sys.executable,'-m','pip','config','list']),'disk_gib':{k:round(v/1024**3,2) for k,v in zip(['total','used','free'],disk)},'other_environments':environments,'note':'ChronoGAN currently shares the SugarSense embedded Python executable but loads packages from chronogan/.packages. No packages have been installed into SugarSense.'}
(ROOT/'results'/'gpu').mkdir(exist_ok=True)
(ROOT/'results'/'gpu'/'environment-before.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
