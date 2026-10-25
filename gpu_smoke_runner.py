"""Launch an explicitly CUDA run and monitor NVIDIA utilization independently."""
import csv
import hashlib
import json
import subprocess
import sys
import time
from download_dataset import ROOT

destination=ROOT/'results'/'gpu'; destination.mkdir(exist_ok=True)
preserved=[ROOT/'models'/'chronogan_best.pt',ROOT/'checkpoints'/'last.pt']
hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in preserved if path.exists()}
command=[sys.executable,str(ROOT/'run.py'),'train.py','--config','config.gpu-smoke.yaml','--device','cuda']
with (destination/'training.log').open('w') as log, (destination/'nvidia-monitor.csv').open('w',newline='') as monitor:
    writer=csv.writer(monitor); writer.writerow(['elapsed_seconds','gpu_utilization_percent','memory_used_mib','memory_free_mib'])
    started=time.perf_counter()
    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
    while process.poll() is None:
        sample=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
        if sample.returncode==0:
            writer.writerow([round(time.perf_counter()-started,3),*sample.stdout.strip().split(', ')]); monitor.flush()
        time.sleep(.5)
    returncode=process.wait()
for path in preserved:
    key=str(path.relative_to(ROOT))
    if path.exists() and key in hashes:
        assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[key],f'CPU checkpoint changed: {path}'
(destination/'cpu-checkpoint-hashes.json').write_text(json.dumps(hashes,indent=2))
print((destination/'training.log').read_text())
if returncode: raise SystemExit(returncode)
print('GPU smoke completed; original CPU checkpoint hashes unchanged.')
