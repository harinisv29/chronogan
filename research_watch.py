"""Low-frequency GPU telemetry; exits once the study reaches its terminal marker."""
import csv
import json
import subprocess
import time
from pathlib import Path

root=Path(__file__).resolve().parent/'runs'/'full-training'
path=root/'gpu-monitor.csv'
with path.open('a',newline='') as stream:
    writer=csv.writer(stream)
    if path.stat().st_size==0: writer.writerow(['utc_epoch_seconds','stage','gpu_util_percent','used_mib','free_mib'])
    while not (root/'results'/'study-finished.json').exists():
        status=root/'results'/'study-progress.json'
        stage=json.loads(status.read_text()) if status.exists() else {'stage':'main_gan_training'}
        result=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
        if result.returncode==0:
            writer.writerow([time.time(),json.dumps(stage),*result.stdout.strip().split(', ')]); stream.flush()
        time.sleep(10)
