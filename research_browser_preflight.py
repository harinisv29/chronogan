"""CPU-only staging check; never touches the running research controller."""
import json
import os
import subprocess
import sys
import time
import requests
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    folder=ROOT/'runs/full-training/browser-preflight';(folder/'results').mkdir(parents=True,exist_ok=True)
    environment={**os.environ,'CUDA_VISIBLE_DEVICES':'-1','CHRONOGAN_PORT':'5002','CHRONOGAN_RUN_DIR':'.'}
    with (folder/'server.log').open('w') as log:
        process=subprocess.Popen([sys.executable,str(ROOT/'run.py'),'app.py'],cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        base='http://127.0.0.1:5002'
        for attempt in range(40):
            if process.poll() is not None: raise RuntimeError('Preflight server exited')
            try:
                if requests.get(base+'/api/status',timeout=2).status_code==200:break
            except requests.RequestException:pass
            time.sleep(.5)
        else: raise RuntimeError('Preflight server not ready')
        import research_verify
        research_verify.STUDY=folder
        result=research_verify.browser_check(base)
        result['scope']='CPU-only preflight on preserved original checkpoint; final selected-checkpoint verification is separate and pending.'
        (folder/'result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2))
    finally:
        if process.poll() is None: process.terminate();process.wait(timeout=20)


if __name__=='__main__': main()
