"""Single-process official stable CUDA install with retry, cache and SHA-256 checks."""
import hashlib
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
INDEX='https://download.pytorch.org/whl/cu130'
NAME='torch-2.11.0+cu130-cp312-cp312-win_amd64.whl'
SHA256='ef8beae16d781c3244ef28dc7bee6d8871c26bbde65d5bf66e902cb61972c4ab'

def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as source:
        for data in iter(lambda:source.read(4*1024*1024),b''): result.update(data)
    return result.hexdigest()

def main():
    if sys.platform!='win32' or sys.version_info[:2]!=(3,12):
        raise SystemExit('This installer is pinned to Windows x64 Python 3.12')
    cache=ROOT/'.gpu-download'/'cu130'; cache.mkdir(parents=True,exist_ok=True)
    wheel=cache/NAME
    common=['--timeout','180','--retries','10','--resume-retries','10','--cache-dir',str(ROOT/'.pip-cache')]
    if wheel.exists() and digest(wheel)!=SHA256:
        wheel.replace(wheel.with_suffix('.checksum-invalid'))
    if not wheel.exists():
        subprocess.run([sys.executable,'-m','pip','download','torch==2.11.0+cu130','--no-deps','--only-binary=:all:','--dest',str(cache),'--index-url',INDEX,*common],check=True)
    if digest(wheel)!=SHA256: raise SystemExit('CUDA wheel checksum mismatch; refusing installation')
    print('Official CUDA 13.0 wheel SHA-256 verified.',flush=True)
    subprocess.run([sys.executable,'-m','pip','install','--target',str(ROOT/'.cuda-packages'),'--no-deps',str(wheel),'setuptools<82',*common],check=True)
    print('Run: .runtime\\python\\python.exe run.py check_gpu.py')

if __name__=='__main__': main()
