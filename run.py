"""Launcher supporting a regular Python install or the workspace's embedded runtime."""
import os
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'.packages'))
if os.environ.get('CHRONOGAN_CPU_ONLY')!='1' and (ROOT/'.cuda-packages'/'torch').is_dir():
    sys.path.insert(0,str(ROOT/'.cuda-packages'))
os.chdir(ROOT)
if len(sys.argv)<2:
    raise SystemExit('Usage: python run.py download_dataset.py|train.py|evaluate.py|app.py [arguments]')
target=sys.argv.pop(1)
runpy.run_path(str(ROOT/target),run_name='__main__')
