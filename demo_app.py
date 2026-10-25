"""CPU-only interactive demo; independent of the ongoing research controller."""
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['CHRONOGAN_RUN_DIR']='runs/full-training'
os.environ['CHRONOGAN_DEMO']='1'
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import hashlib
import json
from pathlib import Path
import socket
import torch
from app import app,active_artifacts
from train import load_checkpoint


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=5003);args=parser.parse_args()
    torch.set_num_threads(2)
    app.config['TEMPLATES_AUTO_RELOAD']=True
    app.jinja_env.auto_reload=True
    assert not torch.cuda.is_available(),'Demo must never use CUDA'
    checkpoint,_=active_artifacts();state=load_checkpoint(checkpoint,'cpu')
    assert 'generator' in state and state['config'].get('use_mask',True) and state['config'].get('use_delta',True)
    with socket.socket() as probe: probe.bind(('127.0.0.1',args.port))
    folder=Path(__file__).resolve().parent/'runs/interactive-demo';folder.mkdir(parents=True,exist_ok=True)
    metadata={'pid':os.getpid(),'url':f'http://127.0.0.1:{args.port}','checkpoint':str(checkpoint),
              'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'device':'cpu',
              'python':__import__('sys').executable,'torch':str(torch.__version__),
              'label':'Full ChronoGAN research prototype demonstration; not the final selected deployment model.'}
    (folder/'server.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2),flush=True)
    app.run(host='127.0.0.1',port=args.port,debug=False,use_reloader=False,threaded=True)
