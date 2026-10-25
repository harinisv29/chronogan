import json
from pathlib import Path
import random
import numpy as np
import torch

def atomic_json(path,value):
    path=Path(path); temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)); temp.replace(path)

def atomic_save(path,value):
    path=Path(path); temp=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,temp); temp.replace(path)

def generator_from_state(state,device='cpu'):
    from models.generator import Generator
    config=state['config']
    model=Generator(len(state['stats']['features']),config['hidden'],config['noise']).to(device)
    model.use_mask=config.get('use_mask',True)
    model.use_delta=config.get('use_delta',True)
    model.load_state_dict(state['generator']); return model
