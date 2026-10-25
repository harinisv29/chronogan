"""Patient splits and training-only normalization. All model inputs are causal."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from download_dataset import ROOT, DATA

FEATURES = ['HR','O2Sat','Temp','SBP','MAP','DBP','Resp','EtCO2','BaseExcess','HCO3','FiO2','pH','PaCO2','SaO2','AST','BUN','Alkalinephos','Calcium','Chloride','Creatinine','Bilirubin_direct','Glucose','Lactate','Magnesium','Phosphate','Potassium','Bilirubin_total','TroponinI','Hct','Hgb','PTT','WBC','Fibrinogen','Platelets']

def read_patient(path):
    frame = pd.read_csv(path, sep='|' if str(path).lower().endswith('.psv') else ',')
    if not len(frame) or len(frame)>10000:
        raise ValueError('Patient must contain 1 to 10000 hourly rows')
    if not any(f in frame for f in FEATURES):
        raise ValueError('No supported clinical measurement columns found')
    for col in frame.columns:
        if col in FEATURES + ['ICULOS','SepsisLabel']:
            frame[col] = pd.to_numeric(frame[col],errors='raise')
            if np.isinf(frame[col].to_numpy(dtype=float)).any():
                raise ValueError(f'Infinite value in {col}')
    if 'ICULOS' in frame and (frame.ICULOS.isna().any() or (frame.ICULOS.diff().dropna()<=0).any()):
        raise ValueError('ICULOS must be present, increasing and in hourly order')
    return frame

def delta_from_mask(mask, times=None):
    times = np.arange(len(mask), dtype=float) if times is None else np.asarray(times)
    delta = np.zeros_like(mask,dtype=np.float32)
    for t in range(1,len(mask)):
        delta[t] = times[t]-times[t-1] + (1-mask[t-1])*delta[t-1]
    return delta

def make_splits(paths, seed=42):
    paths = np.array(sorted(str(Path(p).resolve()) for p in paths))
    if len(paths)<10:
        raise ValueError('At least ten real patients required')
    rng = np.random.default_rng(seed)
    rng.shuffle(paths)
    a,b = int(.7*len(paths)),int(.85*len(paths))
    return dict(train=paths[:a].tolist(), val=paths[a:b].tolist(), test=paths[b:].tolist())

def fit_stats(paths,progress=False):
    count = np.zeros(len(FEATURES)); total=count.copy(); squares=count.copy()
    for index,path in enumerate(paths,1):
        x = read_patient(path).reindex(columns=FEATURES).to_numpy(dtype=float)
        count += np.isfinite(x).sum(0)
        total += np.nansum(x,0); squares += np.nansum(x*x,0)
        if progress and index%2000==0: print(f'Normalization: {index}/{len(paths)} training patients',flush=True)
    mean = total / np.maximum(count,1)
    std = np.sqrt(np.maximum(squares/np.maximum(count,1)-mean**2,1e-6))
    std[count==0]=1
    return dict(features=FEATURES, mean=mean.tolist(), std=std.tolist(), count=count.tolist())

def tensors(frame, stats):
    raw = frame.reindex(columns=FEATURES).to_numpy(dtype=np.float32)
    mask = np.isfinite(raw).astype(np.float32)
    x = np.nan_to_num((raw-np.array(stats['mean']))/np.array(stats['std']), nan=0).astype(np.float32)
    times = frame.ICULOS.to_numpy() if 'ICULOS' in frame else np.arange(len(frame))
    return x, mask, delta_from_mask(mask,times)

class PatientDataset(Dataset):
    def __init__(self, paths, stats, length=24):
        self.stats,self.length = stats,length
        self.windows=[]
        for path in paths:
            frame=read_patient(path)
            x,m,d=tensors(frame,stats)
            labels=frame.get('SepsisLabel',pd.Series(np.full(len(frame),np.nan))).to_numpy(dtype=np.float32)
            for start in range(0,len(x),length):
                stop=min(start+length,len(x)); n=stop-start
                # Reset elapsed time at independent window boundary.
                tm=frame.ICULOS.to_numpy()[start:stop] if 'ICULOS' in frame else None
                dd=delta_from_mask(m[start:stop],tm)
                pad=lambda v: np.pad(v,((0,length-n),(0,0)))
                self.windows.append((pad(x[start:stop]),pad(m[start:stop]),pad(dd),np.r_[np.ones(n),np.zeros(length-n)].astype(np.float32),np.pad(labels[start:stop],(0,length-n),constant_values=np.nan)))
    def __len__(self): return len(self.windows)
    def __getitem__(self,i): return tuple(torch.from_numpy(v) for v in self.windows[i])

def prepare(config):
    paths = sorted(DATA.glob('training_set*/*.psv'))
    expected=config.get('expected_patients')
    if expected and len(paths)!=expected:
        raise ValueError(f'Full configuration requires {expected} real patients; found {len(paths)}. Complete download and verification first.')
    if config.get('split_file'):
        split_path=ROOT/config['split_file']
        if not split_path.exists(): raise ValueError('Run prepare_full_training.py first to verify and freeze full-cohort splits')
        splits=json.loads(split_path.read_text())
        values=[p for group in splits.values() for p in group]
        if len(set(values))!=len(values) or set(values)!={str(p.resolve()) for p in paths}:
            raise ValueError('Saved patient splits contain duplicates or differ from the verified cohort')
        if config.get('statistics_file'):
            stats=json.loads((ROOT/config['statistics_file']).read_text())
            if stats['features']!=FEATURES or not np.isfinite(stats['mean']).all() or not (np.array(stats['std'])>0).all():
                raise ValueError('Invalid frozen normalization statistics')
            return splits,stats
        return splits,fit_stats(splits['train'])
    if config.get('max_patients',0):
        rng=np.random.default_rng(config['seed']); rng.shuffle(paths)
        paths=paths[:config['max_patients']]
    splits=make_splits(paths,config['seed']); stats=fit_stats(splits['train'])
    return splits, stats

if __name__=='__main__':
    import yaml
    config=yaml.safe_load((ROOT/'config.yaml').read_text())
    splits,stats=prepare(config)
    (DATA/'splits.json').write_text(json.dumps(splits,indent=2))
    (DATA/'statistics.json').write_text(json.dumps(stats,indent=2))
    print({k:len(v) for k,v in splits.items()})
