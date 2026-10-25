"""Verify the complete real cohort, freeze splits, and estimate resources; never train."""
import hashlib
import json
import math
import numpy as np
import pandas as pd
import yaml
import ctypes
import torch
from download_dataset import ROOT,DATA
from download_full_dataset import manifest,valid
from preprocess import FEATURES,read_patient,make_splits,fit_stats

def main():
    verification=ROOT/'results'/'gpu'/'cuda-verification.json'
    smoke=ROOT/'runs'/'gpu-smoke'/'results'/'training.json'
    if not verification.exists() or not json.loads(verification.read_text()).get('allocation_release_succeeded') or not smoke.exists():
        raise SystemExit('GPU verification and successful smoke training are required before full-cohort preparation')
    rows=manifest(); paths=[DATA/r['relative'] for r in rows]
    if len(paths)!=40336 or len(set(paths))!=40336: raise ValueError('Need exactly 40336 unique official patients')
    counts={s:sum(r['relative'].startswith('training_set'+s+'/') for r in rows) for s in ['A','B']}
    if counts!={'A':20336,'B':20000}: raise ValueError('Unexpected hospital counts')
    patient_windows={}; total_rows=0; missing=None; columns=None
    for i,(row,path) in enumerate(zip(rows,paths),1):
        if not valid(path,row): raise ValueError(f'Missing or checksum-invalid real patient: {path}')
        frame=read_patient(path)
        if list(frame.columns)!=FEATURES+['Age','Gender','Unit1','Unit2','HospAdmTime','ICULOS','SepsisLabel']:
            raise ValueError(f'Unexpected clinical columns: {path}')
        if len(frame)>1 and not np.allclose(frame.ICULOS.diff().dropna(),1): raise ValueError(f'Non-hourly patient: {path}')
        if not frame.SepsisLabel.isin([0,1]).all(): raise ValueError(f'Invalid SepsisLabel: {path}')
        patient_windows[str(path.resolve())]=math.ceil(len(frame)/24)
        total_rows+=len(frame); columns=list(frame.columns)
        missing=frame.isna().sum() if missing is None else missing.add(frame.isna().sum(),fill_value=0)
        if i%2000==0: print(f'Parsed and checksum-verified {i}/40336 real patients',flush=True)
    splits=make_splits(paths,42)
    train,val,test=map(set,[splits['train'],splits['val'],splits['test']])
    assert not train&val and not train&test and not val&test
    output=DATA/'full'; output.mkdir(exist_ok=True)
    (output/'splits.json').write_text(json.dumps(splits,indent=2))
    print('Fitting normalization on training patients only...',flush=True)
    stats=fit_stats(splits['train'],progress=True); (output/'statistics.json').write_text(json.dumps(stats,indent=2))
    summary={'source':'https://physionet-open.s3.amazonaws.com/challenge-2019/1.0.0/training/','subset':False,'patients':40336,'sets':counts,'observations':total_rows,'variables':columns,'missingness_percent':(100*missing/total_rows).to_dict(),'integrity':'Every file matched official S3 size and single-part ETag/MD5; schema, temporal grid and labels validated'}
    (DATA/'summary.json').write_text(json.dumps(summary,indent=2))
    config=yaml.safe_load((ROOT/'config.gpu-full.yaml').read_text())
    benchmark='gpu-amp' if config.get('amp') else 'gpu-smoke'
    history=json.loads((ROOT/'runs'/benchmark/'results'/'training.json').read_text())
    windows={k:sum(patient_windows[p] for p in group) for k,group in splits.items()}
    # The exact same hidden size, sequence length and batch size are used in smoke/full configs.
    from train import load_checkpoint
    state=load_checkpoint(ROOT/'runs'/benchmark/'checkpoints'/'last.pt')
    if config.get('amp') and not state.get('amp_enabled'): raise ValueError('Full AMP config requires a successful measured AMP trial')
    smoke_windows={k:sum(patient_windows[p] for p in group) for k,group in state['splits'].items()}
    batch=64
    full_batches=math.ceil(windows['train']/batch)+math.ceil(windows['val']/batch)
    smoke_batches=math.ceil(smoke_windows['train']/batch)+math.ceil(smoke_windows['val']/batch)
    estimated_epoch=history[-1]['epoch_seconds']*full_batches/smoke_batches
    estimate={'patients':40336,'splits':{k:len(v) for k,v in splits.items()},'windows':windows,'batch_size':batch,'sequence_length':24,'hidden':64,'precision':'BF16 autocast, FP32 reconstruction/adversarial losses; successful separate same-cohort stability trial' if config.get('amp') else 'FP32','resource_benchmark':benchmark,'measured_smoke_peak_allocated_mib':max(r['peak_cuda_allocated_mib'] for r in history),'measured_smoke_peak_reserved_mib':max(r['peak_cuda_reserved_mib'] for r in history),'estimated_tensor_dataset_ram_gib':(windows['train']+windows['val'])*24*(len(FEATURES)*3+2)*4/1024**3,'estimated_epoch_seconds':estimated_epoch,'estimated_epoch_range_seconds':[estimated_epoch*.5,estimated_epoch*2],'estimated_50_epoch_hours':50*estimated_epoch/3600,'estimate_limit':'Provisional scaling from smoke epoch batch count, excluding download/preprocessing; CPU scheduling, caching and steady-state GPU utilization can change throughput. No full training run was launched.','command':'.\\.runtime\\python\\python.exe run.py train.py --config config.gpu-full.yaml --device cuda'}
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(name,ctypes.c_ulonglong) for name in ['total_physical','available_physical','total_page','available_page','total_virtual','available_virtual','extended_virtual']]
    memory=Memory(); memory.length=ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)): raise OSError('Could not query host RAM')
    free,total=torch.cuda.mem_get_info()
    estimate['host_total_ram_gib']=memory.total_physical/1024**3
    estimate['host_available_ram_gib']=memory.available_physical/1024**3
    estimate['gpu_free_mib']=free/1024**2
    estimate['resources_ready']=bool(memory.available_physical/1024**3>estimate['estimated_tensor_dataset_ram_gib']*3+2 and free/1024**2>estimate['measured_smoke_peak_reserved_mib']*2+512)
    (ROOT/'results'/'gpu'/'full-training-plan.json').write_text(json.dumps(estimate,indent=2))
    print(json.dumps(estimate,indent=2))

if __name__=='__main__': main()
