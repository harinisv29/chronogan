"""End-to-end full checkpoint and staged Flask verification on a held-out patient."""
import io
import json
import os
import subprocess
import sys
import time
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from preprocess import read_patient,FEATURES
from predict import reconstruct
from train import load_checkpoint
from utils.research import ROOT,STUDY
from utils.experiment import atomic_json


def verify_test_masks(state):
    from torch.utils.data import DataLoader
    import torch
    from utils.research import cached_data,SCENARIOS,hide
    loader=DataLoader(cached_data(state,'test'),batch_size=64,shuffle=False)
    checks=[]
    for pattern,rate in SCENARIOS:
        rng=torch.Generator(device='cuda').manual_seed(2026); digest=hashlib.sha256(); scored=0
        for _,m,_,v,_ in loader:
            _,held=hide(m.cuda(),v.cuda(),pattern,rate,rng)
            digest.update(held.cpu().numpy().tobytes());scored+=int(held.sum())
        recorded=json.loads((STUDY/'results'/f'evaluation-{pattern}-{rate}.json').read_text())
        assert all(row['mask_sha256']==digest.hexdigest() and row['held_out_cells']==scored for row in recorded['experiments'])
        checks.append({'pattern':pattern,'rate':rate,'mask_sha256':digest.hexdigest(),'scored_cells':scored})
    atomic_json(STUDY/'results'/'mask-reproducibility.json',{'scenarios':checks,'passed':True})
    return checks


def browser_check(base):
    import demo_verify
    state=load_checkpoint(STUDY/'models'/'chronogan_best.pt')
    patient=read_patient(state['splits']['test'][0])
    demo_verify.FOLDER=STUDY/'browser-check'
    demo_verify.FOLDER.mkdir(parents=True,exist_ok=True)
    result=demo_verify.browser(base,patient.to_csv(index=False),'HR',expected_device=None)
    import shutil
    shutil.copy2(demo_verify.FOLDER/'uncertainty-browser.png',STUDY/'results'/'dashboard-browser.png')
    return result


def verify():
    selection=json.loads((STUDY/'results'/'deployment-selection.json').read_text())
    checkpoint=STUDY/selection['checkpoint']; state=load_checkpoint(checkpoint)
    mask_checks=verify_test_masks(state)
    path=state['splits']['test'][0]
    assert path not in state['splits']['train'] and path not in state['splits']['val']
    original=read_patient(path)
    result,mask,uncertainty,metadata=reconstruct(original,checkpoint=checkpoint,samples=20,device='cuda',seed=2468)
    cpu,_,cpu_uncertainty,cpu_meta=reconstruct(original,checkpoint=checkpoint,samples=20,device='cpu',seed=2468)
    repeat,_,_,_=reconstruct(original,checkpoint=checkpoint,samples=20,device='cpu',seed=2468)
    assert np.array_equal(cpu[FEATURES].to_numpy(),repeat[FEATURES].to_numpy(),equal_nan=True)
    assert (np.isfinite(cpu[FEATURES].to_numpy())|cpu[[f+'__withheld' for f in FEATURES]].to_numpy()).all()
    for feature in FEATURES:
        known=original[feature].notna(); assert np.array_equal(original.loc[known,feature],result.loc[known,feature])
        assert np.array_equal(original.loc[known,feature],cpu.loc[known,feature])
    assert (np.isfinite(result[FEATURES].to_numpy())|result[[f+'__withheld' for f in FEATURES]].to_numpy()).all()
    folder=STUDY/'inference'; folder.mkdir(exist_ok=True)
    result.to_csv(folder/'reconstructed.csv',index=False); mask.to_csv(folder/'mask.csv',index=False); uncertainty.to_csv(folder/'uncertainty.csv',index=False)
    atomic_json(folder/'metadata.json',{**metadata,'patient':path,'patient_not_in_train_or_validation':True})
    environment={**os.environ,'CHRONOGAN_RUN_DIR':'runs/full-training','CHRONOGAN_PORT':'5001'}
    assert metadata['checkpoint_sha256']==selection['checkpoint_sha256']
    checks={'cpu_inference':True,'cuda_inference':True,'cpu_seed_reproducibility':True,'replayed_mask_scenarios':len(mask_checks),
            'selected_method':selection['selected_method'],'uncertainty_available':metadata['uncertainty_available'],
            'cross_device_note':'Both devices must preserve observations and finite estimates; random draws and floating-point kernels need not produce identical estimates.'}
    bounds={'HR':(0,350),'O2Sat':(0,100),'Temp':(15,50),'SBP':(0,400),'MAP':(0,350),'DBP':(0,300),'Resp':(0,120),'pH':(6,9),'Glucose':(0,2000),'Creatinine':(0,50),'WBC':(0,500),'Platelets':(0,2500)}
    checks['range_diagnostics']={name:{'imputed_outside_broad_soft_bounds':int(((result[name]<lo)|(result[name]>hi))[mask[name]].sum()),'bounds':[lo,hi]} for name,(lo,hi) in bounds.items()}
    checks['range_diagnostics_note']='Counts are disclosed, not clipped or treated as clinical validation; these are broad research bounds.'
    process=None
    try:
        with (STUDY/'application-check.log').open('w') as log:
            process=subprocess.Popen([sys.executable,str(ROOT/'run.py'),'app.py'],cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
            base='http://127.0.0.1:5001'
            for attempt in range(40):
                if process.poll() is not None: raise RuntimeError('Staging Flask server exited; inspect application-check.log')
                try:
                    response=requests.get(base+'/api/status',timeout=2)
                    if response.status_code==200: break
                except requests.RequestException: pass
                time.sleep(.5)
            else: raise RuntimeError('Staging server did not become ready')
            with requests.Session() as client:
                for route in ['/','/static/js/app.js','/static/css/style.css','/api/status','/api/research-report','/api/research-errors']:
                    r=client.get(base+route,timeout=30); r.raise_for_status(); checks[route]=r.status_code
                status=client.get(base+'/api/status',timeout=30).json()
                assert status['dataset']['patients']==40336 and status['evaluation']['patients']==6051
                assert status['active_checkpoint'].replace('\\','/').startswith('runs/full-training/')
                r=client.post(base+'/api/upload',files={'file':('held-out.csv',original.to_csv(index=False).encode(),'text/csv')},timeout=30); r.raise_for_status(); checks['csv_upload']=r.status_code
                r=client.post(base+'/api/reconstruct',timeout=120); r.raise_for_status(); checks['reconstruction']=r.status_code
                assert len(r.json()['uncertainty'])==len(original)
                for kind in ['csv','mask','uncertainty','metadata','pdf']:
                    r=client.get(base+'/api/export/'+kind,timeout=30); r.raise_for_status(); assert len(r.content)>0
                    checks['export_'+kind]={'status':r.status_code,'bytes':len(r.content)}
                    if kind=='csv':
                        exported=pd.read_csv(io.BytesIO(r.content))
                        for feature in FEATURES:
                            known=original[feature].notna(); np.testing.assert_allclose(exported.loc[known,feature],original.loc[known,feature],rtol=0,atol=1e-10)
                with open(path,'rb') as source:
                    r=client.post(base+'/api/upload',files={'file':('held-out.psv',source,'text/plain')},timeout=30); r.raise_for_status(); checks['psv_upload']=r.status_code
                r=client.post(base+'/api/reconstruct',json={'model':'chronogan'},timeout=120);r.raise_for_status()
                assert r.json()['metadata']['uncertainty_available']
                checks['full_chronogan_comparison']=r.status_code
                checks['browser_charts']=browser_check(base)
    finally:
        if process is not None and process.poll() is None:
            process.terminate(); process.wait(timeout=20)
    checks.update(patient=path,observations_preserved=True,held_out_inference=True)
    atomic_json(STUDY/'results'/'application-verification.json',checks)
    return checks

if __name__=='__main__': print(json.dumps(verify(),indent=2))
