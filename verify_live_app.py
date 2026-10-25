"""Check the running dashboard with a fresh session and a held-out public patient."""
import io
import json
import numpy as np
import pandas as pd
import requests
from download_dataset import ROOT
from preprocess import FEATURES,read_patient
from train import load_checkpoint

base='http://127.0.0.1:5000'
state=load_checkpoint(ROOT/'models'/'chronogan_best.pt')
original=read_patient(state['splits']['test'][0])
checks={}
with requests.Session() as client:
    for path in ['/','/static/js/app.js','/static/css/style.css','/api/status']:
        response=client.get(base+path,timeout=30); response.raise_for_status(); checks[path]=response.status_code
    status=client.get(base+'/api/status',timeout=30).json()
    checks['verified_patient_count']=status['dataset']['patients']
    checks['original_baseline_experiments']=len(status['evaluation']['experiments'])
    response=client.post(base+'/api/upload',files={'file':('held-out.csv',original.to_csv(index=False).encode(),'text/csv')},timeout=30)
    response.raise_for_status(); checks['csv_upload']=response.status_code
    response=client.post(base+'/api/reconstruct',timeout=120); response.raise_for_status()
    checks['reconstruction']=response.status_code
    checks['uncertainty_rows']=len(response.json()['uncertainty'])
    for kind in ['csv','mask','uncertainty','metadata','pdf']:
        response=client.get(base+'/api/export/'+kind,timeout=30); response.raise_for_status()
        checks['export_'+kind]={'status':response.status_code,'bytes':len(response.content)}
        if kind=='csv':
            result=pd.read_csv(io.BytesIO(response.content))
            for feature in FEATURES:
                known=original[feature].notna()
                np.testing.assert_allclose(original.loc[known,feature],result.loc[known,feature],rtol=0,atol=1e-10)
checks['observations_preserved']=True
(ROOT/'results'/'gpu'/'live-app-verification.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
