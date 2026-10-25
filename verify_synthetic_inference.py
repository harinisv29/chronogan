"""Verify the supplied synthetic record on the live CPU demo, without training."""
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from preprocess import FEATURES,read_patient
import demo_verify

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/inference-audit/20261022'


def main():
    source=OUT/'synthetic-input.csv';patient=read_patient(source)
    server=json.loads((ROOT/'runs/interactive-demo/server.json').read_text());checkpoint=Path(server['checkpoint']);expected=hashlib.sha256(checkpoint.read_bytes()).hexdigest();base=server['url']
    with requests.Session() as client:
        response=client.post(base+'/api/upload',files={'file':('synthetic-icu.csv',source.read_bytes(),'text/csv')},timeout=30);response.raise_for_status()
        response=client.post(base+'/api/reconstruct',json={'model':'chronogan'},timeout=120);response.raise_for_status();body=response.json()
        assert body['metadata']['inference_device']=='cpu' and body['metadata']['output_validation']['mode']=='withhold'
        result=pd.DataFrame(body['result'])
        for name in FEATURES:
            original=patient[name];known=original.notna();np.testing.assert_array_equal(original[known],result.loc[known,name])
            np.testing.assert_array_equal(original.isna(),result[name+'__imputed'])
            values=pd.to_numeric(result[name]).to_numpy();withheld=result[name+'__withheld'].to_numpy()
            assert (np.isfinite(values)|withheld).all() and not np.isinf(values).any()
        for name in ['AST','TroponinI']:
            assert result[name].isna().all() and result[name+'__withheld'].all() and result[name+'__raw'].notna().all()
        export_checks={}
        for kind in ['csv','mask','uncertainty','metadata','pdf','raw','raw-uncertainty']:
            response=client.get(base+'/api/export/'+kind,timeout=30);response.raise_for_status();assert response.content
            target=OUT/('live-export.'+('pdf' if kind=='pdf' else 'json' if kind=='metadata' else kind+'.csv'));target.write_bytes(response.content)
            export_checks[kind]=len(response.content)
        exported=pd.read_csv(io.BytesIO(client.get(base+'/api/export/csv').content))
        assert exported.AST.isna().all() and exported.TroponinI.isna().all()
    demo_verify.FOLDER=OUT/'browser';demo_verify.FOLDER.mkdir(exist_ok=True)
    browser=demo_verify.browser(base,source.read_text(),'TroponinI')
    assert browser['validation_warnings_visible'] and browser['axis_unit']=='TroponinI (ng/mL)'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==expected
    report={'passed':True,'rows':len(patient),'observations_preserved':True,'missingness_preserved':True,'checkpoint_unchanged':True,'metadata':body['metadata'],'exports_bytes':export_checks,'browser':browser}
    (OUT/'synthetic-workflow.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'passed':True,'rows':len(patient),'validation':body['metadata']['output_validation']['mode'],'withheld':body['metadata']['output_validation']['withheld_count'],'raw_invalid':body['metadata']['output_validation']['raw_invalid_count'],'axis':browser['axis_unit'],'exports':list(export_checks)},indent=2))


if __name__=='__main__':main()
