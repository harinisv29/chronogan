"""Check only official bulk-download endpoints without downloading patient data."""
from concurrent.futures import ThreadPoolExecutor
import json
import requests
from download_dataset import ROOT

urls=[
 'https://physionet.org/static/published-projects/challenge-2019/early-prediction-of-sepsis-from-clinical-data-the-physionetcomputing-in-cardiology-challenge-2019-1.0.0.zip',
 'https://physionet.org/files/challenge-2019/1.0.0/training/training_setA.zip',
 'https://physionet.org/files/challenge-2019/1.0.0/training.zip',
 'https://archive.physionet.org/challenge/2019/training_setA.zip',
 'https://archive.physionet.org/pnw/challenge-2019/training_setA.zip',
 'https://physionet-open.s3.amazonaws.com/challenge-2019/1.0.0/training/training_setA/p000001.psv',
 'https://physionet.org/content/challenge-2019/1.0.0/get-zip/']
def probe(url):
    try:
        response=requests.head(url,timeout=(15,30),allow_redirects=True)
        row={'url':url,'status':response.status_code,'content_length':response.headers.get('Content-Length'),'content_type':response.headers.get('Content-Type')}
    except requests.RequestException as exc: row={'url':url,'error':str(exc)}
    print(row,flush=True); return row
if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=4) as pool: rows=list(pool.map(probe,urls))
    (ROOT/'results'/'gpu'/'dataset-sources.json').write_text(json.dumps(rows,indent=2))
