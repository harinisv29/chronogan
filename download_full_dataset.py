"""Download the official public S3 objects with per-file size/ETag validation."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import threading
import xml.etree.ElementTree as ET
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from download_dataset import ROOT,DATA

BASE='https://physionet-open.s3.amazonaws.com/'
PREFIX='challenge-2019/1.0.0/training/'
NS={'s3':'http://s3.amazonaws.com/doc/2006-03-01/'}
local=threading.local()

def session():
    if not hasattr(local,'session'):
        local.session=requests.Session()
        local.session.mount('https://',HTTPAdapter(max_retries=Retry(total=8,connect=8,read=8,backoff_factor=1,status_forcelist=[429,500,502,503,504])))
    return local.session

def manifest():
    path=DATA/'official-s3-manifest.json'
    if path.exists(): return json.loads(path.read_text())
    rows=[]; token=None
    while True:
        params={'list-type':'2','prefix':PREFIX,'max-keys':1000}
        if token: params['continuation-token']=token
        response=session().get(BASE,params=params,timeout=(20,120)); response.raise_for_status()
        document=ET.fromstring(response.content)
        for item in document.findall('s3:Contents',NS):
            key=item.findtext('s3:Key',namespaces=NS)
            relative=key.removeprefix(PREFIX)
            parts=Path(relative).parts
            if len(parts)==2 and parts[0] in ['training_setA','training_setB'] and relative.endswith('.psv'):
                rows.append({'key':key,'relative':relative,'size':int(item.findtext('s3:Size',namespaces=NS)),'etag':item.findtext('s3:ETag',namespaces=NS).strip('"')})
        print(f'Listed {len(rows)} official patient objects',flush=True)
        token=document.findtext('s3:NextContinuationToken',namespaces=NS)
        if not token: break
    counts={s:sum(r['relative'].startswith('training_set'+s+'/') for r in rows) for s in ['A','B']}
    if counts!={'A':20336,'B':20000}: raise ValueError(f'Unexpected official S3 manifest counts: {counts}')
    DATA.mkdir(exist_ok=True); path.write_text(json.dumps(rows,indent=2)); return rows

def valid(path,row):
    if not path.exists() or path.stat().st_size!=row['size']: return False
    # These small patient files have single-part S3 ETags, which are content MD5s.
    if len(row['etag'])!=32 or '-' in row['etag']: raise ValueError('Unsupported multipart ETag; require an explicit checksum')
    return hashlib.md5(path.read_bytes(),usedforsecurity=False).hexdigest()==row['etag']

def fetch(row):
    path=DATA/row['relative']; path.parent.mkdir(exist_ok=True)
    if valid(path,row): return 'reused'
    response=session().get(BASE+row['key'],timeout=(20,120)); response.raise_for_status()
    partial=path.with_suffix('.part'); partial.write_bytes(response.content)
    if not valid(partial,row): raise ValueError(f'S3 size/checksum mismatch: {row["key"]}')
    if not response.content.startswith(b'HR|') or b'SepsisLabel' not in response.content.splitlines()[0]:
        raise ValueError('Unexpected clinical PSV header')
    partial.replace(path); return 'downloaded'

def main():
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--workers',type=int,default=32); args=p.parse_args()
    if not 1<=args.workers<=64: p.error('Use 1 to 64 workers')
    rows=manifest(); reused=0; downloaded=0
    counts={s:sum(r['relative'].startswith('training_set'+s+'/') for r in rows) for s in ['A','B']}
    if counts!={'A':20336,'B':20000} or len({r['key'] for r in rows})!=40336 or len({Path(r['relative']).stem for r in rows})!=40336:
        raise ValueError('Official manifest must contain exactly 40336 unique patient IDs and keys')
    for row in rows:
        parts=Path(row['relative']).parts
        if len(parts)!=2 or parts[0] not in ['training_setA','training_setB'] or row['key']!=PREFIX+row['relative']:
            raise ValueError('Unsafe or inconsistent manifest path')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending=[pool.submit(fetch,row) for row in rows]
        for done,future in enumerate(as_completed(pending),1):
            if future.result()=='reused': reused+=1
            else: downloaded+=1
            if done%500==0 or done==len(rows):
                print(f'Patient files verified: {done}/{len(rows)}; reused {reused}; downloaded {downloaded}',flush=True)
    out={'source':BASE+PREFIX,'patients':len(rows),'reused':reused,'downloaded':downloaded,'per_file_integrity':'Exact S3 byte size and single-part ETag/MD5 match'}
    (DATA/'full-download-verification.json').write_text(json.dumps(out,indent=2)); print(out)

if __name__=='__main__': main()
