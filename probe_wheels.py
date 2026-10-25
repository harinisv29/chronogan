import json
import requests
from download_dataset import ROOT

rows=[]
with requests.Session() as session:
    for version,index in [('2.11.0','cu126'),('2.11.0','cu130'),('2.7.1','cu126')]:
        url=f'https://download.pytorch.org/whl/{index}/torch-{version}%2B{index}-cp312-cp312-win_amd64.whl'
        with session.get(url,headers={'Range':'bytes=0-0'},timeout=(15,60)) as response:
            row=dict(version=version,index=index,url=url,status=response.status_code,size=response.headers.get('Content-Range'),sha256=response.headers.get('x-amz-meta-checksum-sha256'))
            rows.append(row); print(row,flush=True)
(ROOT/'results'/'gpu'/'wheel-options.json').write_text(json.dumps(rows,indent=2))
