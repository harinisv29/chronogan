"""Verify the independent CPU demo through HTTP and a GPU-disabled browser."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import time
from urllib.parse import urlsplit
import numpy as np
import pandas as pd
import requests
from preprocess import FEATURES,read_patient
from train import load_checkpoint

ROOT=Path(__file__).resolve().parent
FOLDER=ROOT/'runs/interactive-demo'


class CDP:
    def __init__(self,url):
        parsed=urlsplit(url);self.socket=socket.create_connection((parsed.hostname,parsed.port),timeout=120);self.number=0
        key=base64.b64encode(os.urandom(16)).decode()
        self.socket.sendall(f'GET {parsed.path} HTTP/1.1\r\nHost: {parsed.netloc}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n'.encode())
        headers=b''
        while not headers.endswith(b'\r\n\r\n'): headers+=self.socket.recv(1)
        assert b' 101 ' in headers,headers
    def read(self,n):
        result=b''
        while len(result)<n:
            chunk=self.socket.recv(n-len(result))
            if not chunk: raise ConnectionError('Browser debugging socket closed')
            result+=chunk
        return result
    def send(self,payload,opcode=1):
        mask=os.urandom(4);n=len(payload)
        prefix=bytes([128|opcode,128|(n if n<126 else 126 if n<65536 else 127)])
        if n>=126: prefix+=struct.pack('!H' if n<65536 else '!Q',n)
        self.socket.sendall(prefix+mask+bytes(value^mask[i%4] for i,value in enumerate(payload)))
    def receive(self):
        chunks=[]
        while True:
            first,second=self.read(2);n=second&127
            if n==126:n=struct.unpack('!H',self.read(2))[0]
            elif n==127:n=struct.unpack('!Q',self.read(8))[0]
            mask=self.read(4) if second&128 else None
            data=self.read(n)
            if mask:data=bytes(value^mask[i%4] for i,value in enumerate(data))
            opcode=first&15
            if opcode==8:raise ConnectionError('Browser debugging socket closed')
            if opcode==9:self.send(data,10);continue
            chunks.append(data)
            if first&128:return json.loads(b''.join(chunks))
    def call(self,method,params=None):
        self.number+=1;number=self.number
        self.send(json.dumps({'id':number,'method':method,'params':params or {}}).encode())
        while True:
            response=self.receive()
            if response.get('id')==number:
                if 'error' in response:raise RuntimeError(response['error'])
                return response.get('result',{})


def browser(base,csv_text,feature,expected_device='cpu'):
    binary=Path(r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe')
    with socket.socket() as probe:probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
    args=[str(binary),'--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check',f'--remote-debugging-port={port}',f'--user-data-dir={FOLDER/"browser-profile"}','--window-size=1440,1080',base]
    process=None;client=None
    try:
        with (FOLDER/'browser.log').open('w') as log:
            process=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        for _ in range(60):
            try:
                pages=requests.get(f'http://127.0.0.1:{port}/json/list',timeout=2).json()
                page=next((p for p in pages if p.get('type')=='page' and p.get('url','').startswith(base)),None)
                if page:break
            except requests.RequestException:pass
            time.sleep(.5)
        else:raise RuntimeError('Browser debugging endpoint did not become ready')
        client=CDP(page['webSocketDebuggerUrl'])
        for _ in range(80):
            try:
                ready=client.call('Runtime.evaluate',{'expression':"document.readyState==='complete' && typeof drawSeries==='function' && typeof Plotly!=='undefined'",'returnByValue':True})
                if ready.get('result',{}).get('value'):break
            except RuntimeError as error:
                if 'context' not in str(error).lower():raise
            time.sleep(.2)
        else:raise RuntimeError('Demo page scripts did not become ready')
        expression='''(async()=>{
          for(let i=0;i<100&&(!trained||typeof Plotly==='undefined');i++)await new Promise(r=>setTimeout(r,100));
          if(document.querySelector('aside')||document.querySelector('#dashboard'))throw Error('Old dashboard still present');
          const transfer=new DataTransfer();transfer.items.add(new File([CSV_TEXT],'browser-patient.csv',{type:'text/csv'}));
          document.getElementById('file').files=transfer.files;document.getElementById('file').dispatchEvent(new Event('change',{bubbles:true}));
          const before=performance.getEntriesByType('resource').filter(r=>r.name.endsWith('/api/reconstruct')).length;
          if(before!==0||reconstructed!==null)throw Error('Inference ran before Analyze Patient');
          document.getElementById('analyze').click();
          const duplicate_prevented=document.getElementById('analyze').disabled;
          for(let i=0;i<1200&&busy;i++)await new Promise(r=>setTimeout(r,100));
          if(!reconstructed||!document.getElementById('error').hidden)throw Error(document.getElementById('error').textContent||'No reconstruction result');
          document.getElementById('feature').value=FEATURE;
          drawSeries();await new Promise(r=>setTimeout(r,700));document.getElementById('visualization').scrollIntoView();
          return {device:reconstructed.metadata.inference_device,interval:reconstructed.metadata.interval,
            uncertainty_available:reconstructed.metadata.uncertainty_available,
            plotted_uncertainty:reconstructed.uncertainty.some((row,i)=>reconstructed.result[i][FEATURE+'__imputed']&&Number.isFinite(row[FEATURE+'__p05'])&&Number.isFinite(row[FEATURE+'__p95'])),
            traces:document.getElementById('series').data.map(t=>t.name),
            points:document.getElementById('series').data[0].y.length,
            imputed_cells_marked:document.querySelectorAll('#resultTable .imputed').length,
            demo_banner:document.getElementById('demoNotice')?.textContent||'Research prototype',
            duplicate_prevented,hour_count:document.getElementById('hourCount').textContent,
            missing_count:document.getElementById('missingCount').textContent,
            reconstructed_count:document.getElementById('reconstructedCount').textContent,
            validation_summary:document.getElementById('validationSummary')?.textContent,
            validation_warnings_visible:!document.getElementById('validationWarnings')?.hidden,
            axis_unit:document.getElementById('series').layout.yaxis.title.text,
            inference_requests:performance.getEntriesByType('resource').filter(r=>r.name.endsWith('/api/reconstruct')).length,
            svg_count:document.querySelectorAll('#series svg').length};
        })()'''.replace('CSV_TEXT',json.dumps(csv_text)).replace('FEATURE',json.dumps(feature))
        result=client.call('Runtime.evaluate',{'expression':expression,'awaitPromise':True,'returnByValue':True})
        assert 'exceptionDetails' not in result,result
        value=result['result']['value']
        if expected_device is not None:assert value['device']==expected_device
        assert value['svg_count']>0 and value['duplicate_prevented'] and value['inference_requests']==1
        if value['uncertainty_available'] and value['plotted_uncertainty']:assert '90% noise interval' in value['traces']
        position=client.call('Runtime.evaluate',{'expression':"(()=>{const r=document.getElementById('visualization').getBoundingClientRect();return {x:Math.max(0,r.left+scrollX),y:r.top+scrollY,width:r.width,height:r.height,scale:1}})()",'returnByValue':True})['result']['value']
        screenshot=client.call('Page.captureScreenshot',{'format':'png','captureBeyondViewport':True,'clip':position})
        (FOLDER/'uncertainty-browser.png').write_bytes(base64.b64decode(screenshot['data']))
        extra=client.call('Runtime.evaluate',{'expression':"(()=>{document.getElementById('search').value='HR';document.getElementById('search').dispatchEvent(new Event('input'));const rows=[...document.querySelectorAll('#resultTable tbody tr')];return {search_filters_hr:rows.length>0&&rows.every(r=>r.cells[1].textContent==='HR'),page_rows:rows.length,next_page_enabled:!document.getElementById('nextPage').disabled}})()",'returnByValue':True})['result']['value']
        assert extra['search_filters_hr'] and extra['page_rows']<=30
        value['table_checks']=extra
        client.call('Emulation.setDeviceMetricsOverride',{'width':390,'height':844,'deviceScaleFactor':1,'mobile':True})
        client.call('Runtime.evaluate',{'expression':"(async()=>{window.dispatchEvent(new Event('resize'));window.scrollTo(0,0);await new Promise(r=>setTimeout(r,700));return true})()",'awaitPromise':True})
        mobile=client.call('Runtime.evaluate',{'expression':"({width:innerWidth,content:document.documentElement.scrollWidth,no_sidebar:!document.querySelector('aside')})",'returnByValue':True})['result']['value']
        assert mobile['content']<=mobile['width']+1,mobile
        value['mobile_layout']=mobile
        shot=client.call('Page.captureScreenshot',{'format':'png','captureBeyondViewport':False})
        (FOLDER/'mobile-browser.png').write_bytes(base64.b64decode(shot['data']))
        return value
    finally:
        if client:client.socket.close()
        if process is not None and process.poll() is None:process.terminate();process.wait(timeout=20)


def main():
    server=json.loads((FOLDER/'server.json').read_text());base=server['url'];checkpoint=Path(server['checkpoint'])
    expected=server['checkpoint_sha256'];assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==expected
    state=load_checkpoint(checkpoint,'cpu');path=state['splits']['test'][0];patient=read_patient(path)
    (FOLDER/'sample-patient.psv').write_bytes(Path(path).read_bytes())
    patient.to_csv(FOLDER/'sample-patient.csv',index=False)
    checks={'url':base,'checkpoint':str(checkpoint),'device':'cpu','patients_used':1,'hourly_rows':len(patient),'sample_source':path}
    for suffix,content in [('csv',patient.to_csv(index=False).encode()),('psv',Path(path).read_bytes())]:
        with requests.Session() as client:
            status=client.get(base+'/api/status',timeout=15);status.raise_for_status()
            assert status.json()['demo_mode'] and status.json()['active_checkpoint'].replace('\\','/')=='runs/full-training/models/chronogan_best.pt'
            response=client.post(base+'/api/upload',files={'file':('patient.'+suffix,content)},timeout=30);response.raise_for_status()
            response=client.post(base+'/api/reconstruct',json={'model':'chronogan'},timeout=120);response.raise_for_status();body=response.json()
            assert body['metadata']['inference_device']=='cpu' and body['metadata']['checkpoint_sha256']==expected
            output=pd.DataFrame(body['result']);values=output[FEATURES].astype(float).to_numpy();withheld=output[[f+'__withheld' for f in FEATURES]].to_numpy()
            assert (np.isfinite(values)|withheld).all() and not np.isinf(values).any()
            for feature in FEATURES:
                known=patient[feature].notna();np.testing.assert_array_equal(patient.loc[known,feature],output.loc[known,feature])
                assert (output[feature+'__imputed']==~known).all()
                lo=np.array([row[feature+'__p05'] for row in body['uncertainty']],dtype=float);hi=np.array([row[feature+'__p95'] for row in body['uncertainty']],dtype=float)
                suppressed=np.array([row[feature+'__interval_withheld'] for row in body['uncertainty']])
                assert (np.isfinite(lo)|suppressed).all() and (np.isfinite(hi)|suppressed).all() and (lo[~suppressed]<=hi[~suppressed]).all()
            exports={}
            for kind in ['csv','mask','uncertainty','metadata','pdf','raw','raw-uncertainty']:
                response=client.get(base+'/api/export/'+kind,timeout=30);response.raise_for_status();assert response.content
                exports[kind]=len(response.content)
            checks[suffix]={'upload':True,'reconstruction':True,'observations_preserved':True,'uncertainty_finite_ordered':True,'exports_bytes':exports}
    feature=next(name for name in FEATURES if patient[name].isna().any() and patient[name].notna().any())
    checks['browser']=browser(base,patient.to_csv(index=False),feature)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==expected
    checks['checkpoint_unchanged']=True
    (FOLDER/'verification.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
