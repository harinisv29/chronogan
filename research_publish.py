"""Publish only a fully verified study to the existing local dashboard."""
import json
import os
import subprocess
import sys
import time
import requests
from utils.research import ROOT,STUDY
from utils.experiment import atomic_json


def publish():
    marker=STUDY/'results'/'published.json'
    assert marker.exists() and json.loads(marker.read_text())['study_complete']
    base='http://127.0.0.1:5000'
    try:
        current=requests.get(base+'/api/status',timeout=5).json()
    except requests.RequestException:
        current={}
    if current.get('active_checkpoint','').replace('\\','/').startswith('runs/full-training/'):
        return
    # PID and exact command were inspected before scheduling. Never stop a trainer
    # or an unrelated/new process that happens to reuse this PID.
    script="""$taskProcess=Get-CimInstance Win32_Process -Filter 'ProcessId=10688';
if ($null -ne $taskProcess) {
 if ($taskProcess.Name -ne 'python.exe' -or $taskProcess.CommandLine -ne '\"C:\\Users\\Admin\\.vscode\\SugarSense\\.tools\\python\\python.exe\" run.py app.py') { throw 'Existing dashboard PID changed; refusing to stop it' }
 Stop-Process -Id 10688 -ErrorAction Stop
}"""
    subprocess.run(['powershell','-NoProfile','-Command',script],check=True,capture_output=True,text=True)
    environment={**os.environ,'CHRONOGAN_RUN_DIR':'runs/full-training','CHRONOGAN_PORT':'5000'}
    with (STUDY/'dashboard.log').open('a') as log:
        server=subprocess.Popen([sys.executable,str(ROOT/'run.py'),'app.py'],cwd=ROOT,env=environment,
                                stdout=log,stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NO_WINDOW)
    for attempt in range(60):
        if server.poll() is not None: raise RuntimeError('Dashboard failed to start; inspect dashboard.log')
        try:
            status=requests.get(base+'/api/status',timeout=3).json()
            if status.get('active_checkpoint','').replace('\\','/').startswith('runs/full-training/'):
                assert status['evaluation']['patients']==6051
                atomic_json(STUDY/'results'/'dashboard-publication.json',{'pid':server.pid,'url':base,'test_patients':6051,'verified':True})
                return
        except requests.RequestException:
            pass
        time.sleep(.5)
    raise RuntimeError('Dashboard did not publish the selected full-training checkpoint')


if __name__=='__main__': publish()
