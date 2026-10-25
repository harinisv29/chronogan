import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from utils.research import ROOT,STUDY,hashfile
from utils.experiment import atomic_json

def finalize():
    from research_report import generate
    from research_verify import verify
    generate()
    args=[sys.executable,'-c',"import sys;sys.path.insert(0,'.packages');sys.path.insert(0,'.cuda-packages');import pytest;raise SystemExit(pytest.main(['tests','-q','--basetemp','runs/full-training/final-test-tmp','-o','cache_dir=runs/full-training/.pytest_cache','--junitxml','runs/full-training/tests-final.xml']))"]
    environment={**os.environ,'CHRONOGAN_RUN_DIR':'runs/full-training'}
    with (STUDY/'tests-final.log').open('w') as log: subprocess.run(args,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
    verify()
    suite=ET.parse(STUDY/'tests-final.xml').getroot().find('testsuite')
    tests={key:int(suite.get(key,'0')) for key in ['tests','failures','errors','skipped']}
    assert not tests['failures'] and not tests['errors']
    summary_path=STUDY/'results'/'research-summary.json'
    summary=json.loads(summary_path.read_text()); summary['automated_tests']=tests
    summary['application_verification']=json.loads((STUDY/'results'/'application-verification.json').read_text())
    atomic_json(summary_path,summary)
    with (STUDY/'results'/'research-summary.md').open('a',encoding='utf-8') as report:
        report.write(f'\n\n## Application verification\n\nAutomated tests: {tests}. Staged HTTP checks passed for dashboard/assets, status, held-out CSV/PSV upload, selected reconstruction and stochastic comparison, and CSV/mask/uncertainty/metadata/PDF exports. Recorded observations were preserved. CPU/CUDA inference, seeded CPU replay, ten matched-mask scenario hashes, and headless-browser Plotly rendering were checked. The saved browser screenshot covers the dashboard; upload interactions were tested via HTTP rather than browser automation.\n')
    selection=json.loads((STUDY/'results'/'deployment-selection.json').read_text())
    marker={'checkpoint_sha256':selection['checkpoint_sha256'],'research_reference_sha256':hashfile(STUDY/'models'/'chronogan_best.pt'),'selected_method':selection['selected_method'],'tests':tests,'study_complete':True}
    from mit_solve_materials import generate as generate_demo
    generate_demo()
    atomic_json(STUDY/'results'/'published.json',marker)
    from research_publish import publish
    publish()
    atomic_json(STUDY/'results'/'study-finished.json',marker)
    print(json.dumps(marker,indent=2),flush=True)

if __name__=='__main__': finalize()
