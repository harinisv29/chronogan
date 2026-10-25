import hashlib
import json
from app import app,active_artifacts
from download_dataset import ROOT

def test_dashboard_assets_and_actual_experiment_results():
    app.config.update(TESTING=True)
    with app.test_client() as client:
        response=client.get('/')
        assert response.status_code==200 and b'plotly' in response.data.lower()
        assert b'<aside' not in response.data and b'id="dashboard"' not in response.data
        assert b'id="uploadForm"' in response.data
        for asset in ['/static/js/app.js','/static/css/style.css','/static/js/patient.js','/static/css/patient.css','/static/vendor/plotly-2.35.2.min.js']:
            assert client.get(asset).status_code==200
        state=client.get('/api/status')
        assert state.status_code==200
        checkpoint,results=active_artifacts()
        evaluation=results/'evaluation.json'
        if checkpoint.exists() and evaluation.exists():
            recorded=json.loads(evaluation.read_text())
            reference=results.parent/'models'/'chronogan_best.pt'
            if recorded.get('checkpoint_sha256')==hashlib.sha256(reference.read_bytes()).hexdigest():
                assert state.json['evaluation']==recorded
