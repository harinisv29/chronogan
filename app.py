import io
import json
import os
import secrets
import threading
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
from flask import Flask,request,jsonify,render_template,session,send_file
from werkzeug.exceptions import RequestEntityTooLarge
from reportlab.pdfgen import canvas
from download_dataset import ROOT
from preprocess import read_patient,FEATURES
from predict import reconstruct
from utils.clinical_validation import validate_upload_schema

app=Flask(__name__)
DEMO_MODE=os.environ.get('CHRONOGAN_DEMO')=='1'
app.secret_key=os.environ.get('CHRONOGAN_SECRET') or secrets.token_hex(32)
app.config.update(MAX_CONTENT_LENGTH=4*1024*1024,SESSION_COOKIE_HTTPONLY=True,SESSION_COOKIE_SAMESITE='Strict')
storage=TemporaryDirectory(prefix='chronogan_')
inference_lock=threading.Lock()

def folder():
    if 'id' not in session: session['id']=secrets.token_hex(16)
    path=Path(storage.name)/session['id']; path.mkdir(exist_ok=True)
    return path

def load_json(path):
    return json.loads(path.read_text()) if path.exists() else None

def active_artifacts():
    requested=os.environ.get('CHRONOGAN_RUN_DIR')
    published=ROOT/'runs'/'full-training'/'results'/'published.json'
    run=ROOT/requested if requested else ROOT/'runs'/'full-training' if published.exists() else ROOT
    selection=load_json(run/'results'/'deployment-selection.json')
    checkpoint=run/selection['checkpoint'] if selection and not DEMO_MODE else run/'models'/'chronogan_best.pt'
    return checkpoint,run/'results'

def payload(frame):
    return json.loads(frame.to_json(orient='records'))

@app.get('/')
def index(): return render_template('index.html',features=FEATURES,demo_mode=DEMO_MODE)

@app.get('/api/status')
def status():
    checkpoint,results=active_artifacts()
    reference=results.parent/'models'/'chronogan_best.pt'
    fingerprint=hashlib.sha256(reference.read_bytes()).hexdigest() if reference.exists() else None
    evaluation=load_json(results/'evaluation.json'); sepsis=load_json(results/'sepsis.json')
    if evaluation and evaluation.get('checkpoint_sha256')!=fingerprint: evaluation=None
    if sepsis and sepsis.get('checkpoint_sha256')!=fingerprint: sepsis=None
    return jsonify(dataset=load_json(ROOT/'data'/'summary.json'),training=load_json(results/'training.json'),evaluation=evaluation,sepsis=sepsis,uncertainty=load_json(results/'uncertainty.json'),ablations=load_json(results/'ablations.json'),deployment=None if DEMO_MODE else load_json(results/'deployment-selection.json'),demo_mode=DEMO_MODE,research_reference_checkpoint=str(reference.relative_to(ROOT)),trained=checkpoint.exists(),active_checkpoint=str(checkpoint.relative_to(ROOT)))

@app.post('/api/upload')
def upload():
    item=request.files.get('file')
    if item is None or not item.filename: return jsonify(error='Choose a CSV or PSV file'),400
    suffix=Path(item.filename).suffix.lower()
    if suffix not in ['.csv','.psv']: return jsonify(error='Only CSV and PSV files are supported'),400
    target=folder()/('incoming'+suffix); item.save(target)
    try:
        validate_upload_schema(target)
        frame=read_patient(target)
        # Uploads must have a uniform hourly grid for the training protocol.
        if 'ICULOS' in frame and len(frame)>1 and not np.allclose(frame.ICULOS.diff().dropna(),1):
            raise ValueError('Rows must use a consecutive hourly grid; insert NaN rows for missing hours')
        frame.to_csv(folder()/'patient.csv',index=False)
        for name in ['reconstructed.csv','mask.csv','uncertainty.csv','metadata.json','raw-predictions.csv','raw-uncertainty.csv']:
            (folder()/name).unlink(missing_ok=True)
        mask=frame.reindex(columns=FEATURES).isna()
        return jsonify(preview=payload(frame.head(100)),rows=len(frame),columns=list(frame.columns),missingness=mask.mean().to_dict(),mask=mask.astype(int).values.tolist(),timeline=(1-mask.mean(axis=1)).tolist())
    except (ValueError,pd.errors.ParserError,UnicodeError) as exc: return jsonify(error=str(exc)),400

@app.post('/api/reconstruct')
def reconstruct_api():
    path=folder()/'patient.csv'
    if not path.exists(): return jsonify(error='Upload a patient first'),400
    checkpoint,results=active_artifacts()
    choice=(request.get_json(silent=True) or {}).get('model','selected')
    if choice not in ['selected','chronogan']: return jsonify(error='Unknown reconstruction model'),400
    if choice=='chronogan': checkpoint=results.parent/'models'/'chronogan_best.pt'
    if not checkpoint.exists(): return jsonify(error='No trained model. Run train.py first.'),409
    try:
        with inference_lock:
            result,mask,uncertainty,meta=reconstruct(read_patient(path),checkpoint=checkpoint)
    except (ValueError,FloatingPointError) as exc:return jsonify(error=str(exc)),400
    result.to_csv(folder()/'reconstructed.csv',index=False); mask.to_csv(folder()/'mask.csv',index=False)
    uncertainty.to_csv(folder()/'uncertainty.csv',index=False)
    result[[column for column in result if column.endswith(('__raw','__imputed','__withheld','__corrected','__reason'))]].to_csv(folder()/'raw-predictions.csv',index=False)
    uncertainty[[column for column in uncertainty if '__raw_' in column]].to_csv(folder()/'raw-uncertainty.csv',index=False)
    (folder()/'metadata.json').write_text(json.dumps(meta,indent=2))
    return jsonify(result=payload(result),uncertainty=payload(uncertainty),metadata=meta)

@app.get('/api/export/<kind>')
def export(kind):
    names={'csv':'reconstructed.csv','mask':'mask.csv','uncertainty':'uncertainty.csv','metadata':'metadata.json','raw':'raw-predictions.csv','raw-uncertainty':'raw-uncertainty.csv'}
    if kind in names:
        path=folder()/names[kind]
        if not path.exists(): return jsonify(error='Reconstruct a patient first'),409
        return send_file(path,as_attachment=True,download_name=names[kind])
    if kind=='pdf':
        meta=load_json(folder()/'metadata.json')
        if meta is None: return jsonify(error='Reconstruct a patient first'),409
        output=io.BytesIO(); pdf=canvas.Canvas(output); y=800
        validation=meta.get('output_validation',{})
        lines=['ChronoGAN research reconstruction report','Research estimates; not verified clinical observations.',*[f'{k}: {v}' for k,v in meta.items() if k not in ['units','output_validation']],f'Output policy: {validation.get("mode","not recorded")}',f'Raw invalid estimates: {validation.get("raw_invalid_count",0)}; withheld: {validation.get("withheld_count",0)}',*validation.get('warnings',[])]
        for line in lines:
            for start in range(0,len(line),95):
                if y<55:pdf.showPage();y=800
                pdf.drawString(40,y,line[start:start+95]); y-=18
        data=pd.read_csv(folder()/'reconstructed.csv')
        if y<75:pdf.showPage();y=800
        pdf.drawString(40,y-18,f'Hourly records: {len(data)}; originally missing cells: {int(data.filter(like="__imputed").sum().sum())}')
        _,results=active_artifacts()
        evaluation=load_json(results/'evaluation.json')
        reference=results.parent/'models'/'chronogan_best.pt'
        selection=load_json(results/'deployment-selection.json')
        reference_hash=hashlib.sha256(reference.read_bytes()).hexdigest() if reference.exists() else None
        if evaluation and evaluation.get('checkpoint_sha256')==reference_hash and meta['checkpoint_sha256'] in [reference_hash,selection.get('checkpoint_sha256') if selection else None]:
            pdf.showPage(); y=800
            for line in ['Held-out real-patient reconstruction experiment','Aggregate errors are standardized, not clinical units.',f'Test patients: {evaluation["patients"]}',* [f'{r["method"]}: MAE {r["normalized_mae"]:.4f}, RMSE {r["normalized_rmse"]:.4f}' for r in evaluation['experiments'] if r['rate']==.2 and r['pattern']=='random'],'Protocol: observed-only artificial masking; patient-disjoint split.','Noise intervals are not calibrated clinical confidence intervals.','No clinical superiority is claimed.']:
                pdf.drawString(40,y,line); y-=20
        pdf.showPage(); pdf.save(); output.seek(0)
        return send_file(output,mimetype='application/pdf',as_attachment=True,download_name='chronogan-report.pdf')
    return jsonify(error='Unknown export type'),404

@app.errorhandler(RequestEntityTooLarge)
def too_large(_): return jsonify(error='File exceeds 4 MB upload limit'),413

@app.get('/api/research-report')
def research_report():
    _,results=active_artifacts(); path=results/'research-summary.md'
    if not path.exists(): return jsonify(error='The full research study has not been published'),409
    return send_file(path,as_attachment=True,download_name='chronogan-research-summary.md')

@app.get('/api/research-errors')
def research_errors():
    _,results=active_artifacts(); path=results/'per-variable-errors.csv'
    if not path.exists(): return jsonify(error='Per-variable research results are not available'),409
    return send_file(path,as_attachment=True,download_name='chronogan-per-variable-errors.csv')

if __name__=='__main__': app.run(host='127.0.0.1',port=int(os.environ.get('CHRONOGAN_PORT','5000')),debug=False)
