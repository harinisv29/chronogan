"""Local demo evidence generated only from saved, measured study artifacts."""
import csv
import json
from pathlib import Path
import shutil
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parent
STUDY=ROOT/'runs/full-training'
TARGET=ROOT/'results/mit-solve'


def generate():
    TARGET.mkdir(parents=True,exist_ok=True)
    def read(path): return json.loads(path.read_text()) if path.exists() else None
    candidates=[('Full ChronoGAN',STUDY),('Without masks',STUDY/'ablations/no_mask'),('Without time gaps',STUDY/'ablations/no_delta'),('Without adversarial loss',STUDY/'ablations/no_adversarial')]
    rows=[]
    for name,folder in candidates:
        history=read(folder/'results/training.json')
        if history:
            best=min(history,key=lambda row:row['validation_normalized_mae'])
            rows.append({'method':name,'completed_epochs':len(history),'best_epoch':best['epoch'],'validation_mae':best['validation_normalized_mae'],'complete':(folder/'results/training.png').exists(),'protocol':'Training checkpoint selection; one fixed validation noise draw.'})
    for name in ['missingness_aware_gru','brits']:
        folder=STUDY/'baselines'/name; history=read(folder/'history.json')
        if history:
            best=min(history,key=lambda row:row['validation_mae'])
            rows.append({'method':name,'completed_epochs':len(history),'best_epoch':best['epoch'],'validation_mae':best['validation_mae'],'complete':(folder/'complete.json').exists(),'protocol':'Masked validation MAE; BRITS is retrospective.'})
    evaluation=read(STUDY/'results/evaluation.json'); sepsis=read(STUDY/'results/sepsis.json'); selection=read(STUDY/'results/deployment-selection.json')
    stage=read(STUDY/'results/study-progress.json'); tests=read(STUDY/'results/application-verification.json')
    evidence={'snapshot_utc':datetime.now(timezone.utc).isoformat(),'stage':stage,'validation_results':rows,'deployment_selection':selection,'test_reconstruction':evaluation,'sepsis':sepsis,'application_verification':tests,
              'claim_boundary':'Validation results are not held-out test performance. Reconstruction does not establish clinical benefit. No hospital partnership, clinician feedback, patient outcome or clinical deployment is claimed.'}
    (TARGET/'evidence-snapshot.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    lines=['# Verified performance and research status','','This file is generated from saved local evidence. Pending items are not results.',f'Snapshot UTC: {evidence["snapshot_utc"]}.',f'Current recorded stage: `{stage}`.','','| Model | Completed? | Epochs | Best epoch | Validation standardized MAE |','|---|---|---:|---:|---:|']
    lines.extend(f'| {row["method"]} | {row["complete"]} | {row["completed_epochs"]} | {row["best_epoch"]} | {row["validation_mae"]:.6f} |' for row in rows)
    lines+=['','Validation checkpoint scores above use the original training selection protocol. The final deployment uses the separately frozen common validation protocol and includes causal baselines. A BRITS in-progress value is provisional, not a completed baseline result.']
    if selection: lines+=['',f'Deployment selected method: **{selection["selected_method"]}**, validation MAE {selection["validation_normalized_mae"]:.6f}; no test-based selection. Checkpoint: `{selection["checkpoint"]}`.']
    else: lines+=['','Final deployment selection: pending. Policy is frozen in runs/full-training/results/deployment-policy.json.']
    if evaluation:
        lines+=['','## Held-out reconstruction','','| Method | MAE | RMSE |','|---|---:|---:|']
        lines.extend(f'| {r["method"]} | {r["normalized_mae"]:.6f} | {r["normalized_rmse"]:.6f} |' for r in evaluation['experiments'] if r['pattern']=='random' and r['rate']==.2)
    else: lines+=['','Held-out reconstruction and uncertainty calibration: pending.']
    if sepsis:
        lines+=['','## Held-out hourly sepsis-label prediction','','| Input | AUROC | AUPRC | Sensitivity | Specificity | Brier |','|---|---:|---:|---:|---:|---:|']
        for method,result in sepsis['methods'].items(): lines.append('| '+method+' | '+' | '.join(f'{result[key]:.6f}' for key in ['auroc','auprc','sensitivity','specificity','brier'])+' |')
    else: lines+=['','Four downstream prediction experiments and their held-out results: pending.']
    performance=read(ROOT/'runs/performance-20261022/report.json')
    if performance: lines+=['',f'Matched CUDA replay epoch time: {performance["before"]["epoch_seconds"]:.3f} to {performance["after"]["epoch_seconds"]:.3f} seconds, a {performance["speedup_percent"]:.2f}% reduction. Both full-epoch weights and validation score were identical. These are matched benchmark timings, not an assertion that different architectures have equal runtimes.']
    lines+=['','Application verification: '+('passed; inspect evidence-snapshot.json' if tests else 'pending final checkpoint selection and staged checks'),'',evidence['claim_boundary']]
    (TARGET/'verified-performance-current.md').write_text('\n'.join(lines),encoding='utf-8')
    for name in ['comparison.png','robustness.png','temporal-pattern-errors.png','uncertainty-calibration.png','uncertainty-important-variables.png','calibration.png','sepsis-comparison.png','reconstruction-example.png','research-summary.md','dashboard-browser.png']:
        source=STUDY/'results'/name
        if source.exists(): shutil.copy2(source,TARGET/name)
    return evidence


if __name__=='__main__':
    generate(); print(TARGET/'verified-performance-current.md')
