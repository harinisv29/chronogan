"""Finalize saved measurements using the standard library only: no torch or CUDA."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/loss-weight-study/cuda-v1'
LABELS={'chronogan':'Original ChronoGAN','reconstruction_only':'Archived reconstruction-only','mask_aware_gru':'Mask-aware GRU','no_mask':'Archived no-mask conditioning','no_delta':'Archived no-time-gap conditioning','rec10_adv0':'Fine-tune R10/A0','rec10_adv001':'Fine-tune R10/A0.01','rec10_adv005':'Fine-tune R10/A0.05','rec10_adv01':'Fine-tune R10/A0.1 (control)','rec20_adv01':'Fine-tune R20/A0.1'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def interval(v,percent=False):
    scale=100 if percent else 1
    return f'[{v[0]*scale:.4f}, {v[1]*scale:.4f}]'+('%' if percent else '')

def main():
    source=OUT/'results.json';saved=json.loads(source.read_text());results=saved['results'];protocol=saved['protocol']
    assert json.loads((OUT/'progress.json').read_text())['stage']=='complete'
    assert len(results)==6 and all(len(methods)==10 for methods in results.values())
    assert all(sha(ROOT/f)==h for f,h in protocol['frozen'].items())
    contrasts=list(csv.DictReader((OUT/'paired-contrasts.csv').open(newline='')))
    averages={name:sum(methods[name]['raw']['standardized_mae'] for methods in results.values())/len(results) for name in LABELS}
    ranking=sorted(averages,key=averages.get);rows=[]
    lines=['# Final report: ChronoGAN loss-weight and missingness research','',
        '## Completion status','',
        'The bounded CUDA loss-weight study is complete: 15 three-epoch warm-start fine-tuning runs, three adaptation seeds, and three masking/noise seeds, with all six requested missingness scenarios. This finalization only reads saved measurements; it performs no training, model inference or GPU work. The separate full-cohort research controller was still training BRITS at finalization. That broader study is not declared complete by this report. No dashboard or original research artifacts were changed.','',
        '## Findings and recommendation','',
        f"The mask-aware causal GRU is the recommended primary research reference architecture. It has the lowest descriptive, equally weighted mean standardized MAE across the six scenarios: {averages['mask_aware_gru']:.6f}, versus reconstruction-only {averages['reconstruction_only']:.6f} and original GAN {averages['chronogan']:.6f}. This aggregate is not a preselected deployment criterion, has no aggregate superiority confidence interval, and does not establish a universal winner.",
        'GRU has the lowest MAE at 40% and 60% random missingness and 20% block missingness. Reconstruction-only wins 10% random and 40% blocks numerically. The no-delta ablation wins 20% random numerically. Reconstruction-only versus GRU intervals include zero in five of six scenarios; at 20% blocks, the GRU advantage has an exploratory paired interval excluding zero. Keep both GRU and reconstruction-only as serious candidates rather than selecting from tiny point-estimate differences.',
        'Adversarial training has not demonstrated a reconstruction-accuracy benefit here. Among matched warm-start runs, zero adversarial weight has lower aggregate MAE than the unchanged-loss control in every scenario, but its gains are small. Paired intervals exclude zero only at 10%, 40% and 60% random masking; they include zero at 20% random and both block settings. This supports investigating reconstruction-dominant training, not claiming that adversarial loss always hurts. The archived reconstruction-only model has lower MAE and RMSE than the original GAN in all six of these multi-seed scenarios; its separate historical training trajectory limits causal attribution. The prior single-seed block-RMSE regression did not repeat in the pooled results, which illustrates seed sensitivity.','',
        '## Study design and confidence intervals','',
        'Patient-disjoint subsets: 512 training patients for adaptation, 64 validation patients for best-epoch selection, and 64 held-out audit patients. Original frozen training-only feature statistics and causal independent 24-hour windows are shared by every model. Artificially hidden targets are drawn only from known observations; original missing cells have no ground truth and never enter error calculations. Seeds, exact patient paths, checkpoint hashes and archived masks are saved in protocol.json and masks-*.npz.',
        'Weight pairs (reconstruction, adversarial): (10,0), (10,0.01), (10,0.05), (10,0.1), (20,0.1). All adaptations share the same pretrained generator/discriminator, fresh Adam at 0.0001, batch32, clipping5, matched per-seed shuffles/masks/noise, three epochs and fixed validation masks. Archived checkpoints have one original training seed; adaptations have three new seeds. Each model is evaluated with three mask/noise seeds and 20 GAN noise samples. These are short warm-start experiments, not independent from-scratch replications.',
        'Overall 95% percentile intervals use 1000 paired bootstrap resamples of patient clusters, masking seeds and adaptation seeds. Model differences use shared patient/mask draws; adaptation-control contrasts also pair the training seeds. Archived models cannot provide from-scratch training-seed uncertainty. Intervals are exploratory and not adjusted for multiple comparisons. Previously inspected test patients are not a pristine confirmation cohort. Overlap of separate model intervals is not the paired significance test: use the difference intervals below.',
        'MAE/RMSE in the tables are standardized by frozen training standard deviations; native-unit errors are in per-feature.csv and per-feature-confidence.csv. Coverage is the fraction of artificially hidden known cells released after the unchanged safety policy. Invalid percentage describes raw model outputs before withholding. Counts include repeated evaluation exposures across seed replicates; they are not unique clinical measurements. Raw, released and withheld subsets are separately recorded in results.json.','',
        '## Complete performance tables','']
    for scenario,methods in results.items():
        lines += ['### '+scenario.replace('_',' '),'','| Model | MAE [95% CI] | RMSE [95% CI] | Coverage [95% CI] | Raw invalid |','|---|---:|---:|---:|---:|']
        for name in LABELS:
            value=methods[name];raw=value['raw'];invalid=sum(f['invalid'] for f in raw['per_feature'].values())
            assert sum(f['n'] for f in raw['per_feature'].values())==raw['n']
            assert all(math.isfinite(raw[k]) for k in ['standardized_mae','standardized_rmse','invalid_rate'])
            assert value['released']['invalid_rate']==0 and 0<=value['coverage']<=1
            lines.append(f"| {LABELS[name]} | {raw['standardized_mae']:.6f} {interval(value['mae_95_ci'])} | {raw['standardized_rmse']:.6f} {interval(value['rmse_95_ci'])} | {value['coverage']:.2%} {interval(value['coverage_95_ci'],True)} | {raw['invalid_rate']:.2%} ({invalid}/{raw['n']}) |")
            rows.append({'scenario':scenario,'model':name,'label':LABELS[name],'mae':raw['standardized_mae'],'mae_ci_low':value['mae_95_ci'][0],'mae_ci_high':value['mae_95_ci'][1],'rmse':raw['standardized_rmse'],'rmse_ci_low':value['rmse_95_ci'][0],'rmse_ci_high':value['rmse_95_ci'][1],'coverage':value['coverage'],'coverage_ci_low':value['coverage_95_ci'][0],'coverage_ci_high':value['coverage_95_ci'][1],'raw_invalid_rate':raw['invalid_rate'],'raw_invalid_count':invalid,'raw_scored_count':raw['n'],'raw_nonfinite_count':raw['nonfinite_count'],'withholding_rate':value['withholding_rate']})
        winner=min(methods,key=lambda name:methods[name]['raw']['standardized_mae'])
        lines += ['',f'Lowest observed MAE: **{LABELS[winner]}**. This is a descriptive test ranking, not a deployment selection.','']
    lines += ['## Paired confidence intervals','',
        'Difference = first model MAE minus reference MAE. Negative favors the first model. An interval spanning zero does not establish a difference.','',
        '| Scenario | Comparison | MAE difference | Paired 95% CI | Exploratory interpretation |','|---|---|---:|---:|---|']
    for row in contrasts:
        if (row['model'],row['reference']) not in [('rec10_adv0','rec10_adv01'),('reconstruction_only','mask_aware_gru')]:continue
        low,high=float(row['paired_95_ci_low']),float(row['paired_95_ci_high']);note='Includes zero' if low<=0<=high else ('Favors first model' if high<0 else 'Favors reference')
        lines.append(f"| {row['scenario']} | {LABELS[row['model']]} vs {LABELS[row['reference']]} | {float(row['mae_difference']):.6f} | [{low:.6f}, {high:.6f}] | {note} |")
    lines += ['','## Descriptive average across scenarios','','Equal scenario weights are used solely to summarize this completed pilot. No model was deployed using this post-hoc ranking.','','| Model | Mean standardized MAE |','|---|---:|']
    lines.extend(f'| {LABELS[name]} | {averages[name]:.6f} |' for name in ranking)
    lines += ['','## What should improve next','',
        '1. Keep explicit mask conditioning. Removing it worsened archived MAE. Preserve causal visible-mask delta recomputation and observed-value copy-through.',
        '2. Investigate learned time-gap decay with matched from-scratch seeds. The archived no-delta model improves several point estimates. Uniform-gap probes show some hidden gates approach zero while the median remains one; this uneven decay motivates investigation but is not a measured cohort failure or evidence that removing all time features is correct.',
        '3. Retain reconstruction-dominant GAN training as a candidate, with zero or small adversarial weight tested against a matched control. Do not select the largest reconstruction weight merely from its ratio: clipping and temporal/plausibility penalties also change the optimization.',
        '4. Improve contiguous-gap robustness and context handling. GRU is substantially stronger at 20% blocks here. Any longer-context or overlapping-window method must preserve patient boundaries, chronology and causal inference, and match training to inference.',
        '5. Address nonnegative biomarkers in a separately trained, feature-specific transformation/domain-aware objective. Do not apply log transforms to existing weights or substitute arbitrary normal values. Preserve raw predictions, mathematical-domain checks and legitimate abnormal observations.',
        '6. Validate support and uncertainty before changing release coverage. Fully unobserved features remain withheld; model-noise intervals are not calibrated clinical intervals. A lower raw error does not validate unsupported laboratory estimates.',
        '7. Confirm results using independent patients, more from-scratch training seeds, predefined primary outcomes and paired confidence intervals. No further training is started by this finalization.','',
        '## Integrity, tests and artifacts','',
        f"The CUDA study took {saved['seconds']:.1f} seconds and peaked at {saved['peak_cuda_allocated_mib']:.2f} MiB allocated CUDA tensor memory. This is process tensor allocation, not total shared GPU VRAM. Existing GPU research was left running.",
        'All nine protected original files retain their recorded hashes, including model checkpoints, normalization, splits, clinical registry and deployment policy. Observed values were asserted unchanged during evaluation, and all released predictions passed configured domains. No dashboard model or safety rule was changed.',
        'Automated suite: 53 passed, 1 skipped, 0 failed (tests-final.xml). CUDA was hidden for that suite to avoid redundant GPU work; the completed experiment itself used CUDA forward/backward computation.',
        '- final-performance.csv: all 60 scenario/model rows, numeric CI bounds, invalid rates/counts and coverage.',
        '- paired-contrasts.csv: all saved paired contrasts against GRU and unchanged-loss control.',
        '- per-feature-confidence.csv: raw/released/withheld native-unit feature errors and conditional patient-cluster intervals.',
        '- seed-scores.csv, patient-metrics-*.npz, masks-*.npz: seed-level and patient-level audit trail.',
        '- report.md: detailed prior protocol, gradient diagnostics, decay probes and reproduction commands.',
        '- results.json: original measured results, unchanged by finalization.',
        '- random-missingness.png: measured error versus random missingness.','',
        'This is the final report for the bounded loss-weight pilot, not a claim that the separate full-cohort controller has finished or that any model is clinically validated.']
    with (OUT/'final-performance.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (OUT/'final-research-report.md').write_text('\n'.join(lines),encoding='utf-8')
    manifest={'source_results_sha256':sha(source),'source_contrasts_sha256':sha(OUT/'paired-contrasts.csv'),'performance_rows':len(rows),'protected_files_verified':len(protocol['frozen']),'new_training_started':False,'dashboard_modified':False,'gpu_computation_performed_by_finalizer':False,'descriptive_ranking':[{ 'model':name,'mean_mae':averages[name]} for name in ranking]}
    (OUT/'final-report-manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
