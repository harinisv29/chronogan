"""Figures and a measured research summary; no fabricated or inferred performance."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.research import STUDY
from utils.experiment import atomic_json
from train import load_checkpoint

def generate():
    root=STUDY/'results'
    read=lambda name:json.loads((root/name).read_text())
    state=load_checkpoint(STUDY/'checkpoints'/'last.pt'); history=state['history']
    evaluation=read('evaluation.json'); sepsis=read('sepsis.json'); uncertainty=read('uncertainty.json'); ablations=read('ablations.json')
    primary=[r for r in evaluation['experiments'] if r['pattern']=='random' and r['rate']==.2]
    with (root/'reconstruction-comparisons.csv').open('w',newline='') as file:
        fields=['method','pattern','rate','normalized_mae','normalized_rmse','held_out_cells','actual_hiding_fraction','mask_sha256']
        writer=csv.DictWriter(file,fieldnames=fields,extrasaction='ignore'); writer.writeheader(); writer.writerows(evaluation['experiments'])
    with (root/'uncertainty-comparisons.csv').open('w',newline='') as file:
        fields=['method','pattern','rate','nominal_coverage','empirical_coverage','mean_normalized_width']
        writer=csv.DictWriter(file,fieldnames=fields,extrasaction='ignore'); writer.writeheader(); writer.writerows(uncertainty['results'])
    full=next(r for r in primary if r['method']=='chronogan')
    selection=read('deployment-selection.json')
    selected=next(r for r in primary if r['method']==selection['selected_method'])
    nonadversarial=next(r for r in primary if r['method']=='no_adversarial')
    component_evidence={'primary_point_error':{name:next(row for row in primary if row['method']==name)['normalized_mae'] for name in ['chronogan','no_delta','no_adversarial']},
                        'temporal_first_difference_error':{name:next(row for row in primary if row['method']==name).get('temporal_normalized_mae') for name in ['chronogan','no_delta','no_adversarial']},
                        'block_errors':[{'method':row['method'],'rate':row['rate'],'normalized_mae':row['normalized_mae'],'temporal_normalized_mae':row.get('temporal_normalized_mae')} for row in evaluation['experiments'] if row['pattern']=='block' and row['method'] in ['chronogan','no_delta','no_adversarial']],
                        'noise_coverage':[row for row in uncertainty['results'] if row['method'] in ['chronogan','no_adversarial'] and row['nominal_coverage']==.9],
                        'interpretation':'Descriptive one-seed comparisons. Lower temporal errors or better empirical coverage support only those measured diagnostics; no blinded realism assessment or clinical benefit was established.'}
    atomic_json(root/'component-benefits.json',component_evidence)
    fig,ax=plt.subplots(figsize=(9,4))
    temporal_rows=[row for row in primary if row.get('temporal_normalized_mae') is not None]
    ax.bar([row['method'] for row in temporal_rows],[row['temporal_normalized_mae'] for row in temporal_rows]); ax.set_ylabel('Standardized first-difference MAE / hour');fig.autofmt_xdate();fig.tight_layout();fig.savefig(root/'temporal-pattern-errors.png');plt.close(fig)
    preserved=json.loads((STUDY/'readiness.json').read_text())['preserved_files']
    from download_dataset import ROOT
    for relative,expected in preserved.items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected,f'Previous experiment changed: {relative}'
    for filename in ['training.json','evaluation.json','sepsis.json','uncertainty.json','ablations.json']:
        assert (root/filename).exists()
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for key in ['adversarial_loss','reconstruction_mse','temporal_loss']:
        axes[0].plot([r['epoch'] for r in history],[r[key] for r in history],label=key)
    axes[0].legend(fontsize=8); axes[0].set_title('Generator objective components')
    axes[1].plot([r['epoch'] for r in history],[r['validation_normalized_mae'] for r in history]); axes[1].set_title('Validation normalized MAE')
    fig.tight_layout(); fig.savefig(root/'loss-components.png'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5)); ax.plot([0,1],[0,1],'--',color='gray',label='Nominal = empirical')
    for method in ['chronogan','no_mask','no_delta','no_adversarial']:
        rows=[r for r in uncertainty['results'] if r['method']==method and r['pattern']=='random' and r['rate']==.2]
        ax.plot([r['nominal_coverage'] for r in rows],[r['empirical_coverage'] for r in rows],marker='o',label=method)
    ax.legend(); ax.set_xlabel('Nominal noise interval mass'); ax.set_ylabel('Empirical held-out coverage')
    fig.tight_layout(); fig.savefig(root/'uncertainty-calibration.png'); plt.close(fig)
    vital_names=['HR','O2Sat','Temp','SBP','MAP','DBP','Resp','Creatinine','Glucose']
    fig,ax=plt.subplots(figsize=(12,4))
    for offset,method in [(-.2,'chronogan'),(.2,'no_adversarial')]:
        row=next(r for r in uncertainty['results'] if r['method']==method and r['pattern']=='random' and r['rate']==.2 and r['nominal_coverage']==.9)
        ax.bar(np.arange(len(vital_names))+offset,[row['per_variable'][name]['coverage'] if row['per_variable'][name] else np.nan for name in vital_names],width=.4,label=method)
    ax.axhline(.9,linestyle='--',color='gray',label='Nominal 90%');ax.set_xticks(range(len(vital_names)),vital_names);ax.set_ylabel('Empirical noise-interval coverage');ax.legend();fig.tight_layout();fig.savefig(root/'uncertainty-important-variables.png');plt.close(fig)
    features=state['stats']['features']; scale=state['stats']['std']; fig,ax=plt.subplots(figsize=(14,5))
    for offset,method in [(-.2,'forward'),(.2,'chronogan')]:
        row=next(r for r in primary if r['method']==method)
        ax.bar(np.arange(len(features))+offset,[row['per_variable'][f]['mae']/s if row['per_variable'][f] else np.nan for f,s in zip(features,scale)],width=.4,label=method)
    ax.set_xticks(range(len(features)),features,rotation=90,fontsize=8); ax.set_ylabel('Per-variable MAE / training SD'); ax.legend()
    fig.tight_layout(); fig.savefig(root/'per-variable-errors.png'); plt.close(fig)
    with (root/'per-variable-errors.csv').open('w',newline='') as file:
        writer=csv.writer(file); writer.writerow(['method','variable','mae_original_units','rmse_original_units','scored_cells'])
        for row in primary:
            for name,value in row['per_variable'].items():
                if value: writer.writerow([row['method'],name,value['mae'],value['rmse'],value['n']])
    fig,ax=plt.subplots(figsize=(8,4)); variants=[r for r in primary if r['method'] in ['chronogan','no_mask','no_delta','no_adversarial']]
    ax.bar([r['method'] for r in variants],[r['normalized_mae'] for r in variants]); ax.set_ylabel('Normalized MAE')
    fig.tight_layout(); fig.savefig(root/'ablations.png'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4)); names=list(sepsis['methods'])
    for ax,key in zip(axes,['auroc','auprc']):
        ax.bar(names,[sepsis['methods'][n][key] for n in names]); ax.set_title(key.upper())
    fig.tight_layout(); fig.savefig(root/'sepsis-comparison.png'); plt.close(fig)
    coverage=next(r for r in uncertainty['results'] if r['method']=='chronogan' and r['pattern']=='random' and r['rate']==.2 and r['nominal_coverage']==.9)
    best_state=load_checkpoint(STUDY/'models'/'chronogan_best.pt')
    summary={'best_epoch':best_state['epoch']+1,'patients':{k:len(v) for k,v in state['splits'].items()},'completed_epochs':len(history),'best_validation_normalized_mae':state['best'],'test_primary_protocol':'20% random originally-observed-only masking','test_normalized_mae':full['normalized_mae'],'test_normalized_rmse':full['normalized_rmse'],'gpu_epoch_seconds_total':sum(r['epoch_seconds'] for r in history),'peak_cuda_allocated_mib':max(r['peak_cuda_allocated_mib'] for r in history),'peak_cuda_reserved_mib':max(r['peak_cuda_reserved_mib'] for r in history),'coverage_90_noise_interval':coverage['empirical_coverage'],'baselines':primary,'sepsis':sepsis['methods'],'ablations':ablations['contrasts'],'prior_experiments_hashes_preserved':True,'best_checkpoint':str(STUDY/'models'/'chronogan_best.pt'),'last_checkpoint':str(STUDY/'checkpoints'/'last.pt')}
    atomic_json(root/'research-summary.json',summary)
    environment=read('training-audit.json')
    summary['environment']={key:environment[key] for key in ['python_version','torch_version','cuda_runtime','gpu','gpu_total_mib']}
    summary['deployment']=selection
    summary['deployment_checkpoint']=str(STUDY/selection['checkpoint'])
    summary['selected_test_normalized_mae']=selected['normalized_mae']
    summary['selected_test_normalized_rmse']=selected['normalized_rmse']
    summary['component_evidence']=component_evidence
    atomic_json(root/'research-summary.json',summary)
    lines=['# ChronoGAN full-cohort research study','',f'Frozen patients: {summary["patients"]}. Seed 42; batch 64; BF16 autocast with FP32 losses; Adam 0.001; maximum 50 epochs; validation early stopping patience 7.',f'Main run completed {len(history)} epochs. Best epoch: {summary["best_epoch"]}. Best validation standardized MAE: {state["best"]:.6f}. Main training/validation epoch time total: {summary["gpu_epoch_seconds_total"]/3600:.3f} hours (excludes initial preprocessing).',f'Peak PyTorch allocation/reservation: {summary["peak_cuda_allocated_mib"]:.2f}/{summary["peak_cuda_reserved_mib"]:.2f} MiB. Whole-device telemetry is in ../gpu-monitor.csv and includes desktop applications.',f'Best checkpoint: `{summary["best_checkpoint"]}`; resumable last checkpoint: `{summary["last_checkpoint"]}`.',f'Prior CPU, GPU smoke and BF16 trial artifact hashes were checked and preserved.','','## Held-out reconstruction','',f'Test cohort: 6,051 patients. Primary protocol: 20% random artificial masking of observed values only. Identical masks were shared by every method. ChronoGAN MAE/RMSE: {full["normalized_mae"]:.6f}/{full["normalized_rmse"]:.6f} in training-standardized units. Native-unit errors are reported per variable; a scalar mixing clinical units is intentionally not reported.','','| Method | Standardized MAE | Standardized RMSE |','|---|---:|---:|']
    lines.extend(f'| {r["method"]} | {r["normalized_mae"]:.6f} | {r["normalized_rmse"]:.6f} |' for r in primary)
    lines+=['',f'Environment: Python {environment["python_version"]}; PyTorch {environment["torch_version"]}; CUDA runtime {environment["cuda_runtime"]}; {environment["gpu"]}, {environment["gpu_total_mib"]:.0f} MiB device memory.']
    lines+=['','## Dataset and architecture','','PhysioNet Sepsis Challenge 2019 training cohorts A/B contain 40,336 verified unique patient files and 1,552,210 hourly records. Official-source file sizes and MD5/ETag checks are retained in data/official-s3-manifest.json and data/full-download-verification.json. The patient split is frozen before training; normalization uses only training patients. Original source missing values remain unchanged. Thirty-four physiological/laboratory variables are standardized; SepsisLabel is never an imputation input. Nonoverlapping 24-hour windows reset recurrent state and elapsed gaps at boundaries. Padding is excluded from losses and metrics.','The generator combines a causal mask/time-decay encoder, hidden width 64, a causal GRU and 16-dimensional per-hour noise; known observations are composited exactly. A causal two-layer convolutional GAIN-style discriminator uses partial mask hints. Generator objective weights are reconstruction 10, adversarial 0.1, temporal 0.01 and plausibility 0.001. Each update hides 20% of known training cells. The same fixed validation masking rule selects the best checkpoint. No test metrics determine stopping, hyperparameters or model selection.']
    lines+=['','Mean, forward fill, retrospective linear interpolation, a trained causal mask/gap-aware GRU and BRITS were evaluated on matched masks. BRITS is a reconstruction-only adaptation of bidirectional RITS with feature regression, temporal decay, observed-value estimation losses and directional consistency. It omits the paper classification head, uses these 34 features and padded 24-hour windows, and selects by validation masked MAE. BRITS and linear interpolation are retrospective and excluded from prospective prediction. Reference: https://arxiv.org/abs/1805.10572', 'Random/block protocols use 10/20/40/60% rates. A feature-dependent protocol hides vitals at 10% and labs at 40%. A shared contiguous laboratory-panel dropout is an acquisition stress scenario, not a clinically validated missingness mechanism.', '', '## Noise interval calibration','',f'At nominal 90% mass, ChronoGAN empirical coverage was {coverage["empirical_coverage"]:.2%}; mean standardized interval width was {coverage["mean_normalized_width"]:.6f}. Intervals are 20-draw generator noise quantiles, not validated clinical confidence intervals. Coverage is diagnostic; no test-set recalibration was fitted. Full nominal 50/80/90/95% diagnostics and native widths are saved in uncertainty.json.','','## Downstream prediction','','| Method | AUROC | AUPRC | Sensitivity | Specificity | Brier |','|---|---:|---:|---:|---:|---:|']
    for method,metrics in sepsis['methods'].items(): lines.append('| '+method+' | '+' | '.join(f'{metrics[k]:.6f}' for k in ['auroc','auprc','sensitivity','specificity','brier'])+' |')
    lines+=['','Heads use the same frozen patients, seed, hidden width, batch size, Adam learning rate, maximum epochs and validation-BCE early stopping. All four heads use the identical forward GRU with measurement, original visibility mask and log time-gap channels. Only measurement filling differs: zero-filled raw, mean, forward fill or causal ChronoGAN. No BRITS or retrospective interpolation enters prediction. No prospective input uses future measurements or labels. Challenge labels already encode the six-hour shift. The fixed classification threshold is 0.5. Hourly pooled discrimination/calibration metrics do not establish clinical utility.','','## Prespecified ablations','','| Ablation | Ablated minus full MAE | Approximate familywise-95% patient-bootstrap interval |','|---|---:|---|']
    for r in ablations['contrasts']:
        lo,hi=r['patient_bootstrap_bonferroni_interval']; lines.append(f'| {r["variant"]} | {r["ablated_minus_full_normalized_mae"]:.6f} | [{lo:.6f}, {hi:.6f}] |')
    lines+=['','Positive differences favor the full architecture on this reconstruction protocol. The three prespecified contrasts use 2,000 paired patient-cluster bootstrap resamples with Bonferroni quantiles. Intervals cover patient sampling, not training-seed variation. Explicit encoder mask channels and mask-conditioned input decay are removed in no_mask; visibility masks remain for observed-value compositing and GAN supervision. no_delta removes elapsed-time inputs/decay. no_adversarial sets generator adversarial weight to zero while retaining the same denoising and regularization losses. All runs share data/optimizer/epoch ceiling/patience; selected epoch counts can differ.','','## Limitations and remaining research','','One training seed, one hospital-mixed patient split, 24-hour reset windows, artificial observation-only masking and acquisition stress assumptions limit generalization. Real missing values have no ground truth. Repeated seeds, site-held-out/external validation, missing-not-at-random sensitivity analysis, prospective utility/threshold validation and independent uncertainty calibration remain necessary. Lower reconstruction error does not prove clinical benefit. All generated values are research estimates, not verified clinical observations.']
    performance=STUDY.parent/'performance-20261022'/'report.json'
    if performance.exists():
        measured=json.loads(performance.read_text())
        summary['performance_benchmark']={key:measured[key] for key in ['speedup_percent','correctness','recommendation','comparability']}
        atomic_json(root/'research-summary.json',summary)
        lines+=['','## Execution performance study','',f'Matched benchmark replay epoch time reduction: {measured["speedup_percent"]:.2f}%. Safe optimized path recommendation: {measured["recommendation"]["performance_optimized"]}. This compares isolated no_mask replay epochs, not different ablation architectures. Larger batches and fused Adam were not applied to scientific training. Original full/no_mask runs were not interrupted.',measured['comparability'],'Each later checkpoint records its actual execution settings; execution-config.json also records an enabled optimization. Detailed measurements are in ../../performance-20261022/report.md.']
    lines+=['','## Validation-selected application model','',f'Selected causal method: **{selection["selected_method"]}**. Common validation 20-draw-mean criterion MAE: {selection["validation_normalized_mae"]:.6f}. This is a separate validation protocol from training-checkpoint single-draw selection; all causal candidates shared masks and no test metric was consulted. The method need not be the full GAN.',f'Packaged checkpoint: `{summary["deployment_checkpoint"]}`. Original full ChronoGAN checkpoint is preserved separately. Selected method held-out primary MAE/RMSE: {selected["normalized_mae"]:.6f}/{selected["normalized_rmse"]:.6f}. Deterministic winners have no uncertainty intervals; the full stochastic model remains available as a separate comparison.','','## Component evidence beyond pointwise MAE','',f'Full GAN minus non-adversarial primary test MAE: {full["normalized_mae"]-nonadversarial["normalized_mae"]:.6f}; a positive value favors the non-adversarial variant. Lower full-model MAE is not assumed. Single-draw training validation had already favored the no-gap and no-adversarial variants. Weight competition, optimization variance and unnecessary representation complexity are possible explanations, not established causes.',component_evidence['interpretation'],'Temporal first-difference errors, individual block rates, interval coverage/width and collapsed-variation fractions are provided in component-benefits.json and uncertainty.json. No claim of trajectory realism is made without independent assessment.']
    advantages=[]
    if full['normalized_mae']<nonadversarial['normalized_mae']: advantages.append('lower primary pointwise MAE')
    if full.get('temporal_normalized_mae') is not None and full['temporal_normalized_mae']<nonadversarial['temporal_normalized_mae']: advantages.append('lower primary temporal first-difference MAE')
    block_wins=[]
    for rate in [.1,.2,.4,.6]:
        block_full=next(row for row in evaluation['experiments'] if row['method']=='chronogan' and row['pattern']=='block' and row['rate']==rate)
        block_without=next(row for row in evaluation['experiments'] if row['method']=='no_adversarial' and row['pattern']=='block' and row['rate']==rate)
        if block_full['normalized_mae']<block_without['normalized_mae']: block_wins.append(rate)
    if block_wins: advantages.append('lower block MAE at hiding rates '+str(block_wins))
    without_coverage=next(row for row in uncertainty['results'] if row['method']=='no_adversarial' and row['pattern']=='random' and row['rate']==.2 and row['nominal_coverage']==.9)
    if abs(coverage['empirical_coverage']-.9)<abs(without_coverage['empirical_coverage']-.9): advantages.append('primary nominal-90% coverage closer to 90%, with width tradeoffs reported separately')
    conclusion='Descriptive full-versus-nonadversarial advantages: '+('; '.join(advantages) if advantages else 'none on the prespecified primary point, temporal, block and nominal-90% coverage diagnostics')+'. No training-seed replication or clinical effectiveness is established.'
    lines+=['',conclusion,'Hourly source rows do not establish exact operational measurement-availability timestamps; prospective availability and external utility require separate validation.']
    summary['adversarial_component_conclusion']=conclusion
    atomic_json(root/'research-summary.json',summary)
    (root/'research-summary.md').write_text('\n'.join(lines),encoding='utf-8')
    return summary

if __name__=='__main__': print(json.dumps(generate(),indent=2))
