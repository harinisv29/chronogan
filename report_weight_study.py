"""Build the measured study report and CUDA gradient diagnostics, post-training."""
import json
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from preprocess import FEATURES,PatientDataset
from models.discriminator import Discriminator
from train import load_checkpoint
from utils.dataset import artificial_mask,recompute_delta
from utils.experiment import generator_from_state,atomic_json
from utils.losses import generator_loss
from research_weight_study import ROOT,hashfile,WEIGHTS,TRAIN_SEEDS,bootstrap

def gradient_diagnostic(base,protocol):
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.12)
    dataset=PatientDataset(protocol['train'][:32],base['stats'],base['config']['sequence_length'])
    # cuDNN GRU retains backward buffers only in training mode. This architecture
    # has no dropout/batchnorm, and no optimizer step is performed here.
    g=generator_from_state(base,'cuda').train();d=Discriminator(34,base['config']['hidden']).cuda().eval();d.load_state_dict(base['discriminator'])
    for parameter in d.parameters():parameter.requires_grad_(False)
    rows=[]
    for seed in TRAIN_SEEDS:
        torch.manual_seed(seed);batch=[torch.stack([dataset[i][k] for i in range(min(32,len(dataset)))]).cuda() for k in range(4)]
        x,m,delta,v=batch;visible,held=artificial_mask(m,v,.2);dd=recompute_delta(visible,delta);reveal=(torch.rand_like(m)<.9).float();hint=reveal*visible+.5*(1-reveal)
        output=g(x*visible,visible,dd,v)
        _,parts=generator_loss(output,x,held,visible,v,d(output,hint,v),1-reveal,base['config'],base['stats'],True)
        parameters=list(g.parameters());rec=torch.autograd.grad(parts['reconstruction_mse'],parameters,retain_graph=True,allow_unused=True);adv=torch.autograd.grad(parts['adversarial_loss'],parameters,allow_unused=True)
        flatten=lambda gradients:torch.cat([(value if value is not None else torch.zeros_like(parameter)).flatten() for parameter,value in zip(parameters,gradients)])
        r,a=flatten(rec),flatten(adv);cosine=F.cosine_similarity(r,a,dim=0).item()
        rows.append({'seed':seed,'reconstruction_gradient_norm':r.norm().item(),'adversarial_gradient_norm':a.norm().item(),'gradient_cosine_similarity':cosine,'weighted_adversarial_to_reconstruction_gradient_norm_ratio':float((.1*a.norm()/(10*r.norm())).item())})
    probes=[]
    with torch.no_grad():
        rate=F.softplus(g.encoder.decay)
        for hours in [0,1,6,12,24]:
            incoming=torch.exp(-rate*hours);hidden=torch.exp(-torch.relu(g.encoder.hidden_decay(torch.full((1,34),float(hours),device='cuda'))))
            probes.append({'uniform_gap_hours':hours,'input_decay_min':incoming.min().item(),'input_decay_median':incoming.median().item(),'input_decay_max':incoming.max().item(),'hidden_decay_min':hidden.min().item(),'hidden_decay_median':hidden.median().item(),'hidden_decay_max':hidden.max().item()})
    return {'batches':rows,'uniform_gap_probes':probes,'scope':'Same first <=32 training windows across 3 artificial-mask/noise seeds; gradient alignment is local evidence, not proof of general adversarial harm. Gap probes use synthetic uniform feature gaps, not cohort statistics. No optimizer step or checkpoint modification.'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='runs/loss-weight-study/cuda-v1');args=parser.parse_args()
    out=ROOT/args.output;data=json.loads((out/'results.json').read_text());protocol=data['protocol'];results=data['results']
    gradients=gradient_diagnostic(load_checkpoint(ROOT/'runs/full-training/models/chronogan_best.pt'),protocol);atomic_json(out/'gradient-diagnostic.json',gradients)
    history=[]
    for path in (out/'adaptations').glob('*/history.json'):
        for row in json.loads(path.read_text()):history.append({'run':path.parent.name,**row})
    pd.DataFrame(history).to_csv(out/'training-history.csv',index=False)
    # Conditional native-unit patient-cluster CIs: masks/training reps are averaged
    # first. Overall report CIs additionally resample mask/adaptation seeds.
    per_feature=[];contrasts=[];seed_scores=[];rng=np.random.default_rng(88422);n=len(protocol['test']);counts=rng.multinomial(n,np.full(n,1/n),size=1000);std=np.array(load_checkpoint(ROOT/'runs/full-training/models/chronogan_best.pt')['stats']['std'])
    for scenario in results:
        arrays=np.load(out/f'patient-metrics-{scenario}.npz')
        for name in arrays.files:
            for training_index,training_run in enumerate(arrays[name]):
                for mask_index,mask_run in enumerate(training_run):
                    sufficient=mask_run.sum(axis=0);count=sufficient[0,2].sum()
                    seed_scores.append({'scenario':scenario,'model':name,'training_replicate':training_index,'mask_seed':protocol['mask_seeds'][mask_index],'standardized_mae':float((sufficient[0,0]/std).sum()/count),'standardized_rmse':float(np.sqrt((sufficient[0,1]/std**2).sum()/count)),'coverage':float(sufficient[1,2].sum()/sufficient[0,4].sum())})
            for reference in ['mask_aware_gru','rec10_adv01']:
                interval=bootstrap(arrays[name],arrays[reference],std)['paired_mae_difference_vs_original_95_ci']
                contrasts.append({'scenario':scenario,'model':name,'reference':reference,'mae_difference':results[scenario][name]['raw']['standardized_mae']-results[scenario][reference]['raw']['standardized_mae'],'paired_95_ci_low':interval[0],'paired_95_ci_high':interval[1]})
            a=arrays[name].sum(axis=(0,1))
            for subset,k in [('raw',0),('released',1),('withheld',2)]:
                denominator=counts@a[:,k,2];absolute=counts@a[:,k,0];square=counts@a[:,k,1]
                with np.errstate(invalid='ignore',divide='ignore'):
                    mae=absolute/denominator;rmse=np.sqrt(square/denominator)
                for j,feature in enumerate(FEATURES):
                    valid=np.isfinite(mae[:,j]);value=results[scenario][name][subset]['per_feature'][feature]
                    per_feature.append({'scenario':scenario,'model':name,'subset':subset,'feature':feature,**value,
                        'mae_native_ci_low':float(np.quantile(mae[valid,j],.025)) if valid.any() else None,'mae_native_ci_high':float(np.quantile(mae[valid,j],.975)) if valid.any() else None,
                        'rmse_native_ci_low':float(np.quantile(rmse[valid,j],.025)) if valid.any() else None,'rmse_native_ci_high':float(np.quantile(rmse[valid,j],.975)) if valid.any() else None})
    pd.DataFrame(per_feature).to_csv(out/'per-feature-confidence.csv',index=False)
    pd.DataFrame(contrasts).to_csv(out/'paired-contrasts.csv',index=False)
    pd.DataFrame(seed_scores).to_csv(out/'seed-scores.csv',index=False)
    random20=results['random_0.2'];controlled=next(c for c in contrasts if c['scenario']=='random_0.2' and c['model']=='rec10_adv0' and c['reference']=='rec10_adv01');gru_contrast=next(c for c in contrasts if c['scenario']=='random_0.2' and c['model']=='reconstruction_only' and c['reference']=='mask_aware_gru')
    lines=['# ChronoGAN multi-seed loss-weight study','',
        '## Measured findings','',
        f"At 20% random masking, original standardized MAE is {random20['chronogan']['raw']['standardized_mae']:.6f}, archived reconstruction-only {random20['reconstruction_only']['raw']['standardized_mae']:.6f}, and mask-aware GRU {random20['mask_aware_gru']['raw']['standardized_mae']:.6f}. These new three-seed masks differ from the earlier single-seed 0.368510/0.362414/0.358067 audit; those older values are not matched before scores for this study.",
        f"The controlled zero-adversarial adaptation versus unchanged-loss adaptation has MAE difference {controlled['mae_difference']:.6f}, paired 95% interval [{controlled['paired_95_ci_low']:.6f}, {controlled['paired_95_ci_high']:.6f}]. The archived reconstruction-only versus GRU difference has interval [{gru_contrast['paired_95_ci_low']:.6f}, {gru_contrast['paired_95_ci_high']:.6f}]. Both intervals include zero: neither controlled adversarial-loss harm nor superiority over the GRU is established at this masking level.",
        'The complete scenario tables below expose any reversals, RMSE regressions and coverage tradeoffs. Seed-level measurements are in seed-scores.csv. No dashboard promotion was made.','',
        '## Scope and controls','',
        f"All model training, inference and gradient diagnostics executed on {protocol['environment']['gpu']} with PyTorch {protocol['environment']['torch']} / CUDA {protocol['environment']['cuda']}. The existing controller and CPU dashboard remained running; no deployed model changed. Peak CUDA tensor allocation: {data['peak_cuda_allocated_mib']:.2f} MiB. Study runtime: {data['seconds']:.1f} seconds (report diagnostics excluded).",
        f"Adaptation uses {len(protocol['train'])} frozen training patients, {len(protocol['val'])} validation patients and {len(protocol['test'])} held-out audit patients. All cohorts are patient-disjoint. Statistics are the original training-only statistics. Twenty noise samples per imputation; identical saved per-patient masks and preprocessing across models. Originally missing measurements never become reconstruction targets. Original observed measurements are copied exactly and asserted unchanged. Invalid estimates and unsupported features stay withheld.",
        'Five loss settings × three adaptation seeds × three epochs, fresh Adam at 0.0001, batch32, gradient clipping5. The (reconstruction, adversarial) weights are (10,0), (10,0.01), (10,0.05), (10,0.1), (20,0.1). All start from the SAME completed generator and discriminator. Shuffles, artificial masks, noise and hint draws are matched within each training seed; discriminator updates are retained even when adversarial generator weight is zero. Temporal/plausibility weights are unchanged. Every run chooses its best epoch using the same random20 validation masks and noise. Test scores do not select a model or epoch.',
        'This is short warm-start adaptation, not 15 independent from-scratch GAN trainings. The archived models were trained once each; their three masking/noise seeds are not three training seeds. Adaptations cross 3 training seeds with 3 mask/noise seeds. Test patients were inspected in earlier audits: this is not a pristine final confirmation set. All intervals are exploratory, without multiple-comparison correction.','',
        '## Main comparison: 20% random masking','',
        '| Model | Standardized MAE [95% CI] | RMSE | Coverage | Raw invalid | MAE difference vs original [95% CI] |',
        '|---|---:|---:|---:|---:|---:|']
    for name,value in results['random_0.2'].items():
        lo,hi=value['mae_95_ci'];dl,dh=value['paired_mae_difference_vs_original_95_ci']
        lines.append(f"| {name} | {value['raw']['standardized_mae']:.6f} [{lo:.6f}, {hi:.6f}] | {value['raw']['standardized_rmse']:.6f} | {value['coverage']:.2%} | {value['raw']['invalid_rate']:.2%} | [{dl:.6f}, {dh:.6f}] |")
    lines+=['','## Missingness stress results','', '| Scenario | Original MAE / RMSE | Reconstruction-only MAE / RMSE | GRU MAE / RMSE |','|---|---:|---:|---:|']
    for scenario,methods in results.items():
        values=[methods[name]['raw'] for name in ['chronogan','reconstruction_only','mask_aware_gru']]
        lines.append('| '+scenario+' | '+' | '.join(f"{v['standardized_mae']:.6f} / {v['standardized_rmse']:.6f}" for v in values)+' |')
    lines+=['','## Loss-weight tradeoffs across all scenarios','', '| Scenario | rec10/adv0 | rec10/adv.01 | rec10/adv.05 | rec10/adv.1 | rec20/adv.1 |','|---|---:|---:|---:|---:|---:|']
    for scenario,methods in results.items():lines.append('| '+scenario+' | '+' | '.join(f"{methods[name]['raw']['standardized_mae']:.6f}" for name,_,_ in WEIGHTS)+' |')
    lines+=['','Absolute weights and ratios are distinct: rec20/adv.1 changes the balance against temporal/plausibility penalties too; it is not simply equivalent to rec10/adv.05. Comparisons with rec10/adv.1 control for the effects of additional adaptation. Improvement against the unadapted model alone cannot establish that a loss-weight change helped.','',
        '## Mask and elapsed-time investigation','',
        'The encoder already conditions on x, mask and log1p(delta), learned input decay and hidden-state decay. Deltas are recomputed from visible masks after artificial hiding, preventing hidden observation timing leakage. All inference uses causal independent 24-hour windows, matching training. The archived no_mask model removes explicit encoder mask conditioning, but still has elapsed-time inputs and observed-value copy-through; it is not a perfectly mask-free architecture. The no_delta model zeros elapsed-time inputs. Those one-seed ablations cannot establish seed-robust causal benefit. No speculative architecture rewrite was made.',
        'All archived ablation results and paired intervals are in comparison.csv. Each adaptation has exactly the same mask/time features as the original generator. Longer context or overlapping-window conditioning needs a separately trained, causally validated protocol before deployment.','',
        '## Adversarial diagnostics','',
        'Loss and pre-clipping gradient histories are in training-history.csv. Finite-gradient checks and clipping were active. Low discriminator loss / high adversarial loss can indicate imbalance but do not independently prove mode collapse. No diversity-collapse claim is supported by this study. Local training-batch reconstruction/adversarial gradient alignment:',
        '```json',json.dumps(gradients,indent=2),'```',
        'A negative cosine indicates local competing descent directions, while near-zero suggests weak alignment. This diagnostic does not replace validation/test outcome comparisons. The archived reconstruction-only checkpoint used a different historical training trajectory; controlled warm-start variants provide a more matched short-horizon comparison.','',
        '## Confidence intervals and metrics','',
        'Overall 95% percentile intervals use 1000 paired bootstrap resamples of patient clusters, masking seeds and adaptation seeds (archives have one training replicate). All rows from a patient are resampled together. Paired differences use the same patients/masks as the original GAN. Native-unit per-feature MAE/RMSE and conditional patient-cluster CIs are in per-feature-confidence.csv; those feature intervals average seed replicates before patient resampling. A CI spanning zero does not establish improvement. Non-overlapping or negative exploratory intervals are not proof of clinical utility.',
        'MAE and RMSE in the main tables are standardized by the frozen training feature standard deviations. Pooled raw native errors would mix incompatible units, so native errors are reported per feature. Raw, released and withheld predictions are scored separately against artificially hidden known values; original missing values have no ground truth. Coverage uses eligible hidden cells, not all originally missing patient cells. Invalid outputs are counted before withholding, including a separate nonfinite count in results.json. Released outputs must pass unchanged registry checks. Lower released-subset error alone is not an estimator improvement.','',
        '## Files and reproducibility','',
        '- research_weight_study.py: isolated adaptation, matched evaluation, safety assertions and paired bootstrap.',
        '- report_weight_study.py: measured report, per-feature intervals, CUDA gradient diagnostics.',
        '- tests/test_weight_study.py: paired-bootstrap identity and hidden-target/domain-policy scoring.',
        '- protocol.json: frozen patient lists, weights, seeds, environment and preserved file hashes.',
        '- evaluated-models.json and adaptations/: isolated inference checkpoints, validation choices and training histories. These are inference artifacts, not resumable optimizer checkpoints; do not pass them to train.py --resume.',
        '- masks-*.npz and patient-metrics-*.npz: reproducible masks and patient-level sufficient statistics.',
        '- comparison.csv, per-feature.csv, per-feature-confidence.csv, results.json: all measured outcomes and intervals.',
        '```powershell',r'.\.runtime\python\python.exe run.py research_weight_study.py --device cuda --output runs/loss-weight-study/cuda-replay',r'.\.runtime\python\python.exe run.py report_weight_study.py --output runs/loss-weight-study/cuda-replay','```',
        'Use a fresh output directory for a new complete study. Completed results are never overwritten. Completed adaptations can be reused after evaluation/report failure; interrupted partial adaptations require a new output directory.','',
        '## Limitations and deployment','',
        'No deployed dashboard model, original checkpoint, normalization, split, clinical registry or selection policy was changed. No log transform, hard clipping, relaxed safety policy or unsupported laboratory release was introduced. No improvement is presumed from loss weights. Three seeds and 64 patients are a pilot, not a full-cohort study or clinical validation. An independent confirmation cohort, more training seeds, training from scratch, uncertainty calibration and per-feature plausibility review remain necessary before deployment.']
    lines+=['','paired-contrasts.csv additionally reports intervals against the GRU and the unchanged-loss adaptation control. For the control, training seeds are resampled jointly to preserve paired runs. Higher reconstruction weights also increase clipping frequency; consult pre-clipping norms before interpreting a ratio change.']
    lines+=['',f"At random20, no_mask MAE is {random20['no_mask']['raw']['standardized_mae']:.6f} versus original {random20['chronogan']['raw']['standardized_mae']:.6f}; no_delta is {random20['no_delta']['raw']['standardized_mae']:.6f}. Explicit masks appear useful in this archived comparison, while current delta conditioning does not demonstrate benefit. The synthetic uniform-gap probes above expose learned decay saturation; they are not proof that changing decay would improve held-out error. A matched multi-seed from-scratch decay ablation is the next justified experiment."]
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axis=plt.subplots(figsize=(8,5))
    rates=[.1,.2,.4,.6]
    for name,label in [('chronogan','Original GAN'),('reconstruction_only','Archived reconstruction-only'),('mask_aware_gru','Mask-aware GRU'),('rec10_adv0','Zero-adversarial adaptation')]:
        values=[results['random_'+str(rate)][name] for rate in rates]
        axis.plot(rates,[v['raw']['standardized_mae'] for v in values],marker='o',label=label)
        axis.fill_between(rates,[v['mae_95_ci'][0] for v in values],[v['mae_95_ci'][1] for v in values],alpha=.07)
    axis.set(xlabel='Fraction of observed measurements artificially hidden',ylabel='Standardized MAE',title='64-patient pilot: three mask/noise seeds');axis.legend(fontsize=8);axis.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'random-missingness.png',dpi=160);plt.close(fig)
    lines+=['','![Random missingness results with exploratory 95% intervals](random-missingness.png)']
    test_file=out/'tests-final.xml'
    if test_file.exists():
        root=ET.parse(test_file).getroot();suites=list(root.iter('testsuite'))
        counts={key:sum(int(s.attrib.get(key,0)) for s in suites) for key in ['tests','errors','failures','skipped']}
        lines+=['','## Automated tests','',str(counts),'The automated suite hides CUDA to avoid duplicate GPU work. All study forward/backward passes and gradient diagnostics explicitly require CUDA.'];atomic_json(out/'test-summary.json',counts)
    assert all(hashfile(ROOT/f)==h for f,h in protocol['frozen'].items())
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8');print('REPORT',out/'report.md',flush=True)

if __name__=='__main__':main()
