"""Export native-unit results and an honest diagnosis for the isolated experiment."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import xml.etree.ElementTree as ET
from predict import reconstruct
from preprocess import FEATURES,read_patient
from reconstruction_experiment import ROOT,write,digest

def main():
    out=ROOT/'runs/reconstruction-improvement/cuda-v1';report=json.loads((out/'results.json').read_text());candidate=json.loads((out/'archived-no-adversarial/results.json').read_text())
    training_config=yaml.safe_load((ROOT/'config.full-training.yaml').read_text())
    training_config.update(adversarial=0.,run_dir='runs/reconstruction-improvement/retrain-no-adversarial',cache_dir='runs/reconstruction-improvement/retrain-no-adversarial/cache')
    (out/'isolated-training.yaml').write_text(yaml.safe_dump(training_config,sort_keys=False))
    checkpoint=ROOT/'runs/full-training/ablations/no_adversarial/models/chronogan_best.pt'
    source=ROOT/'runs/inference-audit/20261022/synthetic-input.csv';frame=read_patient(source)
    baseline_result,_,_,baseline_metadata=reconstruct(frame,ROOT/'runs/full-training/models/chronogan_best.pt',samples=20,device='cuda',seed=42)
    baseline_result.to_csv(out/'baseline-synthetic-reconstruction.csv',index=False);write(out/'baseline-synthetic-metadata.json',baseline_metadata)
    reconstructed,mask,uncertainty,metadata=reconstruct(frame,checkpoint,samples=20,device='cuda',seed=42)
    for name in FEATURES:
        known=frame[name].notna();np.testing.assert_array_equal(frame.loc[known,name],reconstructed.loc[known,name])
    reconstructed.to_csv(out/'candidate-synthetic-reconstruction.csv',index=False);uncertainty.to_csv(out/'candidate-synthetic-uncertainty.csv',index=False);write(out/'candidate-synthetic-metadata.json',metadata)
    rows=[];feature_rows=[]
    for scenario,methods in report['results'].items():
        methods={**methods,'no_adversarial_candidate':candidate['results'][scenario]}
        for method,result in methods.items():
            rows.append({'scenario':scenario,'method':method,'standardized_mae':result['raw']['standardized_mae'],'standardized_rmse':result['raw']['standardized_rmse'],'iqr_normalized_mae':result['raw']['normalized_mae'],'coverage':result['coverage'],'withholding_rate':result['withholding_rate'],'raw_invalid_rate':result['raw']['invalid_rate'],'released_mae':result['released']['standardized_mae'],'withheld_mae':result['withheld']['standardized_mae']})
            for subset in ['raw','released','withheld']:
                for name,values in result[subset]['per_feature'].items():feature_rows.append({'scenario':scenario,'method':method,'subset':subset,'feature':name,**values})
    pd.DataFrame(rows).to_csv(out/'all-comparisons.csv',index=False);pd.DataFrame(feature_rows).to_csv(out/'per-feature-errors.csv',index=False)
    original=frame[FEATURES].to_numpy(dtype=float);observed=np.isfinite(original);missing=~observed
    no_patient_history=missing&~observed.any(0)[None,:];no_window_history=np.zeros_like(missing)
    for start in range(0,len(original),24):no_window_history[start:start+24]=missing[start:start+24]&(np.cumsum(observed[start:start+24],axis=0)==0)
    old=baseline_metadata['output_validation']
    breakdown={'missing':int(missing.sum()),'entire_feature_unobserved':int(no_patient_history.sum()),'no_prior_observation_in_window':int(no_window_history.sum()),
        'no_prior_window_anchor_but_observed_elsewhere':int((no_window_history&~no_patient_history).sum()),'fixed_seed_current_model_raw_invalid':old['raw_invalid_count'],
        'fixed_seed_current_model_withheld':old['withheld_count'],'fixed_seed_current_model_released':int(missing.sum())-old['withheld_count'],
        'candidate':metadata['output_validation'],'ood_policy':'Training-z outliers warn, not automatically withhold. Unknown upload units remain a warning. Categories overlap; invalid and unsupported counts must not be added.'}
    write(out/'coverage-analysis.json',breakdown)
    lines=['# ChronoGAN bounded CUDA improvement experiment','',
        'This is a limited diagnostic audit, not final clinical validation. It does not alter the active research controller, deployment selection, weights, statistics or application behavior. Previously inspected test patients are not a pristine confirmation cohort.','',
        '## Investigation before edits','',
        '- Architecture: causal mask/delta-aware temporal encoder, GRU generator and causal convolutional discriminator. Masks and deliberately hidden observed targets are used correctly; original NaNs are not reconstruction targets.',
        '- Confirmed previous inference bug: full sequences differed from training 24-hour independent reset windows. Already fixed before this task; not repeated or attributed to this experiment.',
        '- Feature ordering, training-only statistics and affine inverse match. Negative AST/TroponinI are unrestricted generator outputs, amplified by large training standard deviations; not an inverse-transform bug.',
        '- Original training has gradient clipping at 5, fixed-mask validation, early stopping and finite saved weights/losses. Low discriminator loss and large adversarial loss motivate testing its effect, but do not prove collapse or exploding gradients.',
        '- Training includes fixed soft clinical bounds for some variables, absent for AST/TroponinI. These are not universal clinical truths. No new arbitrary clipping or training-bound edits were made.',
        '- Current 24-hour resets are consistent but lose cross-window history. Overlapping windows alter conditioning; future observations cannot be used in a causal demo. No overlap change was installed without validation.',
        '- Unit conventions for arbitrary uploads are not verified. FiO2 remains ambiguous. Applying log1p to inputs of existing weights is incompatible: any transform needs training-only refitting, new weights and independent validation.',
        '- Coverage is mainly a release-policy limitation under sparse feature history. Unsupported/unanchored estimates remain withheld; no safety check was disabled. OOD warnings are not evidence of reliable predictions.',
        '- CSV/PDF/visualization preserve blank withheld values, raw predictions/reasons and original observations. This task changes experimental tooling only.','',
        '## Frozen protocol','',
        '512 training patients for median/IQR only; 64 validation patients for scalar blend and archived-ablation selection; 64 frozen test patients for diagnosis. No patient overlap. All model forward passes use CUDA float32, 20 noise samples, per-process 12% VRAM cap. Identical saved masks for every method. Training-only IQR-normalized MAE differs from standardized MAE. Native MAE/RMSE are reported per feature because pooling native units is meaningless.',
        'Random 10%, 20%, 40%; per-feature contiguous intervals at 20% sequence length; whole-feature 20% stress. The last is an expected abstention case. Linear interpolation uses future values and is labeled retrospective, never a causal deployment candidate.','',
        '## Controlled interventions','',
        f"1. Causal-forward blending rejected on validation: {report['selection']['validation_standardized_mae']}. Chosen GAN weight stays 1.0.",
        f"2. Reused completed single-factor zero-adversarial ablation without retraining: validation {candidate['selection']['baseline_validation_mae']:.6f} -> {candidate['selection']['candidate_validation_mae']:.6f}. Selection frozen before candidate test scoring; separate from the controller's final deployment selection.",'',
        '| Mask | Original MAE / RMSE | Candidate MAE / RMSE | Original / candidate coverage |','|---|---:|---:|---:|']
    for scenario,methods in report['results'].items():
        before=methods['chronogan'];after=candidate['results'][scenario]
        lines.append(f"| {scenario} | {before['raw']['standardized_mae']:.6f} / {before['raw']['standardized_rmse']:.6f} | {after['raw']['standardized_mae']:.6f} / {after['raw']['standardized_rmse']:.6f} | {before['coverage']:.2%} / {after['coverage']:.2%} |")
    lines+=['','## Matched baseline comparison at random 20%','', '| Method | Standardized MAE | RMSE | Coverage |','|---|---:|---:|---:|']
    methods={**report['results']['random_0.2'],'no_adversarial_candidate':candidate['results']['random_0.2']}
    for method,result in methods.items():lines.append(f"| {method} | {result['raw']['standardized_mae']:.6f} | {result['raw']['standardized_rmse']:.6f} | {result['coverage']:.2%} |")
    lines+=['','The candidate improves aggregate raw error here but does not outperform the mask-aware GRU. Per-feature improvements are not universal. No significance or clinical benefit claim is supported by this small single-seed audit. These CUDA masks differ from the earlier CPU audit: the old 0.384330 is not a matched before value for this experiment.','',
        '## Coverage and safety','',
        f"Supplied patient: {breakdown['missing']} missing; {breakdown['entire_feature_unobserved']} belong to entirely unobserved features; {breakdown['no_prior_observation_in_window']} have no prior in-window anchor, including {breakdown['no_prior_window_anchor_but_observed_elsewhere']} observed elsewhere in the patient. Original fixed-seed outputs: {old['raw_invalid_count']} raw invalid, {old['withheld_count']} withheld. Categories overlap.",
        f"Candidate fixed-seed supplied-patient outputs: {metadata['output_validation']['withheld_count']} withheld, {metadata['output_validation']['raw_invalid_count']} raw invalid. All observed measurements exactly preserved; no patient-specific accuracy can be estimated without hidden ground truth.",
        'Whole-feature masking gives zero released coverage by design. Improved raw accuracy does not validate unconditional laboratory estimation. Raw/released/withheld errors and counts are separately exported; retained error alone must not be mistaken for overall estimator improvement.','',
        '## Artifacts and reproducibility','',
        '- `protocol.json`, `selection.json`, `masks.npz`: frozen patient lists, seeds, masks, hashes and validation choice.',
        '- `all-comparisons.csv`, `per-feature-errors.csv`: raw/released/withheld errors, native-unit feature errors, coverage and invalid rates.',
        '- `gap-and-missingness.csv`: patient missingness and consecutive missing-gap strata (1–3, 4–12, 13+ rows); raw standardized absolute sums/counts permit weighted aggregation.',
        '- `coverage-analysis.json`, `candidate-synthetic-*`: supplied patient audit with original observations, raw values, policy reasons and model version.',
        '- `results.json`, `archived-no-adversarial/results.json`: runtime, peak CUDA allocation and preserved hashes.',
        '- New files only: `reconstruction_experiment.py` (evaluation/blend), `compare_archived_candidate.py` (single-factor archived candidate), `report_reconstruction_experiment.py` (report/exports), `tests/test_reconstruction_experiment.py` (mask/causality/metric/real-file fixtures). No app, training or model source edits in this task.','',
        'Run from the project directory, using a fresh output directory for another full evaluation:',
        '```powershell',
        r'.\.runtime\python\python.exe run.py reconstruction_experiment.py --device cuda --output runs/reconstruction-improvement/cuda-replay',
        r'.\.runtime\python\python.exe run.py compare_archived_candidate.py --device cuda --reference runs/reconstruction-improvement/cuda-replay',
        '```',
        'Exact training command for a future isolated retraining study (not launched; the prepared configuration uses a NEW run_dir):',
        '```powershell',r'.\.runtime\python\python.exe run.py train.py --config runs/reconstruction-improvement/cuda-v1/isolated-training.yaml --device cuda','```',
        'Use existing frozen training/validation split and training-only stats. For an exact zero-adversarial ablation set adversarial: 0.0 and keep remaining settings from config.full-training.yaml; never reuse the existing run_dir. Log transforms require newly fit statistics and cannot reuse the unchanged checkpoint.','',
        'Remaining: repeated seeds, independent patient cohort, paired uncertainty intervals, per-feature error tradeoffs, clinical/unit validation, calibrated uncertainty, causal longer-context validation, and carefully isolated domain-aware retraining. Do not increase release coverage for unobserved features from this limited evidence.']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    test_file=out/'tests-final.xml'
    if test_file.exists():
        root=ET.parse(test_file).getroot();suites=[root] if root.tag=='testsuite' else list(root.iter('testsuite'))
        counts={key:sum(int(s.attrib.get(key,0)) for s in suites) for key in ['tests','failures','errors','skipped']}
        write(out/'test-summary.json',counts)
        with (out/'report.md').open('a',encoding='utf-8') as stream:
            stream.write('\n\n## Automated tests\n\n'+str(counts)+'\nCUDA is hidden only for the automated suite to avoid extra GPU work; both model comparison experiments and the matched supplied-patient audit executed on CUDA.\n')
    details=pd.read_csv(out/'gap-and-missingness.csv')
    details['missingness_bin']=pd.cut(details.input_missing_fraction,[0,.5,.7,.8,.9,1.],include_lowest=True).astype(str)
    for filename,keys,data in [('error-by-gap.csv',['scenario','method','gap'],details[details.gap!='all']),('error-by-missingness.csv',['scenario','method','missingness_bin'],details[details.gap=='all'])]:
        grouped=data.groupby(keys,observed=True)[['n','standardized_absolute_sum']].sum().reset_index()
        grouped['standardized_mae']=grouped.standardized_absolute_sum/grouped.n;grouped.to_csv(out/filename,index=False)
    assert all(digest(ROOT/f)==h for f,h in report['protocol']['frozen_hashes'].items())
    print(json.dumps({'report':str(out/'report.md'),'candidate_gpu_device':metadata['inference_device'],'coverage':breakdown['candidate']['withheld_count'],'hashes_preserved':True},indent=2))

if __name__=='__main__':main()
