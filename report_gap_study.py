"""CPU-only report of recorded gap/domain experiments, without model execution."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import json
import argparse
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/gap-feature-study/cuda-v1'

def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='runs/gap-feature-study/cuda-v1');args=parser.parse_args();OUT=ROOT/args.output
    data=json.loads((OUT/'results.json').read_text());results=data['results'];protocol=data['protocol']
    secondary=json.loads((OUT/'frozen-domain/results.json').read_text());rows=[];features=[]
    for scenario,methods in results.items():
        methods={**methods,'frozen_domain_link':secondary['results'][scenario]}
        for name,value in methods.items():
            raw=value['raw'];ci=value['vs_gru'];invalid=sum(v['invalid'] for v in raw['per_feature'].values())
            rows.append({'scenario':scenario,'model':name,'mae':raw['standardized_mae'],'rmse':raw['standardized_rmse'],'coverage':value['coverage'],'withholding_rate':value['withholding_rate'],'raw_invalid_rate':raw['invalid_rate'],'raw_invalid_count':invalid,'raw_scored_count':raw['n'],'mae_ci_low':ci['mae_95_ci'][0],'mae_ci_high':ci['mae_95_ci'][1],'paired_mae_difference_vs_gru_ci_low':ci['paired_mae_difference_vs_original_95_ci'][0],'paired_mae_difference_vs_gru_ci_high':ci['paired_mae_difference_vs_original_95_ci'][1]})
            for subset in ['raw','released','withheld']:
                for f,values in value[subset]['per_feature'].items():features.append({'scenario':scenario,'model':name,'subset':subset,'feature':f,**values})
    pd.DataFrame(rows).to_csv(OUT/'comparison.csv',index=False);pd.DataFrame(features).to_csv(OUT/'per-feature.csv',index=False)
    gaps=pd.read_csv(OUT/'gap-errors.csv');gap_summary=gaps.groupby(['scenario','model','gap'])[['n','absolute_standardized_sum','squared_standardized_sum']].sum().reset_index();gap_summary['mae']=gap_summary.absolute_standardized_sum/gap_summary.n;gap_summary['rmse']=np.sqrt(gap_summary.squared_standardized_sum/gap_summary.n);gap_summary.to_csv(OUT/'gap-summary.csv',index=False)
    history=[]
    for path in (OUT/'models').glob('*/history.json'):
        for row in json.loads(path.read_text()):history.append({'run':path.parent.name,**row})
    pd.DataFrame(history).to_csv(OUT/'training-history.csv',index=False)
    original=results['random_0.2']['mask_aware_gru'];primary_retained=data['retained_for_further_research'];secondary_retained=secondary['retained_for_further_research']
    lines=['# Gap decay, consecutive missingness and feature-domain pilot','',
        '## Outcome','',
        f"Validation eligibility for fine-tuned candidates: {data['eligible_before_test']}. Supported under the predeclared complete retention gate: {primary_retained}. Frozen domain-link support: {secondary_retained}. No production model was changed. Failed candidates remain audit artifacts only, not selected improvements.",
        'All three fine-tuned interventions failed the validation gate against the unchanged, fully trained GRU, even where they slightly improved on the matched fine-tuning control. Test outcomes are disclosed for the predefined audit but cannot rescue a validation-rejected model. The secondary frozen-link analysis separates feature-domain transformation from that fine-tuning regression; its protocol and validation choices were saved before its own test evaluation. It remains exploratory on previously inspected patients.','',
        '## Most promising hypothesis and implementation','',
        'The initial hypothesis was that a feature-specific last-observation carry with learned log-gap decay would help consecutive gaps. The existing strongest GRU uses zero-filled values, masks and log1p(delta), but lacks an explicit last-value path. GapCandidate adds a convex per-feature blend with a causal last value and decay alpha_j * exp(-softplus(rate_j)*log1p(delta)). Carry starts at zero, exactly reproducing the pretrained GRU. It only operates after a prior observation in the current 24-hour window. Observations are copied unchanged. No future observation or cross-patient/window state is used.',
        'Four matched variants: (1) original GRU fine-tuning control, (2) learned gap carry, (3) alternating random20/block20 masks, (4) smooth nonnegative output link. All share the same pretrained GRU, fresh Adam at 0.0001, batch32, gradient clipping5, three epochs, three seeds and fixed random20 validation. Both mask types are drawn for every variant to preserve RNG advancement; only the block-training variant selects block masks for alternating batches. Loss is computed only on deliberately hidden observed targets.',
        'The feature-domain variant uses lower + softplus(latent-lower, beta=20) only for features whose verified registry minimum is zero. It imposes no arbitrary normal range or new upper cap. A one-ULP representable lower-bound guard avoids negative native values from affine inverse roundoff. Signed variables remain unrestricted. This is an explicit modeling link, not a correction to inverse normalization. Raw pre-link normalized predictions are saved for auditing. Other domain violations and unsupported features still trigger the unchanged withholding policy.',
        'Temperature and signed acid/base variables do not receive the nonnegative link. Percentage upper bounds remain checked separately. Upload units are not inferred or converted. Extreme positive biomarkers remain possible: a nonnegative link cannot establish clinical plausibility or calibrated uncertainty.','',
        '## Protocol, controls and statistics','',
        '512 disjoint training patients, 64 validation patients and the same 64 previously audited test patients. Normalization, feature order and all 18 saved test masks (six scenarios x three seeds) match the earlier study exactly. Existing ChronoGAN, reconstruction-only and GRU metrics are reused from verified patient-level sufficient statistics rather than rerunning those checkpoints. New adaptations have three seeds crossed with the same three masks; the secondary frozen model has one unchanged historical training seed.',
        'MAE/RMSE are standardized by frozen training feature standard deviations. Native-unit per-feature errors appear separately in per-feature.csv. Overall 95% percentile intervals resample patients, mask seeds and adaptation seeds; paired comparisons against the matched control preserve training-seed pairing. These pilot intervals are not multiple-testing adjusted, do not represent independent from-scratch confirmation and cannot establish clinical utility.',
        'Coverage is the share of deliberately hidden known values released after policy checks, not coverage of all originally missing clinical cells. Raw, released and withheld subset errors are retained. Invalid rates refer to candidate output BEFORE withholding but AFTER any declared domain link. For the linked models, pre-link head outputs are saved separately as latent diagnostics; lower invalid rate alone is not an accuracy improvement.',
        'Retention was fixed before primary test evaluation: improve validation in all three seeds vs control and mean validation vs unchanged GRU; then paired test MAE CI entirely below zero vs GRU at random20 and block20; no six-scenario mean MAE or RMSE regression. No dashboard promotion regardless. The frozen-link check requires three validation-mask improvements and the same test/aggregate criteria.','',
        '## Validation results','',
        json.dumps(json.loads((OUT/'validation-selection.json').read_text()),indent=2),'',
        'Secondary frozen-link validation:','',json.dumps(secondary['validation'],indent=2),'',
        '## Full performance comparison','']
    for scenario in results:
        lines+=['### '+scenario,'','| Model | Standardized MAE | RMSE | Coverage | Raw invalid | Paired MAE difference vs GRU [95% CI] |','|---|---:|---:|---:|---:|---:|']
        for row in rows:
            if row['scenario']!=scenario:continue
            lines.append(f"| {row['model']} | {row['mae']:.6f} | {row['rmse']:.6f} | {row['coverage']:.2%} | {row['raw_invalid_rate']:.3%} | [{row['paired_mae_difference_vs_gru_ci_low']:.6f}, {row['paired_mae_difference_vs_gru_ci_high']:.6f}] |")
        lines+=['']
    lines+=['## Consecutive-gap detail','',
        'gap-summary.csv reports standardized MAE/RMSE by 1–3, 4–12 and 13+ consecutive unavailable rows, within independent training windows. These counts include both original and artificially hidden missingness when defining a gap; only artificially hidden known values are scored. Gaps include the current missing row and reset at each model window. Gap rows compare new variants with their fine-tuning control. The original GRU comparison for consecutive gaps is provided by the shared block20/block40 scenarios.',
        'All observed measurements were asserted unchanged at inference and all released predictions passed the unchanged mathematical-domain checks. Fully unobserved features remain withheld. No synthetic normal values or zero-filled clinical estimates were released. Zero normalized inputs are still model missing-value placeholders, never ground truth.',
        '## Limitations and next decisions','',
        'Three epochs on 512 patients can regress a fully trained network; unsuccessful fine-tuning is not evidence that a proposed architecture cannot work when properly trained. The zero-initialized carry uses the same small learning rate as the base and may be under-optimized. No hyperparameter was retuned on test scores. Mixed-mask training is selected with random20 validation, which may penalize a block specialist; a new prespecified multi-scenario validation criterion is needed before any such model can be chosen.',
        'The smooth link guarantees a mathematical lower domain for designated missing outputs, not biological plausibility or useful estimates for unobserved features. Extreme positive tails need separate feature-specific target transformations or losses trained and validated from scratch. Long-context modeling must preserve causal access and training/inference consistency. Independent patients and additional from-scratch seeds remain necessary.',
        'No candidate enters production from this pilot. Keep the unchanged GRU as the reference while treating the saved supported flag as permission for further research only, not deployment or clinical use.','',
        '## Reproduction and artifacts','',
        '- models/gap_candidate.py: isolated experimental feature-specific carry and domain link; not imported by dashboard code.',
        '- research_gap_study.py: matched training/evaluation and predefined retention gates.',
        '- check_frozen_domain.py: no-training secondary link-only check.',
        '- report_gap_study.py: CPU-only reporting.',
        '- tests/test_gap_candidate.py: baseline identity, causal carry, observed preservation, finite gradients and real-statistics inverse precision.',
        '- comparison.csv, per-feature.csv, gap-summary.csv and results.json: complete measured outcomes.',
        '- prelink-*.npz: unlinked normalized model outputs before the nonnegative head transformation; decode with frozen statistics, original patient order and 24-hour windows.',
        '- model-hashes.json, protocol.json, validation-selection.json and frozen-domain/protocol.json: provenance and validation choices.',
        '```powershell',r'.\.runtime\python\python.exe run.py research_gap_study.py --device cuda --output runs/gap-feature-study/cuda-replay',r'.\.runtime\python\python.exe run.py check_frozen_domain.py',r'.\.runtime\python\python.exe run.py report_gap_study.py','```',
        'Use --output runs/gap-feature-study/cuda-replay consistently for all three commands to reproduce into a fresh directory. Completed runs are immutable. No existing optimizer checkpoint should be resumed using the new inference-only candidate files.',
        f"Primary CUDA elapsed time {data['seconds']:.1f}s; peak allocated tensor memory {data['peak_cuda_allocated_mib']:.2f} MiB. Secondary CUDA elapsed {secondary['seconds']:.1f}s. These figures exclude other processes and do not represent total GPU VRAM."]
    tree=ET.parse(OUT/'tests.xml').getroot();suites=list(tree.iter('testsuite'));counts={key:sum(int(s.attrib.get(key,0)) for s in suites) for key in ['tests','failures','errors','skipped']};lines+=['','Automated test results: '+str(counts)+'. CUDA hidden for tests only; model experiments used CUDA.']
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in protocol['protected_files'].items())
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8');print('REPORT',OUT/'report.md')

if __name__=='__main__':main()
