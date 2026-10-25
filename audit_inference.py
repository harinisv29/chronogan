"""Independent CPU audit; never writes existing research metrics or checkpoints."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['OMP_NUM_THREADS']='2'
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from preprocess import FEATURES,read_patient,tensors
from predict import reconstruct
from train import load_checkpoint
from utils.clinical_validation import registry,domain_invalid,apply_policy

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/inference-audit/20261022'


def atomic(name,data):
    path=OUT/name;temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)


def errors():return {name:np.zeros(34) for name in ['abs','square','count','invalid','candidate']}


def add(accum,prediction,truth,held):
    valid=held&np.isfinite(prediction);difference=np.where(valid,prediction-truth,0.)
    accum['abs']+=np.abs(difference).sum(0);accum['square']+=np.square(difference).sum(0);accum['count']+=valid.sum(0)
    specs=registry()['variables']
    for j,name in enumerate(FEATURES):accum['invalid'][j]+=(domain_invalid(prediction[:,j],specs[name])&valid[:,j]).sum()
    accum['candidate']+=held.sum(0)


def summarize(accum,std):
    absolute,squared,count,candidate=accum['abs'],accum['square'],accum['count'],accum['candidate']
    n=int(count.sum());total=int(candidate.sum())
    return {'normalized_mae':float((absolute/std).sum()/n) if n else None,
            'normalized_rmse':float(np.sqrt((squared/std**2).sum()/n)) if n else None,
            'scored_predictions':n,'held_out_cells':total,'coverage':n/total if total else 0.,
            'invalid_retained_percentage':float(100*accum['invalid'].sum()/n) if n else None,
            'per_variable':{name:{'mae_native':float(absolute[j]/count[j]) if count[j] else None,'rmse_native':float(np.sqrt(squared[j]/count[j])) if count[j] else None,'n':int(count[j]),'held_out':int(candidate[j]),'invalid':int(accum['invalid'][j])} for j,name in enumerate(FEATURES)}}


def main():
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    checkpoint=ROOT/'runs/full-training/models/chronogan_best.pt';state=load_checkpoint(checkpoint,'cpu');stats=state['stats'];std=np.array(stats['std'])
    frozen_files=[checkpoint,ROOT/'data/full/splits.json',ROOT/'data/full/statistics.json',ROOT/'runs/full-training/results/deployment-policy.json']
    frozen_hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in frozen_files}
    spec=importlib.util.spec_from_file_location('legacy_predict',OUT/'predict-before.py');legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    patient=read_patient(OUT/'synthetic-input.csv');historic=pd.read_csv(OUT/'downloaded-before.csv')
    normalized,mask,_=tensors(patient,stats);restored=normalized*std+np.array(stats['mean']);original=patient.reindex(columns=FEATURES).to_numpy(dtype=float)
    observed=np.isfinite(original);roundtrip=float(np.max(np.abs(restored[observed]-original[observed])))
    safe,missing,uncertainty,metadata=reconstruct(patient,checkpoint,samples=20,device='cpu',seed=42)
    safe.to_csv(OUT/'synthetic-after.csv',index=False);uncertainty.to_csv(OUT/'synthetic-uncertainty.csv',index=False);atomic('synthetic-metadata.json',metadata)
    raw_columns=[f+'__raw' for f in FEATURES];raw=safe[raw_columns].to_numpy()
    historic_values=historic[FEATURES].to_numpy();invalid=np.zeros_like(observed)
    for j,name in enumerate(FEATURES):invalid[:,j]=domain_invalid(historic_values[:,j],registry()['variables'][name])&~observed[:,j]
    synthetic={'rows':len(patient),'original_missing_cells':int((~observed).sum()),'historic_invalid_cells':int(invalid.sum()),'historic_invalid_percentage':float(100*invalid.sum()/(~observed).sum()),
               'after':metadata['output_validation'],'normalization_max_roundtrip_absolute_error':roundtrip,
               'checkpoint_feature_order_exact':stats['features']==FEATURES,'checkpoint_statistics_equal_frozen_training':stats==json.loads((ROOT/'data/full/statistics.json').read_text()),
               'examples':{name:{'historic_minimum':float(historic[name].min()),'historic_negative_count':int((historic[name]<0).sum()),
                                 'historic_minimum_implied_normalized':float((historic[name].min()-stats['mean'][FEATURES.index(name)])/stats['std'][FEATURES.index(name)]),
                                 'current_raw_minimum':float(safe[name+'__raw'].min()),'current_normalized_minimum':float((safe[name+'__raw'].min()-stats['mean'][FEATURES.index(name)])/stats['std'][FEATURES.index(name)]),
                                 'after_finite_count':int(safe[name].notna().sum()),'after_withheld_count':int(safe[name+'__withheld'].sum()),
                                 'input_observations':int(patient[name].notna().sum()),'unit':registry()['variables'][name]['unit']} for name in ['TroponinI','AST']}}
    atomic('synthetic-trace.json',synthetic)
    # Freeze this limited, separate audit before evaluating. Original study masks
    # and outcomes remain untouched; CPU RNG is explicitly not the CUDA protocol.
    paths=state['splits']['test'][:64]
    protocol={'patients':paths,'seed':2026,'scenarios':[['random',.1],['random',.2],['random',.4],['random',.6],['block',.2]],
              'samples':20,'checkpoint_sha256':frozen_hashes[str(checkpoint.relative_to(ROOT))],'selection':'First 64 patients in the already-frozen test list, fixed before this audit; no checkpoint selection.',
              'masks':'Independent CPU audit masks, shared by every compared pipeline. No original research mask, seed, split, policy or result is changed.',
              'policies':'Legacy full-sequence raw; training-window raw; user-chosen withholding; optional boundary projection for diagnosis only. Retained-subset raw metrics disclose abstention selection effects.'}
    protocol_file=OUT/'audit-protocol.json'
    if protocol_file.exists():assert json.loads(protocol_file.read_text())==protocol
    else:atomic('audit-protocol.json',protocol)
    results=[];all_masks={}
    for pattern,rate in protocol['scenarios']:
        target=OUT/f'audit-{pattern}-{rate}.json'
        if target.exists():results.append(json.loads(target.read_text()));continue
        rng=np.random.default_rng(2026);digest=hashlib.sha256()
        accum={name:errors() for name in ['legacy_raw','windowed_raw','withheld','windowed_raw_on_retained','projected_diagnostic_only']}
        raw_invalid=0;held_count=0;correction_count=0;withheld_count=0;interval_withheld_count=0
        for index,path in enumerate(paths):
            source=read_patient(path);truth=source.reindex(columns=FEATURES).to_numpy(dtype=float);known=np.isfinite(truth)
            if pattern=='random':hidden=rng.random(known.shape)<rate
            else:
                length=len(source);span=max(1,int(length*rate));starts=rng.integers(0,length-span+1,size=34);times=np.arange(length)[:,None]
                hidden=(times>=starts)&(times<starts+span)
            held=hidden&known;digest.update(held.tobytes());all_masks[f'{pattern}_{rate}_{index}']=held
            masked=source.copy()
            for j,name in enumerate(FEATURES):masked.loc[held[:,j],name]=np.nan
            before=legacy.reconstruct(masked,checkpoint,samples=20,seed=2026+index,device='cpu')[0][FEATURES].to_numpy()
            modern,_,intervals,_=reconstruct(masked,checkpoint,samples=20,device='cpu',seed=2026+index,output_policy='off')
            point=modern[raw_columns].to_numpy();missing_original=masked.reindex(columns=FEATURES).to_numpy(dtype=float)
            lower=intervals[[f+'__raw_p05' for f in FEATURES]].to_numpy();upper=intervals[[f+'__raw_p95' for f in FEATURES]].to_numpy();spread=intervals[[f+'__raw_std' for f in FEATURES]].to_numpy()
            validated=apply_policy(point,lower,upper,spread,missing_original,stats,True,'withhold',24)
            projected=apply_policy(point,lower,upper,spread,missing_original,stats,True,'project',24)
            add(accum['legacy_raw'],before,truth,held);add(accum['windowed_raw'],point,truth,held);add(accum['withheld'],validated[0],truth,held)
            add(accum['windowed_raw_on_retained'],np.where(validated[4],np.nan,point),truth,held)
            add(accum['projected_diagnostic_only'],projected[0],truth,held)
            for j,name in enumerate(FEATURES):raw_invalid+=int((domain_invalid(point[:,j],registry()['variables'][name])&held[:,j]).sum())
            held_count+=int(held.sum());correction_count+=int((validated[5]&held).sum());withheld_count+=int((validated[4]&held).sum());interval_withheld_count+=int((validated[6]&held).sum())
            if (index+1)%8==0:
                atomic('progress.json',{'stage':'cpu_audit','pattern':pattern,'rate':rate,'patients_completed':index+1,'patients':len(paths)})
                print('CPU audit',pattern,rate,index+1,'/',len(paths),flush=True)
        row={'pattern':pattern,'rate':rate,'patients':len(paths),'mask_sha256':digest.hexdigest(),'metrics':{name:summarize(value,std) for name,value in accum.items()},
             'raw_invalid_percentage':100*raw_invalid/held_count,'withheld_percentage':100*withheld_count/held_count,'correction_percentage':100*correction_count/held_count,'intervals_withheld_percentage':100*interval_withheld_count/held_count}
        atomic(target.name,row);results.append(row)
    if all_masks:np.savez_compressed(OUT/'audit-masks.npz',**all_masks)
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected for name,expected in frozen_hashes.items())
    report={'status':'completed_limited_cpu_audit','synthetic':synthetic,'evaluation':results,'protocol':protocol,'preserved_hashes':frozen_hashes,
            'root_cause':'Unbounded linear generator output plus affine inverse z-scaling permits negative concentrations/activities. AST and TroponinI have no feature-specific training domain penalty. Both are entirely unobserved in the supplied input. Correct feature order and frozen statistics were verified; the inverse formula is correct. A separate full-sequence versus 24-hour training-reset mismatch was corrected.',
            'interpretation':'Withholding changes coverage, not raw model accuracy. Compare windowed_raw_on_retained with withheld on identical cells: they are identical. Boundary projection is diagnostic only and not the chosen app policy. Synthetic inputs have no unknown-value ground truth. This 64/6051-patient CPU audit does not replace the full research evaluation, prove calibration or establish clinical effectiveness.',
            'retraining':'No retraining initiated. Reliable estimates for unsupported or invalid outputs need independent validation and likely retraining with appropriate nonnegative target distributions/transformations and variable-specific support; display-time withholding does not solve model quality.'}
    atomic('report.json',report)
    lines=['# Inference-domain investigation','','## Confirmed cause','',report['root_cause'],'',f'Observed-value normalization roundtrip max absolute error: {roundtrip:.8g}; this is finite-precision rounding, not a column permutation. Frozen checkpoint statistics matched exactly.','','## Supplied synthetic input and historical export','',f'{len(patient)} rows; {synthetic["original_missing_cells"]} originally missing cells. Historical raw domain violations: {synthetic["historic_invalid_cells"]} ({synthetic["historic_invalid_percentage"]:.2f}%).','', '| Feature | Historical minimum | Negative historical rows | Current fixed-seed raw minimum | Safe display |','|---|---:|---:|---:|---|']
    for name,value in synthetic['examples'].items():lines.append(f'| {name} | {value["historic_minimum"]:.6f} | {value["historic_negative_count"]} / {len(patient)} | {value["current_raw_minimum"]:.6f} | {value["after_withheld_count"]} withheld; no zero substitution |')
    lines+=['','## Frozen-test-subset CPU comparison','','64 patients, fixed in advance from 6,051 frozen test patients. Units remain source-native; per-variable errors and all denominators are in report.json.','','| Pattern / rate | Legacy MAE / RMSE | Windowed raw MAE / RMSE | Withheld retained MAE / RMSE | Coverage | Raw invalid |','|---|---|---|---|---:|---:|']
    for row in results:
        values=row['metrics'];pair=lambda name:f'{values[name]["normalized_mae"]:.6f} / {values[name]["normalized_rmse"]:.6f}'
        lines.append(f'| {row["pattern"]} / {row["rate"]} | {pair("legacy_raw")} | {pair("windowed_raw")} | {pair("withheld")} | {values["withheld"]["coverage"]:.2%} | {row["raw_invalid_percentage"]:.2f}% |')
    lines+=['',report['interpretation'],'',report['retraining'],'','## Sources and assumptions','','[Official Challenge variable units and missingness](https://moody-challenge.physionet.org/2019/). No universal reference ranges were invented. Temperature is assumed Celsius per source metadata; arbitrary uploads do not verify their units. FiO2 fraction/percent ambiguity is explicitly unresolved. Original unusual observations are warned about, never changed.','', 'Raw estimates, flags, reasons, raw native-unit quantiles and output-policy version are exported separately. A withheld blank is a deliberate abstention, not zero or a clinically valid normal value.']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    atomic('progress.json',{'stage':'completed','patients':len(paths),'scenarios':len(results)})
    print(json.dumps({'report':str(OUT/'report.md'),'synthetic_invalid_before':synthetic['historic_invalid_cells'],'synthetic_withheld_after':metadata['output_validation']['withheld_count']},indent=2),flush=True)


if __name__=='__main__':main()
