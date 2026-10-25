"""Transparent inference-domain checks; never training constraints or normal ranges."""
import hashlib
import json
from pathlib import Path
import numpy as np
from preprocess import FEATURES

REGISTRY_PATH=Path(__file__).resolve().parents[1]/'clinical_variables.json'


def registry():
    value=json.loads(REGISTRY_PATH.read_text(encoding='utf-8'))
    assert list(value['variables'])==FEATURES,'Clinical registry feature order mismatch'
    return value


def validate_stats(stats):
    if stats.get('features')!=FEATURES:raise ValueError('Checkpoint feature names/order differ from the trained feature definition')
    for name in ['mean','std']:
        values=np.asarray(stats[name],dtype=float)
        if values.shape!=(len(FEATURES),) or not np.isfinite(values).all():raise ValueError('Invalid checkpoint normalization '+name)
    if not (np.asarray(stats['std'])>0).all():raise ValueError('Checkpoint normalization standard deviations must be positive')
    if 'count' in stats and len(stats['count'])!=len(FEATURES):raise ValueError('Checkpoint training-observation counts have wrong length')


def validate_upload_schema(path):
    import csv
    with open(path,encoding='utf-8-sig',newline='') as source:
        header=next(csv.reader(source,delimiter='|' if str(path).lower().endswith('.psv') else ','),[])
    duplicates=sorted({name for name in header if name in FEATURES and header.count(name)>1})
    if duplicates:raise ValueError('Ambiguous duplicate clinical columns: '+', '.join(duplicates))


def domain_invalid(values,spec):
    invalid=~np.isfinite(values)
    if spec['minimum'] is not None:invalid|=values<spec['minimum']
    if spec['maximum'] is not None:invalid|=values>spec['maximum']
    return invalid


def apply_policy(raw,raw_lo,raw_hi,raw_std,original,stats,stochastic,mode='withhold',window_length=None):
    if mode not in ['withhold','project','off']:raise ValueError('Unknown output policy')
    info=registry();observed=np.isfinite(original);missing=~observed
    point=raw.copy();lo=raw_lo.copy();hi=raw_hi.copy();spread=raw_std.copy()
    withheld=np.zeros_like(missing);corrected=withheld.copy();interval_withheld=withheld.copy()
    reasons=np.full(original.shape,'',dtype=object);warnings=[];details={}
    raw_invalid=np.zeros_like(missing);observed_invalid=raw_invalid.copy()
    unsupported=np.zeros_like(missing);unit_warnings=[]
    for j,name in enumerate(FEATURES):
        spec=info['variables'][name];invalid=domain_invalid(raw[:,j],spec)&missing[:,j]
        raw_invalid[:,j]=invalid;observed_invalid[:,j]=domain_invalid(original[:,j],spec)&observed[:,j]
        count=stats.get('count',[None]*len(FEATURES))[j]
        untrained=count is not None and count==0
        unanchored=np.zeros(len(original),dtype=bool)
        for start in range(0,len(original),window_length or len(original)):
            stop=start+(window_length or len(original));unanchored[start:stop]=np.cumsum(observed[start:stop,j])==0
        for condition,label in [(invalid,'outside_mathematical_domain'),(missing[:,j]&untrained,'no_training_observations'),(missing[:,j]&unanchored,'no_prior_observation_in_model_window')]:
            for index in np.flatnonzero(condition):reasons[index,j]+=('; ' if reasons[index,j] else '')+label
        unsupported[:,j]=missing[:,j]&(untrained|unanchored)
        if mode=='withhold':withheld[:,j]=invalid|unsupported[:,j]
        elif mode=='project':
            withheld[:,j]=unsupported[:,j]|(missing[:,j]&~np.isfinite(raw[:,j]))
            projected=raw[:,j].copy()
            if spec['minimum'] is not None:projected=np.maximum(projected,spec['minimum'])
            if spec['maximum'] is not None:projected=np.minimum(projected,spec['maximum'])
            point[:,j]=np.where(missing[:,j],projected,original[:,j])
        corrected[:,j]=missing[:,j]&((point[:,j]!=raw[:,j])|withheld[:,j]) if mode!='off' else False
        if mode!='off':point[withheld[:,j],j]=np.nan
        if stochastic:
            bad_interval=missing[:,j]&(domain_invalid(raw_lo[:,j],spec)|domain_invalid(raw_hi[:,j],spec)|~np.isfinite(raw_std[:,j])|(raw_std[:,j]<0)|(raw_lo[:,j]>raw_hi[:,j]))
            if mode=='project':
                for values in (lo,hi):
                    if spec['minimum'] is not None:values[missing[:,j],j]=np.maximum(values[missing[:,j],j],spec['minimum'])
                    if spec['maximum'] is not None:values[missing[:,j],j]=np.minimum(values[missing[:,j],j],spec['maximum'])
                # Raw spread is not a constrained-distribution standard deviation.
                spread[bad_interval,j]=np.nan
                interval_withheld[:,j]=withheld[:,j]|(missing[:,j]&(~np.isfinite(raw_lo[:,j])|~np.isfinite(raw_hi[:,j])|(raw_lo[:,j]>raw_hi[:,j])))
            elif mode=='withhold':interval_withheld[:,j]=withheld[:,j]|bad_interval
        else:interval_withheld[:,j]=missing[:,j]
        if mode!='off':
            for values in (lo,hi,spread):values[interval_withheld[:,j],j]=np.nan
        # Statistical shift warnings are not clinical reference intervals.
        z=(original[:,j]-stats['mean'][j])/stats['std'][j]
        outliers=observed[:,j]&(np.abs(z)>info['statistical_warning_abs_z'])
        if observed_invalid[:,j].any():warnings.append(f'{name}: {int(observed_invalid[:,j].sum())} observed values violate the configured domain; preserved unchanged. Verify units/source data.')
        if outliers.any():warnings.append(f'{name}: {int(outliers.sum())} observed values exceed |training z|>{info["statistical_warning_abs_z"]}; possible distribution/unit shift, not a clinical reference-range assessment.')
        if not spec['unit_verified_for_arbitrary_uploads'] and observed[:,j].any():unit_warnings.append(name)
        details[name]={'raw_invalid_imputed':int(invalid.sum()),'missing':int(missing[:,j].sum()),'withheld':int(withheld[:,j].sum()),'corrected':int(corrected[:,j].sum()),'observed_domain_warnings':int(observed_invalid[:,j].sum()),'patient_observed':int(observed[:,j].sum()),'training_observed':count,'unit':spec['unit'],'unit_note':spec.get('unit_note','Expected source unit; uploaded units are not independently verified.')}
    total=int(missing.sum());invalid_count=int(raw_invalid.sum())
    if invalid_count:warnings.insert(0,f'{invalid_count} raw model estimates violate a configured mathematical domain. This is a model-quality limitation, not an inverse-normalization correction.')
    if withheld.any():warnings.insert(0,f'{int(withheld.sum())} estimates withheld. Raw estimates and reasons remain in the diagnostic exports; no normal values were substituted.')
    if unit_warnings:warnings.append('Upload units cannot be inferred from values alone. Expected source units are recorded; FiO2 has an explicitly unresolved fraction/percent convention.')
    metadata={'mode':mode,'version':info['version'],'registry_sha256':hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest(),
              'missing_cells':total,'raw_invalid_count':invalid_count,'raw_invalid_percentage':100*invalid_count/total if total else 0.,
              'withheld_count':int(withheld.sum()),'corrected_count':int(corrected.sum()),'unsupported_count':int(unsupported.sum()),
              'intervals_withheld_count':int((interval_withheld&missing).sum()),'observed_values_modified':0,'warnings':warnings,
              'per_variable':details,'meaning':'Withheld values are null/blank, not zero. Imputation flags preserve original missingness. Features without training observations or a prior observed anchor in the current model window are withheld conservatively, without consulting future observations. No accuracy or calibration improvement is claimed.',
              'interval_policy':'Raw quantiles are inverse-transformed in the same native units as raw points. In withhold mode invalid/unsupported points or out-of-domain intervals have no displayed interval; raw intervals are retained. Project mode clips endpoints separately and suppresses a misleading unchanged std; it is not a calibrated constrained distribution.'}
    return point,lo,hi,spread,withheld,corrected,interval_withheld,reasons,metadata
