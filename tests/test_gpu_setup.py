import hashlib
import json
import pytest
import torch
import preprocess
from download_full_dataset import valid
from models.generator import Generator
from download_dataset import ROOT

def test_full_cohort_guard(tmp_path,monkeypatch):
    monkeypatch.setattr(preprocess,'DATA',tmp_path)
    with pytest.raises(ValueError,match='40336'):
        preprocess.prepare({'expected_patients':40336})

def test_saved_patient_split_must_match_cohort(tmp_path,monkeypatch):
    monkeypatch.setattr(preprocess,'DATA',tmp_path)
    split=tmp_path/'splits.json'; split.write_text(json.dumps({'train':['duplicate','duplicate'],'val':[],'test':[]}))
    with pytest.raises(ValueError,match='duplicates'):
        preprocess.prepare({'split_file':str(split)})

def test_real_file_integrity_check(tmp_path):
    data=b'HR|ICULOS\n80|1\n'; path=tmp_path/'patient.psv'; path.write_bytes(data)
    row={'size':len(data),'etag':hashlib.md5(data,usedforsecurity=False).hexdigest()}
    assert valid(path,row)
    path.write_bytes(data.replace(b'80',b'90'))
    assert not valid(path,row)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_architecture_preservation_and_prefix():
    model=Generator(4,12,3).cuda().eval()
    x=torch.randn(2,7,4,device='cuda'); m=(torch.rand_like(x)>.5).float()
    d=torch.ones_like(x); valid=torch.ones(2,7,device='cuda'); z=torch.randn(2,7,3,device='cuda')
    out=model(x,m,d,valid,z)
    assert out.is_cuda and torch.equal(out[m.bool()],x[m.bool()])
    modified=x.clone(); modified[:,4:]+=100
    assert torch.equal(out[:,:4],model(modified,m,d,valid,z)[:,:4])

def test_actual_frozen_full_cohort():
    path=ROOT/'data'/'full'/'splits.json'
    if not path.exists(): pytest.skip('Complete real cohort has not been prepared')
    splits=json.loads(path.read_text())
    train,val,test=[set(splits[key]) for key in ['train','val','test']]
    assert not train&val and not train&test and not val&test
    assert (len(train),len(val),len(test))==(28235,6050,6051)
    manifest=json.loads((ROOT/'data'/'official-s3-manifest.json').read_text())
    expected={str((ROOT/'data'/row['relative']).resolve()) for row in manifest}
    assert train|val|test==expected
    stats=json.loads((ROOT/'data'/'full'/'statistics.json').read_text())
    assert stats['features']==preprocess.FEATURES and 'SepsisLabel' not in stats['features']
    assert all(value>0 for value in stats['std'])
