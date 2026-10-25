"""Reusable normalized windows derived from the frozen real cohort, never source data."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from preprocess import PatientDataset,FEATURES

class CachedWindows(Dataset):
    def __init__(self,folder):
        self.arrays=[np.load(Path(folder)/f'{name}.npy',mmap_mode='c') for name in ['x','mask','delta','valid','labels']]
    def __len__(self): return len(self.arrays[0])
    def __getitem__(self,index): return tuple(torch.from_numpy(a[index]) for a in self.arrays)

def windows(paths,stats,length,cache_dir=None,split='train'):
    if cache_dir is None: return PatientDataset(paths,stats,length)
    fingerprint=hashlib.sha256(json.dumps({'paths':paths,'stats':stats,'length':length,'features':FEATURES},sort_keys=True).encode()).hexdigest()
    folder=Path(cache_dir)/split; folder.mkdir(parents=True,exist_ok=True)
    marker=folder/'ready.json'
    if marker.exists():
        assert json.loads(marker.read_text())['fingerprint']==fingerprint,'Window cache cohort/statistics mismatch'
        return CachedWindows(folder)
    print(f'Building {split} window cache from {len(paths)} existing real patients',flush=True)
    source=PatientDataset(paths,stats,length)
    for j,name in enumerate(['x','mask','delta','valid','labels']):
        np.save(folder/f'{name}.npy',np.stack([sample[j] for sample in source.windows]))
    marker.write_text(json.dumps({'fingerprint':fingerprint,'windows':len(source)}))
    print(f'Cached {split}: {len(source)} windows',flush=True)
    return CachedWindows(folder)
