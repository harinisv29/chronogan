"""Record the merged project-local GPU environment without changing the CPU lock."""
from importlib.metadata import distributions
from download_dataset import ROOT

versions={}
for folder in [ROOT/'.packages',ROOT/'.cuda-packages']:
    for distribution in distributions(path=[str(folder)]):
        name=distribution.metadata['Name']
        versions[name.lower()]=(name,distribution.version)
lines=['--extra-index-url https://download.pytorch.org/whl/cu130']
lines.extend(f'{name}=={version}' for name,version in sorted(versions.values(),key=lambda row:row[0].lower()))
(ROOT/'requirements-gpu-lock.txt').write_text('\n'.join(lines)+'\n')
print(f'Saved {len(versions)} package versions; original CPU requirements-lock.txt unchanged.')
