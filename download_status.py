"""Report actual wheel file length without trusting Windows cached directory metadata."""
from pathlib import Path
import tempfile
from install_gpu import NAME,ROOT

for directory in Path(tempfile.gettempdir()).glob('pip-unpack-*'):
    path=directory/NAME
    if path.exists():
        try:
            with path.open('rb') as source:
                source.seek(0,2); received=source.tell()
            print(f'{received:,} / 1,915,224,818 wheel bytes ({received/1915224818:.1%})',flush=True)
        except OSError as exc: print(str(exc))
print('CUDA package files present:',(ROOT/'.cuda-packages'/'torch'/'__init__.py').exists())
log=ROOT/'results'/'gpu'/'pip-install.log'
if log.exists(): print('\n'.join(log.read_text().splitlines()[-5:]))
